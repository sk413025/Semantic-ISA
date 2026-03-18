"""
Generate test audio files for ASIR integration evaluation.

Data sources:
  Speech: Gemini TTS (GEMINI_API_KEY) → Chinese daily conversation
          Fallback: sine wave + amplitude modulation (speech-like envelope)
  Noise:  Synthetic noise with spectral shaping (numpy + scipy)
          Upgrade: place DEMAND WAV files in examples/audio/noise/
  Mixing: numpy at controlled SNR
  Reverb: exponential decay RIR (numpy)

Usage:
  PYTHONUTF8=1 python -X utf8 -m asir.eval.generate_audio

Output:
  examples/audio/speech/*.wav   (3 TTS speech clips, 16kHz mono)
  examples/audio/scenarios/*.wav (8 stereo 16kHz WAV, one per eval scenario)

DEMAND dataset (optional, CC-BY 4.0):
  Place 16kHz channel-01 WAV files in examples/audio/noise/:
    PRESTO_ch01.wav, TCAR_ch01.wav, STRAFFIC_ch01.wav, etc.
  Download from: https://zenodo.org/records/1227121
"""
import os
import sys
import wave
import struct
import numpy as np
from pathlib import Path

SR = 16000
DURATION_S = 5.0

BASE_DIR = Path(__file__).parent.parent.parent
AUDIO_DIR = BASE_DIR / "examples" / "audio"
SPEECH_DIR = AUDIO_DIR / "speech"
SCENARIO_DIR = AUDIO_DIR / "scenarios"
NOISE_DIR = AUDIO_DIR / "noise"

# DEMAND mapping: our scenario → DEMAND folder name
DEMAND_MAP = {
    "restaurant_dinner": "PRESTO",
    "church_ceremony": "OHALLWAY",
    "quiet_library": "OOFFICE",
    "street_phone_call": "STRAFFIC",
    "supermarket_shopping": "PSTATION",
    "car_conversation": "TCAR",
    "noisy_cafe_complaint": "PCAFETER",
    "severe_loss_quiet_home": "DLIVING",
}

# Noise type for synthetic fallback
NOISE_TYPE_MAP = {
    "restaurant_dinner": "babble",
    "church_ceremony": "ambient",
    "quiet_library": "quiet",
    "street_phone_call": "traffic",
    "supermarket_shopping": "crowd",
    "car_conversation": "car",
    "noisy_cafe_complaint": "babble",
    "severe_loss_quiet_home": "quiet",
}

# Scenario parameters (must match examples.py)
# energy_db controls the overall level of the output WAV:
#   comp_extract_full_features uses: 10*log10(mean(s²)) + 94
#   So target RMS amplitude = 10^((energy_db - 94) / 20)
SCENARIOS = [
    {"name": "restaurant_dinner", "snr_db": 3.0, "rt60_s": 0.7, "energy_db": 72.0},
    {"name": "church_ceremony", "snr_db": 12.0, "rt60_s": 2.5, "energy_db": 60.0},
    {"name": "quiet_library", "snr_db": 30.0, "rt60_s": 0.6, "energy_db": 40.0},
    {"name": "street_phone_call", "snr_db": -2.0, "rt60_s": 0.1, "energy_db": 80.0},
    {"name": "supermarket_shopping", "snr_db": 8.0, "rt60_s": 1.0, "energy_db": 68.0},
    {"name": "car_conversation", "snr_db": 6.0, "rt60_s": 0.15, "energy_db": 70.0},
    {"name": "noisy_cafe_complaint", "snr_db": 2.0, "rt60_s": 0.6, "energy_db": 75.0},
    {"name": "severe_loss_quiet_home", "snr_db": 25.0, "rt60_s": 0.4, "energy_db": 50.0},
]

SPEECH_SENTENCES = [
    "用正常語速說：你好，今天晚餐想吃什麼？我們可以去附近那家餐廳看看。",
    "用正常語速說：請問這個多少錢？我想買兩個，可以算便宜一點嗎？",
    "用正常語速說：不好意思，請問到捷運站怎麼走？走路大概要多久？",
]


# ===== Speech Generation =====

def _load_env():
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())


def generate_speech_gemini():
    """Generate Chinese speech clips via Gemini TTS (24kHz → resample to 16kHz)."""
    _load_env()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  GEMINI_API_KEY not set, using fallback speech")
        return generate_speech_fallback()

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("  google-genai not installed, using fallback speech")
        return generate_speech_fallback()

    client = genai.Client(api_key=api_key)
    speeches = []

    for i, sentence in enumerate(SPEECH_SENTENCES):
        out_path = SPEECH_DIR / f"speech_{i+1}.wav"
        if out_path.exists():
            print(f"  speech_{i+1}.wav exists, loading")
            speeches.append(_load_mono_wav(out_path))
            continue

        print(f"  Generating speech_{i+1} via Gemini TTS...", end=" ", flush=True)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=sentence,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Kore"
                            )
                        )
                    ),
                ),
            )
            pcm_data = response.candidates[0].content.parts[0].inline_data.data
            # Decode 24kHz PCM16 → float32
            n_samples = len(pcm_data) // 2
            samples_24k = np.array(
                struct.unpack(f"<{n_samples}h", pcm_data), dtype=np.float32
            ) / 32768.0

            # Resample 24kHz → 16kHz
            from scipy.signal import resample
            n_16k = int(len(samples_24k) * SR / 24000)
            samples_16k = resample(samples_24k, n_16k).astype(np.float32)

            # Save
            _save_mono_wav(out_path, samples_16k, SR)
            speeches.append(samples_16k)
            print(f"{len(samples_16k)/SR:.2f}s")

        except Exception as e:
            print(f"ERROR: {e}")
            speeches.append(_generate_speech_like_signal(DURATION_S, SR))

    return speeches


def generate_speech_fallback():
    """Fallback: generate speech-like signals using amplitude-modulated tones."""
    speeches = []
    for i in range(3):
        out_path = SPEECH_DIR / f"speech_{i+1}.wav"
        if out_path.exists():
            speeches.append(_load_mono_wav(out_path))
            continue
        sig = _generate_speech_like_signal(DURATION_S, SR, seed=i)
        _save_mono_wav(out_path, sig, SR)
        speeches.append(sig)
    return speeches


def _generate_speech_like_signal(duration_s, sr, seed=0):
    """Amplitude-modulated harmonic signal that mimics speech envelope."""
    rng = np.random.default_rng(seed + 100)
    n = int(duration_s * sr)
    t = np.arange(n) / sr

    # Fundamental + harmonics (speech-like spectrum)
    f0 = 150 + rng.random() * 50  # 150-200 Hz
    sig = np.zeros(n)
    for h in range(1, 8):
        amp = 1.0 / h
        sig += amp * np.sin(2 * np.pi * f0 * h * t + rng.random() * 2 * np.pi)

    # Syllabic amplitude modulation (~4 Hz)
    env = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 4 * t + rng.random() * np.pi))
    # Add some pauses
    pause_mask = np.ones(n)
    for _ in range(3):
        start = rng.integers(0, n - sr // 2)
        pause_mask[start:start + sr // 4] = 0.05
    sig = sig * env * pause_mask
    return (sig / (np.max(np.abs(sig)) + 1e-10)).astype(np.float32)


# ===== Noise Generation =====

def get_noise(scenario_name, duration_s, sr):
    """Get noise: DEMAND file if available, else synthetic."""
    demand_name = DEMAND_MAP.get(scenario_name, "")
    demand_path = NOISE_DIR / f"{demand_name}_ch01.wav"
    if demand_path.exists():
        print(f"    Using DEMAND: {demand_path.name}")
        noise = _load_mono_wav(demand_path)
        n = int(duration_s * sr)
        if len(noise) >= n:
            return noise[:n]
        return np.tile(noise, (n // len(noise)) + 1)[:n]

    noise_type = NOISE_TYPE_MAP.get(scenario_name, "white")
    return _generate_synthetic_noise(noise_type, duration_s, sr, seed=hash(scenario_name))


def _generate_synthetic_noise(noise_type, duration_s, sr, seed=42):
    """Generate spectrally-shaped noise matching environment type."""
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(abs(seed) % (2**31))
    n = int(duration_s * sr)

    if noise_type == "babble":
        # Multi-talker babble: sum of filtered, modulated signals
        noise = np.zeros(n)
        for i in range(6):
            raw = rng.normal(0, 1, n)
            b, a = butter(4, 4000 / (sr / 2))
            filtered = lfilter(b, a, raw)
            rate = 3 + rng.random() * 3
            mod = 0.3 + 0.7 * np.abs(np.sin(
                2 * np.pi * rate * np.arange(n) / sr + rng.random() * np.pi
            ))
            noise += filtered * mod * (0.3 + 0.7 * rng.random())
        return (noise / (np.max(np.abs(noise)) + 1e-10)).astype(np.float32)

    elif noise_type == "traffic":
        # Low-frequency rumble + impulses (horns, engines)
        raw = rng.normal(0, 1, n)
        b, a = butter(3, 500 / (sr / 2))
        rumble = lfilter(b, a, raw)
        for _ in range(15):
            t = rng.integers(0, n - 200)
            rumble[t:t + 200] += rng.normal(0, 3, 200)
        return (rumble / (np.max(np.abs(rumble)) + 1e-10)).astype(np.float32)

    elif noise_type == "car":
        # Steady engine + wind noise
        raw = rng.normal(0, 1, n)
        b, a = butter(3, 800 / (sr / 2))
        engine = lfilter(b, a, raw)
        rpm_mod = 1 + 0.08 * np.sin(2 * np.pi * 1.5 * np.arange(n) / sr)
        noise = engine * rpm_mod
        # Add wind (high-pass)
        b2, a2 = butter(2, 2000 / (sr / 2), btype='high')
        wind = lfilter(b2, a2, rng.normal(0, 0.3, n))
        noise += wind
        return (noise / (np.max(np.abs(noise)) + 1e-10)).astype(np.float32)

    elif noise_type == "crowd":
        # Dense babble + footsteps + PA-like tones
        noise = np.zeros(n)
        for i in range(10):
            raw = rng.normal(0, 1, n)
            b, a = butter(4, 3500 / (sr / 2))
            filtered = lfilter(b, a, raw)
            rate = 2 + rng.random() * 4
            mod = 0.2 + 0.8 * np.abs(np.sin(
                2 * np.pi * rate * np.arange(n) / sr
            ))
            noise += filtered * mod * (0.2 + 0.8 * rng.random())
        return (noise / (np.max(np.abs(noise)) + 1e-10)).astype(np.float32)

    elif noise_type == "quiet":
        # Very low-level ambient
        raw = rng.normal(0, 1, n)
        b, a = butter(2, 500 / (sr / 2))
        return (lfilter(b, a, raw) * 0.02).astype(np.float32)

    elif noise_type == "ambient":
        # Generic indoor ambient (HVAC-like)
        raw = rng.normal(0, 1, n)
        b, a = butter(2, 300 / (sr / 2))
        return (lfilter(b, a, raw) * 0.1).astype(np.float32)

    else:
        return rng.normal(0, 1, n).astype(np.float32)


# ===== Audio Processing =====

def mix_at_snr(speech, noise, target_snr_db):
    """Mix speech and noise at target SNR (dB)."""
    speech_power = np.mean(speech ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_ratio = 10 ** (target_snr_db / 10)
    scale = np.sqrt(speech_power / (noise_power * target_ratio))
    return speech + noise * scale


def apply_reverb(signal, rt60_s, sr):
    """Apply synthetic reverb using exponential decay RIR."""
    if rt60_s < 0.2:
        return signal  # Skip for very short RT60

    n_rir = int(rt60_s * sr)
    rng = np.random.default_rng(7)
    rir = rng.normal(0, 1, n_rir)
    # Exponential decay: -60dB at rt60_s
    decay = np.exp(-6.9 * np.arange(n_rir) / n_rir)  # ln(1000)≈6.9
    rir = rir * decay
    rir[0] = 1.0  # Direct path dominates
    rir = rir / np.sqrt(np.sum(rir ** 2))

    result = np.convolve(signal, rir, mode='same')
    return result.astype(np.float32)


# ===== WAV I/O =====

def _save_mono_wav(path, samples, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def _save_stereo_wav(path, samples, sr):
    """Save mono signal as stereo WAV (duplicate to both channels)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = np.clip(samples, -1, 1)
    # Interleave: L, R, L, R, ...
    stereo = np.empty(len(mono) * 2, dtype=np.float32)
    stereo[0::2] = mono
    stereo[1::2] = mono
    pcm16 = (stereo * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def _load_mono_wav(path):
    """Load WAV file as float32 mono array."""
    with wave.open(str(path), 'rb') as wf:
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        raw = wf.readframes(n_frames)
        samples = np.array(
            struct.unpack(f"<{n_frames * n_channels}h", raw),
            dtype=np.float32,
        ) / 32768.0
        if n_channels > 1:
            samples = samples[::n_channels]  # Take first channel
        return samples


# ===== Main =====

def generate_all():
    """Generate all scenario audio files."""
    print("=" * 60)
    print("  ASIR Audio Generator — Integration Test Scenarios")
    print("=" * 60)

    # Ensure directories
    SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate speech
    print("\n  [1/3] Generating speech clips...")
    speeches = generate_speech_gemini()
    print(f"  Got {len(speeches)} speech clips")

    # Step 2: Generate scenarios
    print(f"\n  [2/3] Mixing {len(SCENARIOS)} scenarios...")
    for sc in SCENARIOS:
        name = sc["name"]
        out_path = SCENARIO_DIR / f"{name}.wav"
        print(f"\n  {name}:")

        # Pick a speech clip (cycle through available)
        idx = SCENARIOS.index(sc) % len(speeches)
        speech = speeches[idx]

        # Ensure duration
        n = int(DURATION_S * SR)
        if len(speech) < n:
            speech = np.tile(speech, (n // len(speech)) + 1)[:n]
        else:
            speech = speech[:n]

        # Get noise
        noise = get_noise(name, DURATION_S, SR)

        # Mix at target SNR
        mixed = mix_at_snr(speech, noise, sc["snr_db"])
        print(f"    Mixed at SNR={sc['snr_db']}dB")

        # Apply reverb
        if sc["rt60_s"] >= 0.2:
            mixed = apply_reverb(mixed, sc["rt60_s"], SR)
            print(f"    Applied reverb RT60={sc['rt60_s']}s")

        # Energy-aware normalization:
        # comp_extract_full_features uses: energy_db = 10*log10(mean(s²)) + 94
        # So target RMS = 10^((target_energy_db - 94) / 20)
        target_energy = sc.get("energy_db", 70.0)
        target_rms = 10 ** ((target_energy - 94) / 20)
        current_rms = np.sqrt(np.mean(mixed ** 2)) + 1e-10
        mixed = (mixed * (target_rms / current_rms)).astype(np.float32)
        # Clip to prevent overflow (but preserve relative levels)
        if np.max(np.abs(mixed)) > 0.99:
            mixed = mixed * (0.95 / np.max(np.abs(mixed)))

        # Save as stereo WAV
        _save_stereo_wav(out_path, mixed, SR)
        print(f"    Saved: {out_path.name} ({os.path.getsize(out_path) / 1024:.0f} KB)")

    # Step 3: Summary
    print(f"\n  [3/3] Summary")
    total = sum(
        os.path.getsize(SCENARIO_DIR / f"{sc['name']}.wav")
        for sc in SCENARIOS
        if (SCENARIO_DIR / f"{sc['name']}.wav").exists()
    )
    print(f"  {len(SCENARIOS)} scenario WAVs, total {total / 1024:.0f} KB")
    print(f"  Output: {SCENARIO_DIR}")


def main():
    generate_all()


if __name__ == "__main__":
    main()
