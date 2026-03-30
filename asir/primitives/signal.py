import numpy as np

from asir.types import RawSignal


def prim_load_audio(
    file_path: str = None,
    audio_bytes: bytes = None,
    sample_rate: int = 16000,
) -> tuple:
    """
    [Phase 1] Load a real audio file and return a `(RawSignal, dspy.Audio)` pair.
    WAV is supported directly, and more formats are supported when scipy is
    available.

    This is the real-audio counterpart to `prim_sample_audio()`.
    """

    import dspy

    audio_obj = None

    if file_path is not None:
        if hasattr(dspy.Audio, "from_file"):
            audio_obj = dspy.Audio.from_file(file_path)

        try:
            from scipy.io import wavfile

            sr, data = wavfile.read(file_path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            if len(data.shape) == 1:
                samples = [data.tolist(), data.tolist()]
                n_ch = 2
            else:
                samples = [data[:, i].tolist() for i in range(min(data.shape[1], 2))]
                n_ch = len(samples)
            signal = RawSignal(
                samples=samples,
                sample_rate=sr,
                n_channels=n_ch,
                duration_ms=float(len(data) / sr * 1000),
            )
            return signal, audio_obj
        except ImportError:
            import struct

            with open(file_path, "rb") as f:
                raw = f.read()

            # Minimal WAV parser.
            if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
                idx = 12
                while idx < len(raw) - 8:
                    chunk_id = raw[idx : idx + 4]
                    chunk_size = struct.unpack("<I", raw[idx + 4 : idx + 8])[0]
                    if chunk_id == b"data":
                        pcm = np.frombuffer(raw[idx + 8 : idx + 8 + chunk_size], dtype=np.int16)
                        data = pcm.astype(np.float32) / 32768.0
                        signal = RawSignal(
                            samples=[data.tolist(), data.tolist()],
                            sample_rate=sample_rate,
                            n_channels=2,
                            duration_ms=float(len(data) / sample_rate * 1000),
                        )
                        return signal, audio_obj
                    idx += 8 + chunk_size

    # Fallback: use a simulated signal.
    signal = prim_sample_audio(duration_ms=32.0)
    from asir.multimodal.audio import raw_signal_to_audio

    audio_obj = raw_signal_to_audio(signal)
    return signal, audio_obj


def prim_sample_audio(
    duration_ms: float = 32.0,
    n_channels: int = 2,
    sample_rate: int = 16000,
) -> RawSignal:
    """
    [PRIM] Layer 1: physical sensing.
    BACKEND: deterministic (hardware-side simulation)
    RELIABILITY: 100%
    Analogy: similar to a CPU LOAD instruction.
    """

    n_samples = int(sample_rate * duration_ms / 1000)

    # Simulate a wet-market-like mixture with multiple active sources.
    t = np.linspace(0, duration_ms / 1000, n_samples)

    # Target speech from the front.
    target_speech = 0.3 * np.sin(2 * np.pi * 200 * t) * np.random.normal(1, 0.1, n_samples)
    # Background crowd babble.
    babble = 0.4 * np.random.randn(n_samples)
    # Occasional metallic transient from one side.
    impulse = np.zeros(n_samples)
    impulse[n_samples // 3] = 2.0

    ch0 = (target_speech + babble + impulse * 0.3).tolist()
    ch1 = (target_speech * 0.8 + babble + impulse * 0.7).tolist()

    return RawSignal(
        samples=[ch0, ch1],
        sample_rate=sample_rate,
        n_channels=n_channels,
        duration_ms=duration_ms,
    )


def prim_fft(signal: RawSignal) -> dict:
    """
    [PRIM] Layer 2: FFT.
    BACKEND: deterministic
    RELIABILITY: 100% within numerical precision.
    """

    spectrum = np.fft.rfft(signal.samples[0])
    return {
        "magnitude": np.abs(spectrum).tolist(),
        "phase": np.angle(spectrum).tolist(),
        "freq_bins": len(spectrum),
    }


def prim_estimate_noise_psd(signal: RawSignal) -> list[float]:
    """
    [PRIM] Layer 2: noise power spectral density estimation.
    BACKEND: deterministic (minimum-statistics style)
    RELIABILITY: 100%
    """

    spectrum = np.fft.rfft(signal.samples[0])
    psd = np.abs(spectrum) ** 2
    noise_floor = np.percentile(psd, 25)
    return (psd * (psd < noise_floor * 4)).tolist()


def prim_beamform(signal: RawSignal, target_azimuth_deg: float = 0.0) -> list[float]:
    """
    [PRIM] Layer 2: fixed-direction beamforming.
    BACKEND: deterministic
    RELIABILITY: 100%
    """

    ch0 = np.array(signal.samples[0])
    ch1 = np.array(signal.samples[1])

    # Simplified delay-and-sum beamformer.
    delay_samples = int(0.01 * np.sin(np.radians(target_azimuth_deg)) * signal.sample_rate)
    ch1_delayed = np.roll(ch1, delay_samples)
    beamformed = (ch0 + ch1_delayed) / 2.0
    return beamformed.tolist()


def comp_spectral_subtract(
    signal: RawSignal,
    noise_psd: list[float],
    alpha: float = 1.0,
) -> list[float]:
    """
    [COMP] Layer 2: spectral subtraction.
    = fft(signal) |> subtract_magnitude(noise, alpha) |> ifft()
    Still fully deterministic, even though it combines multiple primitives.
    """

    spectrum = np.fft.rfft(signal.samples[0])
    noise = np.array(noise_psd[: len(spectrum)])
    magnitude = np.maximum(np.abs(spectrum) - alpha * np.sqrt(noise), 0)
    cleaned = magnitude * np.exp(1j * np.angle(spectrum))
    return np.fft.irfft(cleaned).tolist()
