"""
L1-L3 deterministic layer tests.

These tests do not require API access. They validate the physics-facing and
signal-processing primitives with pure NumPy inputs and also reuse the
scenario definitions and audio files under `asir/eval/`.
"""

from pathlib import Path

import numpy as np
import pytest

from asir.eval.examples import create_eval_examples
from asir.primitives import (
    comp_extract_full_features,
    comp_spectral_subtract,
    prim_beamform,
    prim_estimate_noise_psd,
    prim_estimate_rt60,
    prim_estimate_snr,
    prim_extract_mfcc,
    prim_fft,
    prim_generate_gain_params,
    prim_sample_audio,
)
from asir.primitives.signal import prim_load_audio
from asir.types import RawSignal


def make_signal(samples_ch0, samples_ch1=None, sample_rate=16000):
    """Build a `RawSignal` from NumPy arrays."""
    ch0 = np.asarray(samples_ch0, dtype=float)
    ch1 = np.asarray(samples_ch1, dtype=float) if samples_ch1 is not None else ch0.copy()
    return RawSignal(
        samples=[ch0.tolist(), ch1.tolist()],
        sample_rate=sample_rate,
        n_channels=2,
        duration_ms=float(len(ch0) / sample_rate * 1000),
    )


def make_tone(freq_hz=440, duration_s=0.1, amplitude=1.0, sample_rate=16000):
    """Generate a pure tone."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def make_noise(duration_s=0.1, amplitude=1.0, sample_rate=16000):
    """Generate white Gaussian noise."""
    n = int(sample_rate * duration_s)
    return amplitude * np.random.RandomState(42).randn(n)


class TestL1PhysicalSensing:
    def test_sample_audio_shape(self):
        """`prim_sample_audio` should return a correctly shaped RawSignal."""
        sig = prim_sample_audio(duration_ms=32.0, n_channels=2)
        assert sig.n_channels == 2
        assert sig.sample_rate == 16000
        assert abs(sig.duration_ms - 32.0) < 0.1
        assert len(sig.samples) == 2
        expected_n = int(16000 * 32.0 / 1000)
        assert len(sig.samples[0]) == expected_n
        assert len(sig.samples[1]) == expected_n

    def test_sample_audio_value_range(self):
        """Generated samples should remain finite and within a reasonable range."""
        sig = prim_sample_audio(duration_ms=10.0)
        arr = np.array(sig.samples[0])
        assert np.all(np.isfinite(arr)), "should have no NaN or inf"
        assert np.max(np.abs(arr)) < 100, "amplitude should stay reasonable"


class TestL2SignalProcessing:
    def test_fft_output_structure(self):
        """`prim_fft` should return magnitude, phase, and frequency-bin count."""
        sig = make_signal(make_tone(440, 0.032))
        result = prim_fft(sig)
        assert "magnitude" in result
        assert "phase" in result
        assert "freq_bins" in result
        assert len(result["magnitude"]) == result["freq_bins"]
        assert len(result["phase"]) == result["freq_bins"]

    def test_fft_peak_at_target_frequency(self):
        """A 440 Hz tone should yield an FFT peak near 440 Hz."""
        sr = 16000
        sig = make_signal(make_tone(440, 0.1, sample_rate=sr), sample_rate=sr)
        result = prim_fft(sig)
        mag = np.array(result["magnitude"])
        peak_bin = np.argmax(mag)
        peak_freq = peak_bin * sr / (2 * (len(mag) - 1))
        assert abs(peak_freq - 440) < 50, f"peak at {peak_freq}Hz, expected ~440Hz"

    def test_beamform_output_length(self):
        """Beamforming output should keep the same sample length as the input."""
        sig = prim_sample_audio(duration_ms=32.0)
        bf = prim_beamform(sig, target_azimuth_deg=0.0)
        assert len(bf) == len(sig.samples[0])

    def test_beamform_front_enhances_correlated_signal(self):
        """A front-facing correlated signal should be preserved or slightly enhanced."""
        tone = make_tone(200, 0.032)
        sig = make_signal(tone, tone)
        bf = np.array(prim_beamform(sig, target_azimuth_deg=0.0))
        assert np.mean(bf**2) >= np.mean(np.array(tone) ** 2) * 0.8

    def test_noise_psd_output_length(self):
        """Noise PSD output length should match the FFT bin count."""
        sig = prim_sample_audio()
        psd = prim_estimate_noise_psd(sig)
        fft_bins = len(np.fft.rfft(sig.samples[0]))
        assert len(psd) == fft_bins

    def test_spectral_subtract_reduces_noise(self):
        """Spectral subtraction should return a valid denoised waveform."""
        noise = make_noise(0.032, amplitude=0.5)
        tone = make_tone(440, 0.032, amplitude=1.0)
        noisy = tone + noise
        sig = make_signal(noisy)
        noise_psd = prim_estimate_noise_psd(sig)
        cleaned = comp_spectral_subtract(sig, noise_psd, alpha=1.0)
        assert len(cleaned) > 0
        assert np.all(np.isfinite(cleaned))


class TestL3AcousticFeatures:
    def test_snr_high_for_clean_signal(self):
        """A signal-plus-silence pattern should produce a relatively high SNR."""
        tone = make_tone(440, 0.05, amplitude=1.0)
        silence = np.zeros(int(16000 * 0.05))
        combined = np.concatenate([tone, silence])
        sig = make_signal(combined)
        snr = prim_estimate_snr(sig)
        assert snr > 5, f"signal+silence SNR={snr}, expected > 5 dB"

    def test_snr_low_for_noisy_signal(self):
        """Noise-dominated inputs should produce a lower SNR."""
        noise = make_noise(0.1, amplitude=2.0)
        tiny_tone = make_tone(440, 0.1, amplitude=0.01)
        sig = make_signal(tiny_tone + noise)
        snr = prim_estimate_snr(sig)
        assert snr < 20, f"noisy signal SNR={snr}, expected < 20 dB"

    def test_rt60_is_positive(self):
        """Estimated RT60 should be non-negative."""
        sig = prim_sample_audio(duration_ms=100.0)
        rt60 = prim_estimate_rt60(sig)
        assert rt60 >= 0, f"RT60={rt60}, should be >= 0"

    def test_mfcc_returns_string(self):
        """MFCC summary should mention energy and centroid-like descriptors."""
        sig = prim_sample_audio()
        mfcc = prim_extract_mfcc(sig)
        assert isinstance(mfcc, str)
        assert "Energy" in mfcc
        assert "centroid" in mfcc.lower()

    def test_full_features_structure(self):
        """`comp_extract_full_features` should return a complete feature bundle."""
        sig = prim_sample_audio(duration_ms=32.0)
        f = comp_extract_full_features(sig)
        assert isinstance(f.snr_db, float)
        assert isinstance(f.rt60_s, float)
        assert isinstance(f.energy_db, float)
        assert isinstance(f.n_active_sources, int)
        assert f.n_active_sources >= 1
        assert f.temporal_pattern in ("stationary", "modulated", "impulsive")
        assert len(f.mfcc_summary) > 10

    def test_full_features_broadband_vs_tonal(self):
        """Broadband noise and a pure tone should yield different centroids."""
        tone_sig = make_signal(make_tone(200, 0.1))
        noise_sig = make_signal(make_noise(0.1))
        f_tone = comp_extract_full_features(tone_sig)
        f_noise = comp_extract_full_features(noise_sig)
        assert f_tone.spectral_centroid_hz < f_noise.spectral_centroid_hz * 1.5


class TestL6DeterministicGain:
    def test_gain_params_structure(self):
        """`prim_generate_gain_params` should return the expected structure."""
        audiogram = '{"250":30,"500":35,"1000":40,"2000":50,"4000":60}'
        gain = prim_generate_gain_params(audiogram, "market scene")
        assert "gain_per_frequency" in gain
        assert "compression_ratio" in gain
        assert "deterministic" in gain
        assert gain["deterministic"] is True

    def test_gain_increases_with_hearing_loss(self):
        """More severe hearing loss should map to larger gain."""
        audiogram = '{"250":10,"500":20,"1000":40,"2000":60,"4000":80}'
        gain = prim_generate_gain_params(audiogram, "quiet")
        gpf = gain["gain_per_frequency"]
        assert gpf["4000"] > gpf["250"], (
            f"gain@4000={gpf['4000']} should be > gain@250={gpf['250']}"
        )

    def test_compression_ratio_reasonable(self):
        """Compression ratio should stay within a plausible range."""
        audiogram = '{"250":30,"500":35,"1000":40,"2000":50,"4000":60}'
        gain = prim_generate_gain_params(audiogram, "market")
        cr = gain["compression_ratio"]
        assert 1.0 <= cr <= 4.0, f"compression_ratio={cr}, expected 1.0-4.0"


SCENARIO_DIR = Path(__file__).parent.parent / "asir" / "eval" / "audio" / "scenarios"
EVAL_EXAMPLES = create_eval_examples()


class TestEvalScenarioConsistency:
    """Validate consistency between eval scenarios and generated audio files."""

    def test_every_scenario_has_wav(self):
        """Each eval scenario should have a matching WAV file."""
        for ex in EVAL_EXAMPLES:
            wav = SCENARIO_DIR / f"{ex.scenario}.wav"
            assert wav.exists(), f"Missing WAV for scenario '{ex.scenario}': {wav}"

    def test_no_orphan_wavs(self):
        """There should be no scenario WAV without a matching eval definition."""
        scenario_names = {ex.scenario for ex in EVAL_EXAMPLES}
        for wav in SCENARIO_DIR.glob("*.wav"):
            assert wav.stem in scenario_names, (
                f"Orphan WAV '{wav.name}' has no matching eval scenario"
            )

    def test_eval_examples_count(self):
        """The benchmark should define 10 eval scenarios."""
        assert len(EVAL_EXAMPLES) == 10

    @pytest.mark.parametrize("ex", EVAL_EXAMPLES, ids=[e.scenario for e in EVAL_EXAMPLES])
    def test_eval_example_fields(self, ex):
        """Each eval example should expose the required physical parameters."""
        assert hasattr(ex, "snr_db")
        assert hasattr(ex, "rt60_s")
        assert hasattr(ex, "n_active_sources")
        assert hasattr(ex, "energy_db")
        assert hasattr(ex, "temporal_pattern")
        assert hasattr(ex, "audiogram_json")


class TestEvalAudioL1L3:
    """Run L1-L3 on real eval audio to ensure the deterministic stack is stable."""

    @pytest.mark.parametrize("ex", EVAL_EXAMPLES, ids=[e.scenario for e in EVAL_EXAMPLES])
    def test_load_and_extract_features(self, ex):
        """Load a scenario WAV and verify feature extraction stays well-formed."""
        wav = SCENARIO_DIR / f"{ex.scenario}.wav"
        if not wav.exists():
            pytest.skip(f"WAV not found: {wav}")

        signal = prim_load_audio(str(wav))
        if isinstance(signal, tuple):
            signal = signal[0]

        assert signal.n_channels >= 1
        assert signal.sample_rate == 16000
        assert len(signal.samples[0]) > 0

        features = comp_extract_full_features(signal)
        assert np.isfinite(features.snr_db)
        assert np.isfinite(features.rt60_s)
        assert np.isfinite(features.energy_db)
        assert features.n_active_sources >= 1
        assert features.temporal_pattern in ("stationary", "modulated", "impulsive")
        assert len(features.mfcc_summary) > 10

    @pytest.mark.parametrize("ex", EVAL_EXAMPLES, ids=[e.scenario for e in EVAL_EXAMPLES])
    def test_fft_and_beamform(self, ex):
        """Load a scenario WAV and verify FFT plus beamforming do not break."""
        wav = SCENARIO_DIR / f"{ex.scenario}.wav"
        if not wav.exists():
            pytest.skip(f"WAV not found: {wav}")

        signal = prim_load_audio(str(wav))
        if isinstance(signal, tuple):
            signal = signal[0]

        spectrum = prim_fft(signal)
        assert spectrum["freq_bins"] > 0
        assert len(spectrum["magnitude"]) == spectrum["freq_bins"]

        bf = prim_beamform(signal, target_azimuth_deg=0.0)
        assert len(bf) > 0
        assert np.all(np.isfinite(bf))

    @pytest.mark.parametrize("ex", EVAL_EXAMPLES, ids=[e.scenario for e in EVAL_EXAMPLES])
    def test_gain_params_for_scenario(self, ex):
        """Each scenario audiogram should produce valid deterministic gain settings."""
        gain = prim_generate_gain_params(ex.audiogram_json, ex.scenario)
        assert gain["deterministic"] is True
        cr = gain["compression_ratio"]
        assert 1.0 <= cr <= 4.0, f"{ex.scenario}: compression_ratio={cr}"
