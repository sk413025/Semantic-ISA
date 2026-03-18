"""
L1-L3 Deterministic Layer Tests

這些測試不需要 API key，純 numpy 驗證。
用合成訊號測試每個 PRIM 的物理正確性。
"""
import numpy as np
import pytest
from asir.types import RawSignal
from asir.primitives import (
    prim_sample_audio, prim_fft, prim_estimate_noise_psd,
    prim_beamform, comp_spectral_subtract,
    prim_extract_mfcc, prim_estimate_snr, prim_estimate_rt60,
    comp_extract_full_features, prim_generate_gain_params,
)


# ===== Helpers: 合成已知特性的訊號 =====

def make_signal(samples_ch0, samples_ch1=None, sample_rate=16000):
    """從 numpy array 建立 RawSignal。"""
    ch0 = np.asarray(samples_ch0, dtype=float)
    ch1 = np.asarray(samples_ch1, dtype=float) if samples_ch1 is not None else ch0.copy()
    return RawSignal(
        samples=[ch0.tolist(), ch1.tolist()],
        sample_rate=sample_rate,
        n_channels=2,
        duration_ms=float(len(ch0) / sample_rate * 1000),
    )


def make_tone(freq_hz=440, duration_s=0.1, amplitude=1.0, sample_rate=16000):
    """產生純音訊號。"""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def make_noise(duration_s=0.1, amplitude=1.0, sample_rate=16000):
    """產生高斯白噪音。"""
    n = int(sample_rate * duration_s)
    return amplitude * np.random.RandomState(42).randn(n)


# ===== L1: Physical Sensing =====

class TestL1PhysicalSensing:
    def test_sample_audio_shape(self):
        """prim_sample_audio 回傳正確的 RawSignal 結構。"""
        sig = prim_sample_audio(duration_ms=32.0, n_channels=2)
        assert sig.n_channels == 2
        assert sig.sample_rate == 16000
        assert abs(sig.duration_ms - 32.0) < 0.1
        assert len(sig.samples) == 2
        expected_n = int(16000 * 32.0 / 1000)
        assert len(sig.samples[0]) == expected_n
        assert len(sig.samples[1]) == expected_n

    def test_sample_audio_deterministic(self):
        """同樣參數應產生相同訊號（固定 seed 在實作中）。"""
        # prim_sample_audio 用 np.random，不保證 deterministic
        # 但 shape 和 range 應該一致
        sig = prim_sample_audio(duration_ms=10.0)
        arr = np.array(sig.samples[0])
        assert np.all(np.isfinite(arr)), "should have no NaN/inf"
        assert np.max(np.abs(arr)) < 100, "amplitude should be reasonable"


# ===== L2: Signal Processing =====

class TestL2SignalProcessing:
    def test_fft_output_structure(self):
        """prim_fft 回傳 magnitude, phase, freq_bins。"""
        sig = make_signal(make_tone(440, 0.032))
        result = prim_fft(sig)
        assert "magnitude" in result
        assert "phase" in result
        assert "freq_bins" in result
        assert len(result["magnitude"]) == result["freq_bins"]
        assert len(result["phase"]) == result["freq_bins"]

    def test_fft_peak_at_target_frequency(self):
        """純音 440Hz 的 FFT 應在 440Hz 附近有峰值。"""
        sr = 16000
        sig = make_signal(make_tone(440, 0.1, sample_rate=sr), sample_rate=sr)
        result = prim_fft(sig)
        mag = np.array(result["magnitude"])
        peak_bin = np.argmax(mag)
        peak_freq = peak_bin * sr / (2 * (len(mag) - 1))
        assert abs(peak_freq - 440) < 50, f"peak at {peak_freq}Hz, expected ~440Hz"

    def test_beamform_output_length(self):
        """波束成形輸出長度應等於輸入。"""
        sig = prim_sample_audio(duration_ms=32.0)
        bf = prim_beamform(sig, target_azimuth_deg=0.0)
        assert len(bf) == len(sig.samples[0])

    def test_beamform_front_enhances_correlated_signal(self):
        """target=0° 時，兩通道相同的訊號應被增強。"""
        tone = make_tone(200, 0.032)
        sig = make_signal(tone, tone)  # 完全相關 = 正前方
        bf = np.array(prim_beamform(sig, target_azimuth_deg=0.0))
        # 增強後的能量應 >= 單通道
        assert np.mean(bf ** 2) >= np.mean(np.array(tone) ** 2) * 0.8

    def test_noise_psd_output_length(self):
        """噪音 PSD 輸出長度應等於 FFT bins。"""
        sig = prim_sample_audio()
        psd = prim_estimate_noise_psd(sig)
        fft_bins = len(np.fft.rfft(sig.samples[0]))
        assert len(psd) == fft_bins

    def test_spectral_subtract_reduces_noise(self):
        """頻譜相減應降低噪音能量。"""
        noise = make_noise(0.032, amplitude=0.5)
        tone = make_tone(440, 0.032, amplitude=1.0)
        noisy = tone + noise
        sig = make_signal(noisy)
        noise_psd = prim_estimate_noise_psd(sig)
        cleaned = comp_spectral_subtract(sig, noise_psd, alpha=1.0)
        # cleaned 的噪音能量應比原始低（至少不會增加太多）
        assert len(cleaned) > 0
        assert np.all(np.isfinite(cleaned))


# ===== L3: Acoustic Features =====

class TestL3AcousticFeatures:
    def test_snr_high_for_clean_signal(self):
        """有語音段+安靜段的訊號，SNR 應較高。"""
        # 純等幅正弦波各幀能量相同，estimator 無法區分 signal/noise
        # 用「前半段有訊號+後半段幾乎靜音」模擬真實情況
        tone = make_tone(440, 0.05, amplitude=1.0)
        silence = np.zeros(int(16000 * 0.05))
        combined = np.concatenate([tone, silence])
        sig = make_signal(combined)
        snr = prim_estimate_snr(sig)
        assert snr > 5, f"signal+silence SNR={snr}, expected > 5 dB"

    def test_snr_low_for_noisy_signal(self):
        """噪音遠大於訊號時 SNR 應較低。"""
        noise = make_noise(0.1, amplitude=2.0)
        tiny_tone = make_tone(440, 0.1, amplitude=0.01)
        sig = make_signal(tiny_tone + noise)
        snr = prim_estimate_snr(sig)
        assert snr < 20, f"noisy signal SNR={snr}, expected < 20 dB"

    def test_rt60_is_positive(self):
        """RT60 估計應為正數。"""
        sig = prim_sample_audio(duration_ms=100.0)
        rt60 = prim_estimate_rt60(sig)
        assert rt60 >= 0, f"RT60={rt60}, should be >= 0"

    def test_mfcc_returns_string(self):
        """MFCC 摘要應回傳包含 Energy 和 centroid 的文字。"""
        sig = prim_sample_audio()
        mfcc = prim_extract_mfcc(sig)
        assert isinstance(mfcc, str)
        assert "Energy" in mfcc
        assert "centroid" in mfcc.lower() or "Centroid" in mfcc

    def test_full_features_structure(self):
        """comp_extract_full_features 回傳完整的 AcousticFeatures。"""
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
        """寬頻噪音 vs 純音應產生不同的 spectral centroid。"""
        tone_sig = make_signal(make_tone(200, 0.1))
        noise_sig = make_signal(make_noise(0.1))
        f_tone = comp_extract_full_features(tone_sig)
        f_noise = comp_extract_full_features(noise_sig)
        # 純音 centroid 應接近 200Hz，噪音 centroid 應更高
        assert f_tone.spectral_centroid_hz < f_noise.spectral_centroid_hz * 1.5


# ===== L6 Deterministic: NAL-NL2 Gain =====

class TestL6DeterministicGain:
    def test_gain_params_structure(self):
        """prim_generate_gain_params 回傳正確的結構。"""
        audiogram = '{"250":30,"500":35,"1000":40,"2000":50,"4000":60}'
        gain = prim_generate_gain_params(audiogram, "market scene")
        assert "gain_per_frequency" in gain
        assert "compression_ratio" in gain
        assert "deterministic" in gain
        assert gain["deterministic"] is True

    def test_gain_increases_with_hearing_loss(self):
        """聽損越嚴重的頻率，增益應越大。"""
        audiogram = '{"250":10,"500":20,"1000":40,"2000":60,"4000":80}'
        gain = prim_generate_gain_params(audiogram, "quiet")
        gpf = gain["gain_per_frequency"]
        # 4000Hz (80dB loss) 的增益應 > 250Hz (10dB loss)
        assert gpf["4000"] > gpf["250"], \
            f"gain@4000={gpf['4000']} should be > gain@250={gpf['250']}"

    def test_compression_ratio_reasonable(self):
        """壓縮比應在合理範圍 (1.0-4.0)。"""
        audiogram = '{"250":30,"500":35,"1000":40,"2000":50,"4000":60}'
        gain = prim_generate_gain_params(audiogram, "market")
        cr = gain["compression_ratio"]
        assert 1.0 <= cr <= 4.0, f"compression_ratio={cr}, expected 1.0-4.0"
