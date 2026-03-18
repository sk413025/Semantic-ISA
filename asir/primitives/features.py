import numpy as np
from asir.types import RawSignal, AcousticFeatures


def prim_extract_mfcc(signal: RawSignal, n_coeffs: int = 13) -> str:
    """
    [PRIM] 第三層：MFCC 特徵提取
    BACKEND: deterministic
    返回文字摘要供 LLM 理解（因為 LLM 不直接理解張量）
    """
    # 簡化計算 — 真實系統用 librosa
    spectrum = np.abs(np.fft.rfft(signal.samples[0]))
    energy = np.sum(spectrum ** 2)
    spectral_centroid = np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + 1e-10)
    spectral_rolloff = np.searchsorted(np.cumsum(spectrum), 0.85 * np.sum(spectrum))

    return (f"Energy: {energy:.1f}, Spectral centroid: {spectral_centroid:.0f} bins, "
            f"Rolloff: {spectral_rolloff} bins, Distribution: "
            f"{'broadband' if spectral_centroid > len(spectrum) * 0.3 else 'low-frequency dominant'}")


def prim_estimate_snr(signal: RawSignal) -> float:
    """[PRIM] 第三層：信噪比估計 — deterministic"""
    s = np.array(signal.samples[0])
    signal_power = np.mean(s ** 2)
    # 簡化：用最低 10% 能量的幀作為噪音估計
    frame_size = len(s) // 10
    frames = [s[i:i+frame_size] for i in range(0, len(s) - frame_size, frame_size)]
    noise_power = min(np.mean(f ** 2) for f in frames) if frames else 1e-10
    return float(10 * np.log10(signal_power / (noise_power + 1e-10)))


def prim_estimate_rt60(signal: RawSignal) -> float:
    """[PRIM] 第三層：混響時間估計 — deterministic"""
    # 簡化版：從能量衰減曲線估計
    s = np.array(signal.samples[0])
    energy_curve = np.cumsum(s[::-1] ** 2)[::-1]
    energy_curve = energy_curve / (energy_curve[0] + 1e-10)
    # 找到能量降至 -60dB 的時間
    db_curve = 10 * np.log10(energy_curve + 1e-10)
    idx_60 = np.searchsorted(-db_curve, 60)
    return float(idx_60 / signal.sample_rate) if idx_60 < len(s) else 0.5


def comp_extract_full_features(signal: RawSignal) -> AcousticFeatures:
    """
    [COMP] 第三層：完整特徵提取
    由多個第三層 Primitive 組合
    仍然完全確定性
    """
    mfcc_summary = prim_extract_mfcc(signal)
    snr = prim_estimate_snr(signal)
    rt60 = prim_estimate_rt60(signal)

    s = np.array(signal.samples[0])
    energy = float(10 * np.log10(np.mean(s ** 2) + 1e-10) + 94)  # dB SPL
    spectrum = np.abs(np.fft.rfft(s))
    centroid = float(np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + 1e-10))
    centroid_hz = centroid * signal.sample_rate / (2 * len(spectrum))

    # 估計活躍聲源數（用頻譜峰值數量近似）
    peaks = np.diff(np.sign(np.diff(spectrum)))
    n_sources = int(np.sum(peaks < 0) / 5)  # 簡化估計

    # 時域模式判斷
    frame_energies = [np.mean(s[i:i+160]**2) for i in range(0, len(s)-160, 160)]
    energy_var = np.var(frame_energies) / (np.mean(frame_energies)**2 + 1e-10)
    if energy_var > 1.0:
        temporal = "impulsive"
    elif energy_var > 0.1:
        temporal = "modulated"
    else:
        temporal = "stationary"

    return AcousticFeatures(
        mfcc_summary=mfcc_summary,
        snr_db=round(snr, 1),
        rt60_s=round(rt60, 3),
        pitch_hz=round(centroid_hz, 1) if centroid_hz > 80 else None,
        n_active_sources=max(1, min(n_sources, 10)),
        spectral_centroid_hz=round(centroid_hz, 1),
        energy_db=round(energy, 1),
        temporal_pattern=temporal
    )
