import json
import numpy as np
from asir.types import DSPParameterSet


def comp_strategy_to_dsp_params(strategy,
                                 mic_spacing_m: float = 0.01,
                                 sample_rate: int = 16000) -> DSPParameterSet:
    """
    [COMP] 第六→二層：語意到物理的下行翻譯
    BACKEND: deterministic

    ★ 輸入是語義空間的（ProcessingStrategy with 自然語言推理）
    ★ 輸出是物理空間的（精確的濾波器係數和波束權重）
    ★ 翻譯本身是確定性的——LLM 的作用在於生成策略，
      翻譯過程是純數學
    """

    # 1. 波束成形權重計算（確定性）
    azimuth_rad = np.radians(strategy.target_azimuth_deg)
    d = mic_spacing_m
    c = 343.0  # 聲速 m/s
    freq_center = 2000  # 2kHz 中心頻率
    phase_diff = 2 * np.pi * freq_center * d * np.sin(azimuth_rad) / c
    beam_weights = [1.0, float(np.cos(phase_diff))]

    # 2. 降噪遮罩計算（確定性）
    n_bins = 129  # 256-point FFT
    try:
        preserve = json.loads(strategy.preserve_bands_json)
    except:
        preserve = []

    mask = np.ones(n_bins) * strategy.nr_aggressiveness
    # 保留頻段的遮罩設為較低值
    for band in preserve:
        if isinstance(band, str) and "low" in band.lower():
            mask[:n_bins // 4] *= 0.3
        elif isinstance(band, str) and "mid" in band.lower():
            mask[n_bins // 4: n_bins * 3 // 4] *= 0.3

    # 3. 增益濾波器係數（確定性，從 gain_per_frequency 計算 FIR）
    gain_dict = strategy.gain_per_frequency
    if isinstance(gain_dict, dict):
        gains = list(gain_dict.values())
    else:
        gains = [20.0] * 6

    # 簡化版：從頻域增益生成 FIR 係數
    freq_response = np.interp(
        np.linspace(0, 8000, n_bins),
        [250, 500, 1000, 2000, 4000, 8000][:len(gains)],
        [10 ** (g / 20) for g in gains[:6]]
    )
    filter_coeffs = np.fft.irfft(freq_response, n=32).tolist()

    return DSPParameterSet(
        filter_coeffs=filter_coeffs,
        beam_weights=beam_weights,
        noise_mask=mask.tolist(),
        compression_ratio=strategy.compression_ratio,
        attack_ms=5.0,
        release_ms=50.0
    )
