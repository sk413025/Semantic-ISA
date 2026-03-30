import json

import numpy as np

from asir.types import DSPParameterSet


def comp_strategy_to_dsp_params(
    strategy,
    mic_spacing_m: float = 0.01,
    sample_rate: int = 16000,
) -> DSPParameterSet:
    """
    [COMP] Layer 6 -> Layer 2: semantic-to-physical downward translation.
    BACKEND: deterministic

    The input lives in semantic space: a ProcessingStrategy with natural-language
    reasoning. The output lives in physical space: exact filter coefficients,
    beam weights, and DSP parameters. The translation itself is deterministic:
    the LLM proposes the strategy, and the downward mapping is pure math.
    """

    # 1. Deterministic beamforming weight calculation.
    azimuth_rad = np.radians(strategy.target_azimuth_deg)
    d = mic_spacing_m
    c = 343.0  # Speed of sound in m/s.
    freq_center = 2000  # 2 kHz center frequency.
    phase_diff = 2 * np.pi * freq_center * d * np.sin(azimuth_rad) / c
    beam_weights = [1.0, float(np.cos(phase_diff))]

    # 2. Deterministic noise-mask calculation.
    n_bins = 129  # 256-point FFT.
    try:
        preserve = json.loads(strategy.preserve_bands_json)
    except Exception:
        preserve = []

    mask = np.ones(n_bins) * strategy.nr_aggressiveness
    # Apply a milder mask to bands that should be preserved.
    for band in preserve:
        if isinstance(band, str) and "low" in band.lower():
            mask[: n_bins // 4] *= 0.3
        elif isinstance(band, str) and "mid" in band.lower():
            mask[n_bins // 4 : n_bins * 3 // 4] *= 0.3

    # 3. Deterministic gain-filter coefficients derived from gain_per_frequency.
    gain_dict = strategy.gain_per_frequency
    if isinstance(gain_dict, dict):
        gains = list(gain_dict.values())
    else:
        gains = [20.0] * 6

    # Simplified implementation: generate FIR coefficients from frequency-domain gain.
    freq_response = np.interp(
        np.linspace(0, 8000, n_bins),
        [250, 500, 1000, 2000, 4000, 8000][: len(gains)],
        [10 ** (g / 20) for g in gains[:6]],
    )
    filter_coeffs = np.fft.irfft(freq_response, n=32).tolist()

    return DSPParameterSet(
        filter_coeffs=filter_coeffs,
        beam_weights=beam_weights,
        noise_mask=mask.tolist(),
        compression_ratio=strategy.compression_ratio,
        attack_ms=5.0,
        release_ms=50.0,
    )
