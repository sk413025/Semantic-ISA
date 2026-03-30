import numpy as np

from asir.types import AcousticFeatures, RawSignal


def prim_extract_mfcc(signal: RawSignal, n_coeffs: int = 13) -> str:
    """
    [PRIM] Layer 3: MFCC feature extraction.
    BACKEND: deterministic

    Returns a text summary for the LLM, since the model does not directly
    consume tensors.
    """

    # Simplified computation; a production system would use librosa.
    spectrum = np.abs(np.fft.rfft(signal.samples[0]))
    energy = np.sum(spectrum ** 2)
    spectral_centroid = np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + 1e-10)
    spectral_rolloff = np.searchsorted(np.cumsum(spectrum), 0.85 * np.sum(spectrum))

    return (
        f"Energy: {energy:.1f}, Spectral centroid: {spectral_centroid:.0f} bins, "
        f"Rolloff: {spectral_rolloff} bins, Distribution: "
        f"{'broadband' if spectral_centroid > len(spectrum) * 0.3 else 'low-frequency dominant'}"
    )


def prim_estimate_snr(signal: RawSignal) -> float:
    """
    [PRIM] Layer 3: SNR estimation — deterministic.

    Sort short-frame energies, treat the top 30% as speech+noise and the bottom
    30% as noise-only, then compute:
    SNR = 10*log10((P_high / P_low) - 1)
    """

    s = np.array(signal.samples[0])
    if len(s) < 160:
        return 0.0

    # Split into 10 ms frames (160 samples at 16 kHz).
    frame_len = 160
    n_frames = len(s) // frame_len
    if n_frames < 3:
        return 0.0

    frame_powers = np.array(
        [np.mean(s[i * frame_len : (i + 1) * frame_len] ** 2) for i in range(n_frames)]
    )

    # High-energy frames approximate speech+noise; low-energy frames approximate noise.
    sorted_powers = np.sort(frame_powers)
    n_low = max(1, n_frames * 3 // 10)
    n_high = max(1, n_frames * 3 // 10)

    noise_power = np.mean(sorted_powers[:n_low]) + 1e-10
    signal_noise_power = np.mean(sorted_powers[-n_high:]) + 1e-10

    if signal_noise_power <= noise_power * 1.1:
        return 0.0

    snr_linear = (signal_noise_power / noise_power) - 1.0
    return float(10 * np.log10(max(snr_linear, 1e-10)))


def prim_estimate_rt60(signal: RawSignal) -> float:
    """[PRIM] Layer 3: reverberation-time estimation — deterministic."""

    # Simplified estimate from the energy-decay curve.
    s = np.array(signal.samples[0])
    energy_curve = np.cumsum(s[::-1] ** 2)[::-1]
    energy_curve = energy_curve / (energy_curve[0] + 1e-10)

    # Find the time where energy decays by 60 dB.
    db_curve = 10 * np.log10(energy_curve + 1e-10)
    idx_60 = np.searchsorted(-db_curve, 60)
    return float(idx_60 / signal.sample_rate) if idx_60 < len(s) else 0.5


def comp_extract_full_features(signal: RawSignal) -> AcousticFeatures:
    """
    [COMP] Layer 3: full feature extraction.
    Composed from multiple layer-3 primitives and still fully deterministic.
    """

    mfcc_summary = prim_extract_mfcc(signal)
    snr = prim_estimate_snr(signal)
    rt60 = prim_estimate_rt60(signal)

    s = np.array(signal.samples[0])
    energy = float(10 * np.log10(np.mean(s ** 2) + 1e-10) + 94)
    spectrum = np.abs(np.fft.rfft(s))
    centroid = float(np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + 1e-10))
    centroid_hz = centroid * signal.sample_rate / (2 * len(spectrum))

    # Estimate the number of active sources using spectral-peak count as a proxy.
    peaks = np.diff(np.sign(np.diff(spectrum)))
    n_sources = int(np.sum(peaks < 0) / 5)

    # Temporal-pattern classification.
    frame_energies = [np.mean(s[i : i + 160] ** 2) for i in range(0, len(s) - 160, 160)]
    energy_var = np.var(frame_energies) / (np.mean(frame_energies) ** 2 + 1e-10)
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
        temporal_pattern=temporal,
    )
