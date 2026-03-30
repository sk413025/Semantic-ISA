import json

import dspy
import numpy as np


class GenerateBeamformingParamsSig(dspy.Signature):
    """
    [PRIM] Layer 6: beamforming-parameter generation.
    BACKEND: LLM + physics_constraints
    RELIABILITY: target_direction_error <= 15 degrees in 90% of cases
    CONSTRAINT: beam_width >= 20 degrees (minimum physically realistic width
    for this microphone array)

    This remains a primitive because beam parameters must jointly consider
    semantics (target-speaker direction) and physics (array geometry).

    Beam-width guidance:
    - Quiet / open listening (SNR > 15 dB, 1-2 sources) -> wide beam 90-360 degrees
    - Moderate noise / multiple sources (SNR 5-15 dB) -> medium beam 45-90 degrees
    - Severe noise / focused listening (SNR < 5 dB or explicit focus request) -> narrow beam 20-45 degrees
    """

    scene_understanding: str = dspy.InputField(desc="Layer-5 scene understanding")
    mic_geometry: str = dspy.InputField(
        desc="Microphone-array geometry: spacing, count, and arrangement"
    )

    target_azimuth_deg: float = dspy.OutputField(desc="Target azimuth in degrees [-180, 180]")
    beam_width_deg: float = dspy.OutputField(
        desc="Beam width in degrees [20, 360]. Quiet scenes should use wide beams, noisy scenes narrow beams, and intermediate scenes medium beams."
    )
    null_directions_json: str = dspy.OutputField(desc="JSON list of null directions")
    reasoning: str = dspy.OutputField(desc="Decision reasoning")


class GenerateNoiseReductionParamsSig(dspy.Signature):
    """
    [PRIM] Layer 6: noise-reduction parameter generation.
    BACKEND: LLM
    RELIABILITY: PESQ improvement >= 0.3 in 80% of cases after applying the strategy

    This is a primitive because NR parameters must balance user preference and
    scene demands together.
    """

    scene_understanding: str = dspy.InputField(desc="Layer-5 scene understanding")
    user_preferences: str = dspy.InputField(
        desc="User preferences: noise tolerance, processing preference, and related constraints"
    )

    method: str = dspy.OutputField(
        desc="Noise-reduction method: spectral_subtraction / wiener / dnn_masking"
    )
    aggressiveness: float = dspy.OutputField(desc="Aggressiveness [0, 1]")
    preserve_bands_json: str = dspy.OutputField(desc="JSON list of bands that should be preserved")
    reasoning: str = dspy.OutputField(desc="Decision reasoning")


def prim_generate_gain_params(audiogram_json: str, scene_understanding: str) -> dict:
    """
    [PRIM] Layer 6: gain-parameter generation.
    BACKEND: deterministic (NAL-NL2-style audiological rule)
    RELIABILITY: 100%

    Not every semantic-layer operation requires an LLM. Gain calculation can
    rely on mature audiological formulas, so this primitive is deterministic.
    """

    # Simplified NAL-NL2-style gain calculation.
    try:
        audiogram = json.loads(audiogram_json)
    except Exception:
        audiogram = {"250": 30, "500": 35, "1000": 40, "2000": 50, "4000": 60}

    gain_db = {}
    for freq, hearing_loss in audiogram.items():
        # Simplified rule: gain ~= hearing_loss * 0.46 + adjustment.
        gain_db[freq] = round(float(hearing_loss) * 0.46 + 5, 1)

    # Adjust compression ratio based on mean hearing loss.
    avg_loss = np.mean(list(audiogram.values()))
    compression = 1.0 + (avg_loss / 100) * 2  # 1:1 to 3:1

    return {
        "gain_per_frequency": gain_db,
        "compression_ratio": round(float(compression), 2),
        "attack_ms": 5.0,
        "release_ms": 50.0,
        "deterministic": True,
    }
