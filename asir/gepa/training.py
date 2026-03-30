import dspy

from asir.harness import AcousticSemanticHarness
from asir.primitives.signal import prim_sample_audio


def create_training_examples():
    """
    Create a compact training set spanning representative daily scenarios.

    Each example is a pair of model inputs plus high-level behavioral targets
    used by the GEPA feedback metric.
    """
    examples = []

    examples.append(
        dspy.Example(
            scenario="market_fish_stall",
            snr_db=5.0,
            rt60_s=0.8,
            n_active_sources=6,
            energy_db=78.0,
            temporal_pattern="modulated",
            user_profile=(
                "72-year-old man with bilateral moderate sensorineural hearing loss "
                "who prefers natural sound"
            ),
            user_action="none",
            expected_env="indoor market",
            expected_target_dir="front",
            expected_nr_agg_range=(0.3, 0.6),
            expected_beam_width_range=(30, 60),
            user_prefs_natural=True,
        ).with_inputs(
            "scenario",
            "snr_db",
            "rt60_s",
            "n_active_sources",
            "energy_db",
            "temporal_pattern",
            "user_profile",
            "user_action",
        )
    )

    examples.append(
        dspy.Example(
            scenario="home_quiet",
            snr_db=25.0,
            rt60_s=0.4,
            n_active_sources=1,
            energy_db=55.0,
            temporal_pattern="modulated",
            user_profile=(
                "72-year-old man with bilateral moderate sensorineural hearing loss "
                "who prefers natural sound"
            ),
            user_action="none",
            expected_env="home/living room",
            expected_target_dir="front",
            expected_nr_agg_range=(0.0, 0.2),
            expected_beam_width_range=(60, 120),
            user_prefs_natural=True,
        ).with_inputs(
            "scenario",
            "snr_db",
            "rt60_s",
            "n_active_sources",
            "energy_db",
            "temporal_pattern",
            "user_profile",
            "user_action",
        )
    )

    examples.append(
        dspy.Example(
            scenario="tv_plus_conversation",
            snr_db=10.0,
            rt60_s=0.5,
            n_active_sources=3,
            energy_db=65.0,
            temporal_pattern="modulated",
            user_profile=(
                "72-year-old man with bilateral moderate sensorineural hearing loss "
                "who prefers natural sound"
            ),
            user_action="none",
            expected_env="home/living room",
            expected_target_dir="variable",
            expected_nr_agg_range=(0.2, 0.5),
            expected_beam_width_range=(40, 80),
            user_prefs_natural=True,
        ).with_inputs(
            "scenario",
            "snr_db",
            "rt60_s",
            "n_active_sources",
            "energy_db",
            "temporal_pattern",
            "user_profile",
            "user_action",
        )
    )

    examples.append(
        dspy.Example(
            scenario="park_walking",
            snr_db=15.0,
            rt60_s=0.1,
            n_active_sources=4,
            energy_db=60.0,
            temporal_pattern="stationary",
            user_profile=(
                "72-year-old man with bilateral moderate sensorineural hearing loss "
                "who prefers natural sound"
            ),
            user_action="none",
            expected_env="outdoor park",
            expected_target_dir="variable",
            expected_nr_agg_range=(0.1, 0.4),
            expected_beam_width_range=(60, 180),
            user_prefs_natural=True,
        ).with_inputs(
            "scenario",
            "snr_db",
            "rt60_s",
            "n_active_sources",
            "energy_db",
            "temporal_pattern",
            "user_profile",
            "user_action",
        )
    )

    examples.append(
        dspy.Example(
            scenario="market_dissatisfied",
            snr_db=3.0,
            rt60_s=0.9,
            n_active_sources=8,
            energy_db=82.0,
            temporal_pattern="modulated",
            user_profile=(
                "72-year-old man with bilateral moderate sensorineural hearing loss "
                "who prefers natural sound"
            ),
            user_action="button_press:dissatisfied",
            expected_env="indoor market",
            expected_target_dir="front",
            expected_nr_agg_range=(0.5, 0.8),
            expected_beam_width_range=(20, 40),
            user_prefs_natural=True,
        ).with_inputs(
            "scenario",
            "snr_db",
            "rt60_s",
            "n_active_sources",
            "energy_db",
            "temporal_pattern",
            "user_profile",
            "user_action",
        )
    )

    return examples


class GEPATrainableHarness(dspy.Module):
    """
    Wrapper module used to train the full pipeline with GEPA.

    It bridges the flat training-example fields into the argument structure
    expected by `AcousticSemanticHarness.forward()`.
    """

    def __init__(self, fast_lm=None, strong_lm=None, enable_multimodal: bool = True):
        super().__init__()
        self.harness = AcousticSemanticHarness(
            fast_lm=fast_lm,
            strong_lm=strong_lm,
            enable_multimodal=enable_multimodal,
        )

    def forward(
        self,
        scenario: str = "",
        snr_db: float = 10.0,
        rt60_s: float = 0.5,
        n_active_sources: int = 3,
        energy_db: float = 65.0,
        temporal_pattern: str = "modulated",
        user_profile: str = (
            "72-year-old man with bilateral moderate sensorineural hearing loss "
            "who prefers natural sound"
        ),
        user_action: str = "none",
    ) -> dspy.Prediction:
        raw_signal = prim_sample_audio(duration_ms=32.0, n_channels=2)
        return self.harness(
            raw_signal=raw_signal,
            user_action=user_action,
            user_profile=user_profile,
        )
