"""
Evaluation examples kept separate from the scenarios in `gepa/training.py`.

These examples test generalization across:
- different noise levels, from quiet to extremely noisy
- different spaces, including indoor, outdoor, and reverberant halls
- different user actions, including no action, spoken feedback, and explicit commands
- different hearing-loss severities, from mild to severe

=== Example field guide ===

Physical inputs (converted by `run.py:build_features()` into `AcousticFeatures`):
  scenario          str    scenario identifier, not passed to the LLM
  snr_db            float  signal-to-noise ratio in dB; lower is noisier
  rt60_s            float  reverberation time in seconds; larger is more reverberant
  n_active_sources  int    estimated number of active sources
  energy_db         float  signal energy in dB SPL
  temporal_pattern  str    temporal pattern: stationary / modulated / impulsive
  user_profile      str    user description passed to L4/L5
  user_action       str    user action passed to the L7 pipeline router
  audiogram_json    str    audiogram JSON passed to L6 NAL-NL2 gain calculation

Constraint fields (read by `metrics.py`):
  expect_noisy             bool       -> check_l5: scene noise description should match conditions
  expect_reverberant       bool       -> check_l5: highly reverberant scenes should mention reverb
  expect_strong_nr         bool       -> check_l6: NR aggressiveness should exceed 0.4
  expect_beam_focus        bool       -> check_dsp: beam width should be narrower than 90°
  expect_high_gain         bool       -> check_dsp: severe hearing loss should yield higher gain
  expect_full_depth        bool       -> check_l7: execution depth should be "full"

Tighter constraints (README target values and GEPA targets):
  expect_beam_width_range    (min,max) -> beam width remains within a target range
  expect_beam_azimuth_front  bool      -> beam points forward (azimuth ≈ 0°)
  expect_nr_range            (min,max) -> NR aggressiveness remains in range
  expect_noise_mask_active   bool      -> noise_mask actually attenuates noise
"""

import dspy


def create_eval_examples():
    """Create evaluation examples that differ from the GEPA training set."""
    examples = []

    # --- E1: restaurant, multi-speaker conversation ---
    examples.append(dspy.Example(
        scenario="restaurant_dinner",
        snr_db=3.0,
        rt60_s=0.7,
        n_active_sources=5,
        energy_db=72.0,
        temporal_pattern="modulated",
        user_profile="65-year-old woman with bilateral mild-to-moderate sensorineural hearing loss",
        user_action="none",
        audiogram_json='{"250":20,"500":25,"1000":30,"2000":40,"4000":50}',
        expect_noisy=True,
        expect_reverberant=False,
        expect_strong_nr=True,
        expect_beam_focus=True,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E2: church / ceremony hall (high reverberation) ---
    examples.append(dspy.Example(
        scenario="church_ceremony",
        snr_db=12.0,
        rt60_s=2.5,
        n_active_sources=2,
        energy_db=60.0,
        temporal_pattern="modulated",
        user_profile="72-year-old man with bilateral moderate sensorineural hearing loss who prefers natural sound",
        user_action="none",
        audiogram_json='{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
        expect_noisy=False,
        expect_reverberant=True,
        expect_strong_nr=False,
        expect_beam_focus=True,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E3: quiet library ---
    examples.append(dspy.Example(
        scenario="quiet_library",
        snr_db=30.0,
        rt60_s=0.6,
        n_active_sources=1,
        energy_db=40.0,
        temporal_pattern="stationary",
        user_profile="55-year-old woman with mild high-frequency hearing loss in the right ear",
        user_action="none",
        audiogram_json='{"250":10,"500":15,"1000":15,"2000":25,"4000":35}',
        expect_noisy=False,
        expect_reverberant=False,
        expect_strong_nr=False,
        expect_beam_focus=False,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E4: phone call beside a street ---
    examples.append(dspy.Example(
        scenario="street_phone_call",
        snr_db=-2.0,
        rt60_s=0.1,
        n_active_sources=7,
        energy_db=80.0,
        temporal_pattern="stationary",
        user_profile="45-year-old man with moderate sensorineural hearing loss in the left ear",
        user_action="focus_front",
        audiogram_json='{"250":25,"500":30,"1000":35,"2000":45,"4000":55}',
        expect_noisy=True,
        expect_reverberant=False,
        expect_strong_nr=True,
        expect_beam_focus=True,
        expect_preference_updated=False,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E5: supermarket shopping ---
    examples.append(dspy.Example(
        scenario="supermarket_shopping",
        snr_db=8.0,
        rt60_s=1.0,
        n_active_sources=4,
        energy_db=68.0,
        temporal_pattern="modulated",
        user_profile="72-year-old man with bilateral moderate sensorineural hearing loss who prefers natural sound",
        user_action="none",
        audiogram_json='{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
        expect_noisy=True,
        expect_reverberant=True,
        expect_strong_nr=False,
        expect_beam_focus=False,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E6: in-car conversation ---
    examples.append(dspy.Example(
        scenario="car_conversation",
        snr_db=6.0,
        rt60_s=0.15,
        n_active_sources=3,
        energy_db=70.0,
        temporal_pattern="stationary",
        user_profile="68-year-old man with bilateral moderate sensorineural hearing loss",
        user_action="none",
        audiogram_json='{"250":25,"500":30,"1000":35,"2000":45,"4000":55}',
        expect_noisy=True,
        expect_reverberant=False,
        expect_strong_nr=False,
        expect_beam_focus=True,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E7: user complains "too noisy" ---
    examples.append(dspy.Example(
        scenario="noisy_cafe_complaint",
        snr_db=2.0,
        rt60_s=0.6,
        n_active_sources=6,
        energy_db=75.0,
        temporal_pattern="modulated",
        user_profile="72-year-old man with bilateral moderate sensorineural hearing loss who prefers natural sound",
        user_action="too noisy",
        audiogram_json='{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
        expect_noisy=True,
        expect_reverberant=False,
        expect_strong_nr=True,
        expect_beam_focus=True,
        expect_full_depth=True,
        expect_preference_updated=True,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E8: severe hearing loss in a quiet home ---
    examples.append(dspy.Example(
        scenario="severe_loss_quiet_home",
        snr_db=25.0,
        rt60_s=0.4,
        n_active_sources=1,
        energy_db=50.0,
        temporal_pattern="modulated",
        user_profile="80-year-old woman with bilateral severe sensorineural hearing loss",
        user_action="none",
        audiogram_json='{"250":50,"500":55,"1000":60,"2000":70,"4000":80}',
        expect_noisy=False,
        expect_reverberant=False,
        expect_strong_nr=False,
        expect_beam_focus=False,
        expect_high_gain=True,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E9: wet market vendor conversation (README flagship scenario) ---
    examples.append(dspy.Example(
        scenario="wet_market_vendor",
        snr_db=0.0,
        rt60_s=0.6,
        n_active_sources=8,
        energy_db=78.0,
        temporal_pattern="modulated",
        user_profile="72-year-old man with bilateral moderate sensorineural hearing loss who prefers natural sound",
        user_action="none",
        audiogram_json='{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
        expect_noisy=True,
        expect_reverberant=False,
        expect_strong_nr=True,
        expect_beam_focus=True,
        expect_beam_width_range=(20, 60),
        expect_beam_azimuth_front=True,
        expect_nr_range=(0.5, 0.9),
        expect_noise_mask_active=True,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E10: wet market + user says "too muffled" ---
    examples.append(dspy.Example(
        scenario="market_too_muffled",
        snr_db=0.0,
        rt60_s=0.6,
        n_active_sources=8,
        energy_db=78.0,
        temporal_pattern="modulated",
        user_profile="72-year-old man with bilateral moderate sensorineural hearing loss who prefers natural sound",
        user_action="too muffled",
        audiogram_json='{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
        expect_noisy=True,
        expect_reverberant=False,
        expect_strong_nr=False,
        expect_beam_focus=True,
        expect_full_depth=True,
        expect_preference_updated=True,
        expect_beam_width_range=(20, 60),
        expect_beam_azimuth_front=True,
        expect_nr_range=(0.1, 0.5),
        expect_noise_mask_active=True,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    return examples
