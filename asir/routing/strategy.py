import dspy


class StrategyPlanSig(dspy.Signature):
    """
    [ROUTING] Pre-plan the L6 composite before running beam / NR / gain.

    Answer:
    - What is the core acoustic challenge in this scene?
    - How should beamforming and NR cooperate?
    - What overall processing budget is appropriate:
      conservative, moderate, or aggressive?

    Physical constraints:
    - BTE hearing aids have only two microphones with ~10 mm spacing,
      so the narrowest realistic beam is about 20 degrees.
    - Directional noise -> beamforming should dominate by placing nulls on the noise.
    - Diffuse noise -> NR should dominate because beamforming cannot suppress it well.
    - If the user prefers natural sound, stay conservative and avoid over-processing.
    """

    scene_situation: str = dspy.InputField(desc="L5 scene description")
    scene_challenges: str = dspy.InputField(desc="JSON list of L5-identified challenges")
    user_preferences: str = dspy.InputField(desc="User preferences JSON")
    mic_geometry: str = dspy.InputField(desc="Microphone array geometry")

    primary_challenge: str = dspy.OutputField(
        desc="Primary acoustic challenge: 'directional_noise' | 'diffuse_noise' | 'reverberation' | 'quiet'"
    )
    beam_nr_coordination: str = dspy.OutputField(
        desc="Coordination instructions for beamforming and NR, injected into downstream PRIM contexts. "
        "Example: 'Aim the beam at 0° and keep speech bands near the beam axis less suppressed by NR.'"
    )
    aggressiveness_budget: str = dspy.OutputField(
        desc="'conservative' | 'moderate' | 'aggressive'"
    )
    planning_reasoning: str = dspy.OutputField(desc="Planning rationale")


class StrategyIntegrateSig(dspy.Signature):
    """
    [ROUTING] Post-integrate the L6 composite after all three PRIMs run.

    Check whether the three sub-strategies conflict and produce a final
    strategy confidence score.

    Common conflicts:
    - The beam targets 30° but NR preserve_bands does not protect that direction.
    - NR aggressiveness is 0.8 while the user prefers natural sound.
    - beam_width < 20° violates physical constraints.
    - The gain compression ratio is so high that the sound would become unnatural.

    You may fine-tune NR aggressiveness to resolve conflicts,
    but do not make large changes (keep them within ±0.2).
    """

    beam_summary: str = dspy.InputField(
        desc="Beam result summary: azimuth, width, nulls, reasoning"
    )
    nr_summary: str = dspy.InputField(
        desc="NR result summary: method, aggressiveness, preserve_bands, reasoning"
    )
    gain_summary: str = dspy.InputField(
        desc="Gain result summary: per-frequency gains, compression ratio"
    )
    coordination_plan: str = dspy.InputField(
        desc="Coordination plan proposed by the phase-1 router"
    )
    user_preferences: str = dspy.InputField(desc="User preferences JSON")

    has_conflict: bool = dspy.OutputField(
        desc="Whether the three sub-strategies conflict with one another"
    )
    conflict_description: str = dspy.OutputField(
        desc="Describe the conflict if one exists; otherwise write 'none'"
    )
    adjusted_nr_aggressiveness: float = dspy.OutputField(
        desc="Integrated NR aggressiveness [0,1], optionally adjusted to fit the beam and preferences"
    )
    overall_confidence: float = dspy.OutputField(desc="Overall strategy confidence [0,1]")
    integration_reasoning: str = dspy.OutputField(desc="Integration reasoning")
