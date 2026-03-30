import dspy


class PerceptAggregateRoutingSig(dspy.Signature):
    """
    [ROUTING] L4 composite: inspect the outputs of the three perceptual PRIMs
    and decide how to merge them into a unified perceptual description.

    You do not generate the descriptions yourself; the three PRIMs have already
    done that. Your job is to:
    1. judge the quality and relevance of each description in context,
    2. assign sensible weights,
    3. return an overall confidence that is not excessively dragged down by a
       single low-confidence PRIM.

    Physical heuristics:
    - High SNR (>20 dB) means noise is less critical, so the noise branch can
      receive a lower weight.
    - Many active sources can reduce speech-description reliability because of
      overlap.
    - Cross-check confidences instead of always taking the minimum.
    """

    noise_summary: str = dspy.InputField(
        desc="Noise PRIM summary: type / direction / severity / confidence"
    )
    speech_summary: str = dspy.InputField(
        desc="Speech PRIM summary: speaker count / direction / intelligibility / confidence"
    )
    env_summary: str = dspy.InputField(
        desc="Environment PRIM summary: type / character / confidence"
    )
    snr_db: float = dspy.InputField(desc="Raw SNR in dB for cross-checking")
    n_sources: int = dspy.InputField(desc="Detected number of active sources")

    noise_weight: float = dspy.OutputField(
        desc="Noise-description weight [0,1], reflecting how important noise is in this scene"
    )
    speech_weight: float = dspy.OutputField(desc="Speech-description weight [0,1]")
    env_weight: float = dspy.OutputField(desc="Environment-description weight [0,1]")
    overall_confidence: float = dspy.OutputField(
        desc="Overall post-aggregation confidence [0,1], not necessarily the minimum"
    )
    routing_reasoning: str = dspy.OutputField(
        desc="Why were the weights assigned this way, and which PRIM matters most in this scene?"
    )
