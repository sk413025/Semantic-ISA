import dspy


class PipelineRoutingSig(dspy.Signature):
    """
    [ROUTING] Top-level composite: decide how deep the stack should run before
    processing each frame.

    Hearing aids have strict latency budgets:
    - fast (<10 ms): L1->L2 only, reuse cached DSP params
    - medium (<500 ms): L1->L5, refresh the scene while reusing cached strategy
    - full (>500 ms): L1->L7, perform a full update

    Decision rules:
    - Small signal change + high previous confidence -> fast
    - Moderate signal change -> medium
    - Large change / user action / too long since the last full pass -> full
    """

    signal_change_magnitude: float = dspy.InputField(
        desc="Magnitude of change relative to the previous frame [0,1]"
    )
    last_scene_confidence: float = dspy.InputField(
        desc="Confidence of the last scene estimate [0,1]"
    )
    last_strategy_confidence: float = dspy.InputField(
        desc="Confidence of the last strategy output [0,1]"
    )
    user_action: str = dspy.InputField(
        desc="User action, for example 'none' or 'button_press:dissatisfied'"
    )
    frames_since_full_update: int = dspy.InputField(
        desc="Number of frames since the last full seven-layer update"
    )

    execution_depth: str = dspy.OutputField(desc="'fast' | 'medium' | 'full'")
    force_strategy_update: bool = dspy.OutputField(
        desc="Whether to force a strategy refresh even if the depth is medium"
    )
    routing_reasoning: str = dspy.OutputField(
        desc="Why this execution depth was selected"
    )
