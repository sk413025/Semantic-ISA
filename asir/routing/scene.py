import dspy


class SceneRoutingSig(dspy.Signature):
    """
    [ROUTING] L5 composite: after scene reasoning completes, decide whether a
    contradiction-resolution pass is worth running and whether scene confidence
    should be adjusted based on historical consistency.

    Inputs:
    1. the fresh scene hypothesis from reason_scene,
    2. recent scene history.

    You should decide:
    - whether the new scene obviously contradicts history,
    - whether the mismatch looks like a real scene change or a mistake,
    - whether paying for another LLM call is justified.
    """

    current_scene_situation: str = dspy.InputField(
        desc="The scene description just produced by reason_scene"
    )
    current_scene_confidence: float = dspy.InputField(
        desc="reason_scene confidence [0,1]"
    )
    recent_history: str = dspy.InputField(
        desc="Summary of the most recent N scenes, separated by |; first run uses 'No history'"
    )
    history_length: int = dspy.InputField(
        desc="Number of historical scene records (0 means first run)"
    )

    should_resolve: bool = dspy.OutputField(
        desc="Whether contradiction resolution should be triggered. No history or strong agreement should keep this false."
    )
    history_consistency: str = dspy.OutputField(
        desc="'consistent' | 'gradual_shift' | 'abrupt_change'"
    )
    adjusted_confidence: float = dspy.OutputField(
        desc="Adjusted confidence [0,1]. Agreement with history can raise it; abrupt low-confidence mismatches should lower it."
    )
    routing_reasoning: str = dspy.OutputField(
        desc="Reasoning behind the routing decision"
    )
