import dspy


class ParseIntentSig(dspy.Signature):
    """
    [PRIM] Layer 7: intent parsing.
    BACKEND: LLM
    RELIABILITY: primary_goal_extraction >= 0.90

    Users may not always speak; they might simply press a button. In that
    case, the intent must be inferred from preference history and the current
    scene.
    """

    user_action: str = dspy.InputField(
        desc="User action: natural-language instruction, 'button_press:dissatisfied', or 'none'"
    )
    current_scene: str = dspy.InputField(desc="Summary of the current scene understanding")
    user_history: str = dspy.InputField(desc="Historical user preferences and behavior patterns")

    primary_goal: str = dspy.OutputField(desc="Inferred primary goal")
    secondary_goals_json: str = dspy.OutputField(desc="JSON list of secondary goals")
    constraints_json: str = dspy.OutputField(desc="JSON list of user constraints")
    confidence: float = dspy.OutputField(desc="Intent-parsing confidence [0,1]")


class UpdatePreferencesSig(dspy.Signature):
    """
    [PRIM] Layer 7: preference update.
    BACKEND: LLM
    RELIABILITY: preference_consistency >= 0.85

    Inferring a long-term preference shift from a single piece of feedback is an
    irreducible semantic reasoning task. A binary signal such as satisfaction /
    dissatisfaction must be interpreted in scene context to decide which part of
    the preference vector should change.
    """

    current_preferences: str = dspy.InputField(desc="Current preference settings")
    user_feedback: str = dspy.InputField(
        desc="User feedback: button_press / verbal / implicit_behavior"
    )
    current_scene: str = dspy.InputField(desc="Scene in which the feedback occurred")
    feedback_history: str = dspy.InputField(desc="Summary of the most recent N feedback events")

    updated_preferences_json: str = dspy.OutputField(desc="JSON: updated preferences")
    change_reasoning: str = dspy.OutputField(desc="Reasoning behind the preference update")
    drift_detected: bool = dspy.OutputField(
        desc="Whether preference drift was detected, for example due to hearing changes"
    )
