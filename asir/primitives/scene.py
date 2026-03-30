from typing import Optional

import dspy


class ReasonAboutSceneSig(dspy.Signature):
    """
    [PRIM] Layer 5: scene reasoning (multimodal v0.3).
    BACKEND: LLM (requires strong reasoning, so a larger model is recommended)
    RELIABILITY: situation_relevance >= 0.80

    Scene understanding is an irreducible primitive because it must reason
    jointly over all perceptual dimensions. For example, deciding that metallic
    transients help a user orient toward a market stall requires understanding
    noise type, user context, and navigation needs at the same time.

    In v0.3, the spectrogram provides a visual overview of cross-modal structure
    so the scene-reasoning model can inspect noise, speech, and environmental
    sound patterns at a glance.
    """

    noise_description: str = dspy.InputField(desc="Layer-4 noise description")
    speech_description: str = dspy.InputField(desc="Layer-4 speech description")
    environment_description: str = dspy.InputField(desc="Layer-4 environment description")
    user_profile: str = dspy.InputField(
        desc="User profile: age, hearing loss, preferences, and current activity"
    )
    recent_scene_history: str = dspy.InputField(
        desc="Summary of the most recent N scene inferences for continuity reasoning"
    )
    spectrogram: Optional[dspy.Image] = dspy.InputField(
        desc="Spectrogram image that supports cross-dimensional reasoning over noise, speech, and environmental sound",
        default=None,
    )

    situation: str = dspy.OutputField(desc="Complete scene description")
    challenges_json: str = dspy.OutputField(
        desc="JSON list of acoustic challenges, each containing challenge / severity / physical_cause"
    )
    preservation_notes_json: str = dspy.OutputField(
        desc="JSON list of environmental cues that should be preserved, with reasons"
    )
    confidence: float = dspy.OutputField(desc="Scene-understanding confidence [0,1]")
