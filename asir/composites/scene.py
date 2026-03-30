from typing import Optional

import dspy

from asir.primitives.scene import ReasonAboutSceneSig
from asir.routing.scene import SceneRoutingSig


class SceneWithHistory(dspy.Module):
    """
    [COMP] Layer 5: scene understanding with history.
    = reason_about_scene |> scene_router |> (conditional) resolve_contradictions

    Before the refactor, any non-empty history always triggered contradiction
    resolution. Now the scene router decides dynamically based on scene
    consistency.
    """

    def __init__(self):
        super().__init__()
        self.reason_scene = dspy.ChainOfThought(ReasonAboutSceneSig)
        # Contradiction resolution is itself an LLM primitive.
        self.resolve_contradictions = dspy.ChainOfThought(
            "current_scene, recent_history -> resolved_scene: str, "
            "is_scene_change: bool, resolution_reasoning: str"
        )
        # Learnable routing predictor that decides whether contradiction resolution is worth running.
        self.scene_router = dspy.ChainOfThought(SceneRoutingSig)

    def forward(
        self,
        percept: dspy.Prediction,
        user_profile: str,
        recent_scenes: list[str],
        spectrogram: Optional[dspy.Image] = None,
    ) -> dspy.Prediction:
        history_str = " | ".join(recent_scenes[-5:]) if recent_scenes else "No history"

        # Phase 1: always run reason_scene.
        # Phase 3: pass a spectrogram when available to support scene reasoning.
        scene_kwargs = {
            "noise_description": percept.noise_description,
            "speech_description": percept.speech_description,
            "environment_description": percept.environment_description,
            "user_profile": user_profile,
            "recent_scene_history": history_str,
        }
        if spectrogram is not None:
            scene_kwargs["spectrogram"] = spectrogram

        scene_result = self.reason_scene(**scene_kwargs)

        # Phase 2: ask the router whether contradiction resolution is needed.
        routing = self.scene_router(
            current_scene_situation=scene_result.situation,
            current_scene_confidence=float(scene_result.confidence),
            recent_history=history_str,
            history_length=len(recent_scenes),
        )

        # Phase 3: choose the execution path based on routing.
        if routing.should_resolve and recent_scenes:
            resolved = self.resolve_contradictions(
                current_scene=scene_result.situation,
                recent_history=history_str,
            )
            final_situation = resolved.resolved_scene
        else:
            # Router says resolution is not needed, so skip the extra LLM call.
            final_situation = scene_result.situation

        return dspy.Prediction(
            situation=final_situation,
            challenges_json=scene_result.challenges_json,
            preservation_notes_json=scene_result.preservation_notes_json,
            # Confidence is adjusted by the router instead of blindly using the raw L5 output.
            confidence=float(routing.adjusted_confidence),
            history_consistency=routing.history_consistency,
            routing_reasoning=routing.routing_reasoning,
        )
