from typing import Optional
import dspy
from asir.primitives.scene import ReasonAboutSceneSig
from asir.routing.scene import SceneRoutingSig


class SceneWithHistory(dspy.Module):
    """
    [COMP] 第五層：帶歷史的場景理解
    = reason_about_scene |> ★ scene_router |> (conditional) resolve_contradictions

    改造前：if recent_scenes → 一定跑 resolve_contradictions
    改造後：scene_router 根據場景一致性動態決定是否需要矛盾解決
    """
    def __init__(self):
        super().__init__()
        self.reason_scene = dspy.ChainOfThought(ReasonAboutSceneSig)
        # 矛盾解決也是一個 LLM Primitive
        self.resolve_contradictions = dspy.ChainOfThought(
            "current_scene, recent_history -> resolved_scene: str, "
            "is_scene_change: bool, resolution_reasoning: str"
        )
        # ★ 新增：Routing Predictor — 決定要不要跑矛盾解決
        self.scene_router = dspy.ChainOfThought(SceneRoutingSig)

    def forward(self, percept: dspy.Prediction, user_profile: str,
                recent_scenes: list[str],
                spectrogram: Optional[dspy.Image] = None,
                ) -> dspy.Prediction:
        history_str = " | ".join(recent_scenes[-5:]) if recent_scenes else "No history"

        # === Phase 1: reason_scene 必跑 ===
        # ★ Phase 3: 傳入頻譜圖輔助場景推理
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

        # === Phase 2: Router 決定要不要跑矛盾解決 ===
        routing = self.scene_router(
            current_scene_situation=scene_result.situation,
            current_scene_confidence=float(scene_result.confidence),
            recent_history=history_str,
            history_length=len(recent_scenes)
        )

        # === Phase 3: 根據 routing 決定執行路徑 ===
        if routing.should_resolve and recent_scenes:
            resolved = self.resolve_contradictions(
                current_scene=scene_result.situation,
                recent_history=history_str
            )
            final_situation = resolved.resolved_scene
        else:
            # ★ Router 說不需要 → 跳過矛盾解決，省一次 LLM call
            final_situation = scene_result.situation

        return dspy.Prediction(
            situation=final_situation,
            challenges_json=scene_result.challenges_json,
            preservation_notes_json=scene_result.preservation_notes_json,
            # ★ 信心度由 router 調整，不再只用 reason_scene 的原始值
            confidence=float(routing.adjusted_confidence),
            history_consistency=routing.history_consistency,
            routing_reasoning=routing.routing_reasoning
        )
