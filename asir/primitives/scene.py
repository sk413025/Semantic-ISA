from typing import Optional
import dspy


class ReasonAboutSceneSig(dspy.Signature):
    """
    [PRIM] 第五層：場景推理（多模態版 v0.3）
    BACKEND: LLM (需要強推理能力 — 建議用大模型)
    RELIABILITY: situation_relevance >= 0.80

    ★ 這是一條 Primitive，因為「理解場景」不能被分解為更小的
      語義操作——它需要同時考慮所有感知維度並做跨維度推理。

    例如：「金屬碰撞聲可幫助李伯伯定位攤位」這個判斷
    需要同時理解噪音類型、使用者情境、和空間導航需求。

    ★ v0.3: 頻譜圖提供視覺化的跨維度全景——
      場景推理 LM 能一眼看到噪音、語音、環境聲的時頻結構。
    """
    noise_description: str = dspy.InputField(desc="第四層噪音描述")
    speech_description: str = dspy.InputField(desc="第四層語音描述")
    environment_description: str = dspy.InputField(desc="第四層環境描述")
    user_profile: str = dspy.InputField(
        desc="使用者資料：年齡、聽損程度、偏好、當前活動"
    )
    recent_scene_history: str = dspy.InputField(
        desc="最近 N 個場景理解的摘要，用於連續性判斷"
    )
    # ★ Phase 3: 頻譜圖輔助場景推理（音訊太貴，只在 L4 直接聽）
    spectrogram: Optional[dspy.Image] = dspy.InputField(
        desc="頻譜圖（輔助跨維度推理：同時看到噪音、語音、環境聲的時頻結構）",
        default=None
    )

    situation: str = dspy.OutputField(desc="完整場景敘述")
    challenges_json: str = dspy.OutputField(
        desc="JSON: 聲學挑戰列表，每個含 challenge/severity/physical_cause"
    )
    preservation_notes_json: str = dspy.OutputField(
        desc="JSON: 需要保留的環境聲線索列表（附保留理由）"
    )
    confidence: float = dspy.OutputField(desc="場景理解信心度 [0,1]")
