import dspy


class ParseIntentSig(dspy.Signature):
    """
    [PRIM] 第七層：意圖解析
    BACKEND: LLM
    RELIABILITY: primary_goal_extraction >= 0.90

    使用者可能不會每次都說話——可能只是按了一個按鈕。
    這時 intent 從歷史偏好和當前場景推斷。
    """
    user_action: str = dspy.InputField(
        desc="使用者動作：自然語言指令 或 'button_press:dissatisfied' 或 'none'"
    )
    current_scene: str = dspy.InputField(desc="當前場景理解摘要")
    user_history: str = dspy.InputField(desc="使用者歷史偏好和行為模式")

    primary_goal: str = dspy.OutputField(desc="推斷的主要目標")
    secondary_goals_json: str = dspy.OutputField(desc="JSON: 次要目標列表")
    constraints_json: str = dspy.OutputField(desc="JSON: 使用者約束列表")
    confidence: float = dspy.OutputField(desc="意圖解析信心度 [0,1]")


class UpdatePreferencesSig(dspy.Signature):
    """
    [PRIM] 第七層：偏好更新
    BACKEND: LLM
    RELIABILITY: preference_consistency >= 0.85

    ★ 從單次回饋推斷長期偏好變化是不可分解的語義推理。
      一個二元信號（滿意/不滿意）加上場景 context，
      要推斷偏好向量的哪個維度需要調整。
    """
    current_preferences: str = dspy.InputField(desc="當前偏好設定")
    user_feedback: str = dspy.InputField(
        desc="使用者回饋：button_press/verbal/implicit_behavior"
    )
    current_scene: str = dspy.InputField(desc="回饋發生時的場景")
    feedback_history: str = dspy.InputField(desc="最近 N 次回饋的摘要")

    updated_preferences_json: str = dspy.OutputField(desc="JSON: 更新後的偏好")
    change_reasoning: str = dspy.OutputField(desc="偏好變更的推理過程")
    drift_detected: bool = dspy.OutputField(
        desc="是否偵測到偏好漂移（如聽力退化導致的偏好變化）"
    )
