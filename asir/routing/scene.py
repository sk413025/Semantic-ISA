import dspy


class SceneRoutingSig(dspy.Signature):
    """
    [ROUTING] L5 Composite：場景推理完成後，決定是否需要矛盾解決，
    以及場景信心度是否需要因歷史一致性而調整。

    你看到的是：
    1. reason_scene 剛產出的場景判斷
    2. 最近的場景歷史
    你要決定：
    - 新場景跟歷史有沒有明顯矛盾？
    - 如果有，是真的場景轉換還是 reason_scene 判斷錯誤？
    - 啟動矛盾解決值不值得？（它會多花一次 LLM 呼叫）
    """
    current_scene_situation: str = dspy.InputField(
        desc="reason_scene 剛產出的場景描述"
    )
    current_scene_confidence: float = dspy.InputField(
        desc="reason_scene 的信心度 [0,1]"
    )
    recent_history: str = dspy.InputField(
        desc="最近 N 個場景的摘要，用 | 分隔；首次執行為 'No history'"
    )
    history_length: int = dspy.InputField(
        desc="歷史紀錄筆數（0=首次執行）"
    )

    should_resolve: bool = dspy.OutputField(
        desc="是否需要啟動矛盾解決？"
        "歷史為空或場景與歷史一致時不需要。"
    )
    history_consistency: str = dspy.OutputField(
        desc="'consistent'(場景穩定) | 'gradual_shift'(漸變) | "
        "'abrupt_change'(突變，可能是真的也可能是誤判)"
    )
    adjusted_confidence: float = dspy.OutputField(
        desc="調整後信心度 [0,1]。"
        "歷史一致→可提高；突變且原始信心低→應降低"
    )
    routing_reasoning: str = dspy.OutputField(desc="決策理由")
