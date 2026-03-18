import dspy


class PipelineRoutingSig(dspy.Signature):
    """
    [ROUTING] 頂層 Composite：每一幀執行前，決定要跑哪些層。

    助聽器有嚴格的延遲預算：
    - fast (<10ms): L1→L2 only，用 cached DSP params
    - medium (<500ms): L1→L5，更新場景但用 cached 策略
    - full (>500ms): L1→L7，完整更新

    決策依據：
    - 信號變化小 + 上次信心高 → fast（省電、低延遲）
    - 信號變化中等 → medium（更新場景就好）
    - 信號劇烈變化 / 使用者有動作 / 太久沒完整更新 → full
    """
    signal_change_magnitude: float = dspy.InputField(
        desc="當前信號相比上一幀的變化量 [0,1]"
    )
    last_scene_confidence: float = dspy.InputField(
        desc="上一次場景理解的信心度 [0,1]"
    )
    last_strategy_confidence: float = dspy.InputField(
        desc="上一次策略的信心度 [0,1]"
    )
    user_action: str = dspy.InputField(
        desc="使用者動作：'none' | 'button_press:dissatisfied' | ..."
    )
    frames_since_full_update: int = dspy.InputField(
        desc="距離上一次完整七層更新的幀數"
    )

    execution_depth: str = dspy.OutputField(
        desc="'fast' | 'medium' | 'full'"
    )
    force_strategy_update: bool = dspy.OutputField(
        desc="是否強制更新策略？即使 depth 是 medium"
    )
    routing_reasoning: str = dspy.OutputField(desc="選擇此深度的理由")
