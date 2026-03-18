import dspy


class PerceptAggregateRoutingSig(dspy.Signature):
    """
    [ROUTING] L4 Composite：觀察三個感知 PRIM 的輸出後，
    決定如何整合為統一的感知描述。

    你不做感知描述——三個 PRIM 已經做完了。
    你的職責：
    1. 判斷三個描述各自的品質和在此場景中的重要性
    2. 給出合理的權重分配
    3. 給出不受單一低信心 PRIM 過度拖累的整體信心度

    物理常識提示：
    - SNR 高（>20dB）→ 噪音不嚴重，noise 描述的權重可以降低
    - 聲源數多 → speech 描述的可靠性可能下降（多人重疊）
    - 三個 PRIM 的 confidence 應交叉驗證，不只取 min
    """
    noise_summary: str = dspy.InputField(
        desc="噪音 PRIM 輸出摘要：類型/方向/嚴重度/confidence"
    )
    speech_summary: str = dspy.InputField(
        desc="語音 PRIM 輸出摘要：人數/方向/可懂度/confidence"
    )
    env_summary: str = dspy.InputField(
        desc="環境 PRIM 輸出摘要：類型/特性/confidence"
    )
    snr_db: float = dspy.InputField(desc="原始 SNR(dB)，用於交叉驗證")
    n_sources: int = dspy.InputField(desc="偵測到的聲源數量")

    noise_weight: float = dspy.OutputField(
        desc="噪音描述權重 [0,1]，反映此場景中噪音描述的重要性"
    )
    speech_weight: float = dspy.OutputField(
        desc="語音描述權重 [0,1]"
    )
    env_weight: float = dspy.OutputField(
        desc="環境描述權重 [0,1]"
    )
    overall_confidence: float = dspy.OutputField(
        desc="整合後整體信心度 [0,1]，不一定是三者的 min"
    )
    routing_reasoning: str = dspy.OutputField(
        desc="為什麼這樣分配權重？哪個 PRIM 在此場景最關鍵？"
    )
