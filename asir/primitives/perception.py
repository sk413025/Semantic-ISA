from typing import Optional
import dspy


# --- [PRIM] describe_noise: 從數值翻譯成感知語義 ---
class DescribeNoiseSig(dspy.Signature):
    """
    [PRIM] 第四層：噪音感知描述（多模態版 v0.3）
    BACKEND: LLM (multimodal-aware)
    RELIABILITY: source_identification_accuracy >= 0.80

    將聲學特徵翻譯成人類可理解的噪音描述。
    這是一個不可分解的語義原子操作——因為噪音的
    類型、方向、時間模式之間有強耦合。

    ★ 多模態輸入（v0.3 新增）：
      - audio_clip: 讓支援音訊的 LM 直接「聽」原始音訊
      - spectrogram: 讓所有 vision LM「看到」頻譜結構
      - 兩者都是 Optional：不可用時退回純文字模式
    """
    acoustic_features: str = dspy.InputField(
        desc="聲學特徵的文字描述，包含 SNR、頻譜特性、能量等"
    )
    user_context: str = dspy.InputField(
        desc="使用者上下文：年齡、聽力狀況、當前活動"
    )
    # ★ Phase 3: 多模態輸入
    audio_clip: Optional[dspy.Audio] = dspy.InputField(
        desc="原始音訊片段（支援音訊的 LM 可直接聽取，不可用時為 None）",
        default=None
    )
    spectrogram: Optional[dspy.Image] = dspy.InputField(
        desc="Mel 頻譜圖（所有 vision LM 可看到音訊結構，不可用時為 None）",
        default=None
    )

    noise_sources_json: str = dspy.OutputField(
        desc="JSON 格式的噪音源列表，每個包含 type/direction/temporal/severity"
    )
    confidence: float = dspy.OutputField(
        desc="噪音描述的信心度 [0,1]"
    )


# --- [PRIM] describe_speech: 語音感知描述 ---
class DescribeSpeechSig(dspy.Signature):
    """
    [PRIM] 第四層：語音感知描述（多模態版 v0.3）
    BACKEND: LLM (multimodal-aware)
    RELIABILITY: n_speakers_accuracy >= 0.85 (for 1-4 speakers)
    FAILURE_MODE: >4 speakers -> n_speakers = -1 (meaning "many")

    ★ 語音分析是多模態收益最大的 PRIM——
      直接聽音訊可以判斷說話者數量、方向、可懂度。
    """
    acoustic_features: str = dspy.InputField(
        desc="聲學特徵，特別關注基頻、調變模式、能量包絡"
    )
    # ★ Phase 3: 多模態輸入
    audio_clip: Optional[dspy.Audio] = dspy.InputField(
        desc="原始音訊片段（直接聽取可更準確判斷說話者數量和可懂度）",
        default=None
    )
    spectrogram: Optional[dspy.Image] = dspy.InputField(
        desc="頻譜圖（可看到語音的諧波結構和時間模式）",
        default=None
    )

    n_speakers: int = dspy.OutputField(desc="估計說話者數量，>4 時返回 -1")
    target_direction: str = dspy.OutputField(desc="目標說話者方向描述")
    target_distance: str = dspy.OutputField(desc="估計距離: near/medium/far")
    intelligibility: str = dspy.OutputField(
        desc="可懂度: clear/slightly_masked/heavily_masked/inaudible"
    )
    confidence: float = dspy.OutputField(desc="信心度 [0,1]")


# --- [PRIM] describe_environment: 環境感知描述 ---
class DescribeEnvironmentSig(dspy.Signature):
    """
    [PRIM] 第四層：環境感知描述（多模態版 v0.3）
    BACKEND: LLM (multimodal-aware)
    RELIABILITY: environment_type_accuracy >= 0.75
    FAILURE_MODE: novel_environment -> confidence < 0.5

    ★ 環境辨識特別受益於頻譜圖——
      混響特徵、背景噪音紋理在視覺上非常明顯。
    """
    acoustic_features: str = dspy.InputField(
        desc="聲學特徵，關注混響、頻譜分布、時域模式"
    )
    # ★ Phase 3: 多模態輸入
    audio_clip: Optional[dspy.Audio] = dspy.InputField(
        desc="原始音訊片段（直接聽環境聲可辨識場景類型）",
        default=None
    )
    spectrogram: Optional[dspy.Image] = dspy.InputField(
        desc="頻譜圖（混響、背景紋理在視覺上非常明顯）",
        default=None
    )

    environment_type: str = dspy.OutputField(desc="環境類型描述")
    acoustic_character: str = dspy.OutputField(desc="聲學特性描述")
    confidence: float = dspy.OutputField(desc="環境判斷信心度 [0,1]")
