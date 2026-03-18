import dspy


class StrategyPlanSig(dspy.Signature):
    """
    [ROUTING] L6 Composite 前置規劃：在 beam/NR/gain 三個 PRIM 執行之前，
    根據場景理解先規劃它們的協作方式。

    你要回答：
    - 此場景的核心聲學挑戰是什麼？
    - beam 和 NR 應該怎麼配合？
    - 總處理預算（保守/中等/積極）是多少？

    物理約束提示：
    - BTE 助聽器只有 2 支麥克風，間距 10mm → beam 最窄 ~20°
    - 方向性噪音 → beam 主導（把 null 對準噪音）
    - 擴散噪音 → NR 主導（beam 幫不了）
    - 使用者偏好「自然」→ 保守預算，避免過度處理
    """
    scene_situation: str = dspy.InputField(desc="L5 場景描述")
    scene_challenges: str = dspy.InputField(desc="L5 識別的挑戰列表 JSON")
    user_preferences: str = dspy.InputField(desc="使用者偏好 JSON")
    mic_geometry: str = dspy.InputField(desc="麥克風陣列幾何")

    primary_challenge: str = dspy.OutputField(
        desc="核心聲學挑戰: "
        "'directional_noise' | 'diffuse_noise' | "
        "'reverberation' | 'quiet'"
    )
    beam_nr_coordination: str = dspy.OutputField(
        desc="beam 和 NR 的協作指令（會被注入各 PRIM 的 context）。"
        "例：'Beam 瞄準前方 0°，NR 應保留 beam 主軸方向的語音頻段'"
    )
    aggressiveness_budget: str = dspy.OutputField(
        desc="'conservative' | 'moderate' | 'aggressive'"
    )
    planning_reasoning: str = dspy.OutputField(desc="規劃理由")


class StrategyIntegrateSig(dspy.Signature):
    """
    [ROUTING] L6 Composite 後置整合：三個 PRIM 都跑完了，
    檢查結果有沒有衝突，給出最終策略信心度。

    常見衝突：
    - beam 瞄 30° 但 NR preserve_bands 沒有保護該方向
    - NR aggressiveness=0.8 但使用者偏好自然（通常 0.3-0.5）
    - beam_width 太窄（<20°）違反物理約束
    - gain 壓縮比太高會讓聲音不自然

    你可以微調 NR aggressiveness 來解決衝突，
    但不要大幅改動（±0.2 以內）。
    """
    beam_summary: str = dspy.InputField(
        desc="beam 結果：azimuth, width, nulls, reasoning"
    )
    nr_summary: str = dspy.InputField(
        desc="NR 結果：method, aggressiveness, preserve_bands, reasoning"
    )
    gain_summary: str = dspy.InputField(
        desc="gain 結果：per-frequency gains, compression ratio"
    )
    coordination_plan: str = dspy.InputField(
        desc="Phase 1 router 規劃的協作指令"
    )
    user_preferences: str = dspy.InputField(desc="使用者偏好 JSON")

    has_conflict: bool = dspy.OutputField(
        desc="三個子策略之間有沒有衝突？"
    )
    conflict_description: str = dspy.OutputField(
        desc="如有衝突，描述是什麼；如無，寫 'none'"
    )
    adjusted_nr_aggressiveness: float = dspy.OutputField(
        desc="整合後 NR 攻擊性 [0,1]，可能微調以配合 beam 和偏好"
    )
    overall_confidence: float = dspy.OutputField(
        desc="策略整體信心度 [0,1]"
    )
    integration_reasoning: str = dspy.OutputField(desc="整合推理")
