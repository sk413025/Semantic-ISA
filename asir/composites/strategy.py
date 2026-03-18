import dspy
from asir.primitives.strategy import (
    GenerateBeamformingParamsSig, GenerateNoiseReductionParamsSig,
    prim_generate_gain_params,
)
from asir.routing.strategy import StrategyPlanSig, StrategyIntegrateSig


class GenerateFullStrategy(dspy.Module):
    """
    [COMP] 第六層：完整策略生成
    = ★ strategy_planner → gen_beam + gen_nr + gain → ★ strategy_integrator

    改造前：beam 和 NR 互不知道對方、confidence 公式離譜
    改造後：
      Phase 1 — planner 規劃協作方式，注入 enriched context
      Phase 2 — beam/NR/gain 執行（beam 和 NR 共享協作指令）
      Phase 3 — integrator 檢查衝突、微調、計算信心度
    """
    def __init__(self):
        super().__init__()
        # 原有 PRIM（prompt 不動）
        self.gen_beam = dspy.ChainOfThought(GenerateBeamformingParamsSig)
        self.gen_nr = dspy.ChainOfThought(GenerateNoiseReductionParamsSig)
        # ★ 新增：前置規劃 + 後置整合
        self.strategy_planner = dspy.ChainOfThought(StrategyPlanSig)
        self.strategy_integrator = dspy.ChainOfThought(StrategyIntegrateSig)

    def forward(self, scene: dspy.Prediction, user_prefs_str: str,
                audiogram_json: str) -> dspy.Prediction:
        scene_str = (
            f"Situation: {scene.situation}\n"
            f"Challenges: {scene.challenges_json}\n"
            f"Preservation notes: {scene.preservation_notes_json}"
        )

        # === Phase 1: 前置規劃 — planner 永遠執行 ===
        plan = self.strategy_planner(
            scene_situation=scene.situation,
            scene_challenges=scene.challenges_json,
            user_preferences=user_prefs_str,
            mic_geometry="BTE hearing aid, 2 mics, 10mm spacing, linear array"
        )

        # ★ 把規劃結果注入 PRIM 的 context — 讓 beam 和 NR 知道彼此的大方向
        enriched_scene_str = (
            f"{scene_str}\n"
            f"[Coordination Plan] {plan.beam_nr_coordination}\n"
            f"[Aggressiveness Budget] {plan.aggressiveness_budget}"
        )

        # === Phase 2: 三個 PRIM 執行（各自獨立 try/except）===
        # ★ 修復：gen_beam 和 gen_nr 失敗時用 PRIM 級 fallback，
        #   不再讓整個 Composite fallback — 確保 planner/integrator 有 trace

        beam_used_fallback = False
        try:
            beam_result = self.gen_beam(
                scene_understanding=enriched_scene_str,
                mic_geometry="BTE hearing aid, 2 mics, 10mm spacing, linear array"
            )
        except Exception as e:
            beam_used_fallback = True
            beam_result = dspy.Prediction(
                target_azimuth_deg=0.0,
                beam_width_deg=60.0,
                null_directions_json='[]',
                reasoning=f"[FALLBACK] gen_beam failed: {str(e)[:100]}. "
                          "Using safe defaults: front-facing, wide beam."
            )

        nr_used_fallback = False
        try:
            nr_result = self.gen_nr(
                scene_understanding=enriched_scene_str,
                user_preferences=user_prefs_str
            )
        except Exception as e:
            nr_used_fallback = True
            nr_result = dspy.Prediction(
                method="wiener",
                aggressiveness=0.5,
                preserve_bands_json='["low-frequency environmental"]',
                reasoning=f"[FALLBACK] gen_nr failed: {str(e)[:100]}. "
                          "Using safe defaults: wiener, moderate aggressiveness."
            )

        # Deterministic Primitive: 增益（永遠成功）
        gain_result = prim_generate_gain_params(audiogram_json, scene_str)

        # === Phase 3: 後置整合 — integrator 永遠執行 ===
        # ★ 即使 beam/NR 用了 fallback，integrator 仍然執行
        #   它能看到 [FALLBACK] 標記，做出合理的信心度判斷
        integration = self.strategy_integrator(
            beam_summary=(
                f"azimuth={beam_result.target_azimuth_deg}°, "
                f"width={beam_result.beam_width_deg}°, "
                f"nulls={beam_result.null_directions_json}, "
                f"reasoning={str(beam_result.reasoning)[:200]}"
                f"{' [USED_FALLBACK]' if beam_used_fallback else ''}"
            ),
            nr_summary=(
                f"method={nr_result.method}, "
                f"aggressiveness={nr_result.aggressiveness}, "
                f"preserve={nr_result.preserve_bands_json}, "
                f"reasoning={str(nr_result.reasoning)[:200]}"
                f"{' [USED_FALLBACK]' if nr_used_fallback else ''}"
            ),
            gain_summary=(
                f"gains={gain_result['gain_per_frequency']}, "
                f"compression={gain_result['compression_ratio']}"
            ),
            coordination_plan=plan.beam_nr_coordination,
            user_preferences=user_prefs_str
        )

        # ★ NR aggressiveness 可能被 integrator 微調
        try:
            final_nr_agg = float(integration.adjusted_nr_aggressiveness)
            final_nr_agg = max(0.0, min(1.0, final_nr_agg))  # clamp to [0,1]
        except (ValueError, TypeError):
            final_nr_agg = float(nr_result.aggressiveness)

        combined_reasoning = (
            f"[Plan] {plan.planning_reasoning}\n"
            f"[Beam] {beam_result.reasoning}"
            f"{' ⚠️ FALLBACK' if beam_used_fallback else ''}\n"
            f"[NR] {nr_result.reasoning}"
            f"{' ⚠️ FALLBACK' if nr_used_fallback else ''}\n"
            f"[Gain] deterministic NAL-NL2, compression={gain_result['compression_ratio']}\n"
            f"[Integration] {integration.integration_reasoning}"
        )

        # ★ 如果用了 fallback，整體信心度打折
        try:
            base_confidence = float(integration.overall_confidence)
        except (ValueError, TypeError):
            base_confidence = 0.5
        fallback_penalty = 0.15 * (beam_used_fallback + nr_used_fallback)
        final_confidence = max(0.1, base_confidence - fallback_penalty)

        return dspy.Prediction(
            target_azimuth_deg=float(beam_result.target_azimuth_deg),
            beam_width_deg=float(beam_result.beam_width_deg),
            null_directions_json=beam_result.null_directions_json,
            nr_method=nr_result.method,
            nr_aggressiveness=final_nr_agg,
            preserve_bands_json=nr_result.preserve_bands_json,
            gain_per_frequency=gain_result["gain_per_frequency"],
            compression_ratio=gain_result["compression_ratio"],
            combined_reasoning=combined_reasoning,
            confidence=final_confidence,
            has_conflict=integration.has_conflict,
            conflict_description=integration.conflict_description,
            primary_challenge=plan.primary_challenge,
            aggressiveness_budget=plan.aggressiveness_budget,
            beam_used_fallback=beam_used_fallback,
            nr_used_fallback=nr_used_fallback
        )
