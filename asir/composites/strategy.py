import dspy

from asir.primitives.strategy import (
    GenerateBeamformingParamsSig,
    GenerateNoiseReductionParamsSig,
    prim_generate_gain_params,
)
from asir.routing.strategy import StrategyIntegrateSig, StrategyPlanSig


class GenerateFullStrategy(dspy.Module):
    """
    [COMP] Layer 6: full strategy generation.
    = strategy_planner -> gen_beam + gen_nr + gain -> strategy_integrator

    Before the refactor, beamforming and noise reduction had no shared view of
    the plan, and the confidence logic was brittle. Now:
      Phase 1 -> the planner defines coordination and injects enriched context
      Phase 2 -> beam / NR / gain run, with beam and NR sharing the plan
      Phase 3 -> the integrator checks conflicts, fine-tunes, and scores confidence
    """

    def __init__(self):
        super().__init__()
        # Original PRIMs, keeping their prompts intact.
        self.gen_beam = dspy.ChainOfThought(GenerateBeamformingParamsSig)
        self.gen_nr = dspy.ChainOfThought(GenerateNoiseReductionParamsSig)
        # New pre-planning and post-integration routing modules.
        self.strategy_planner = dspy.ChainOfThought(StrategyPlanSig)
        self.strategy_integrator = dspy.ChainOfThought(StrategyIntegrateSig)

    def forward(
        self,
        scene: dspy.Prediction,
        user_prefs_str: str,
        audiogram_json: str,
    ) -> dspy.Prediction:
        scene_str = (
            f"Situation: {scene.situation}\n"
            f"Challenges: {scene.challenges_json}\n"
            f"Preservation notes: {scene.preservation_notes_json}"
        )

        # Phase 1: planner always runs first.
        plan = self.strategy_planner(
            scene_situation=scene.situation,
            scene_challenges=scene.challenges_json,
            user_preferences=user_prefs_str,
            mic_geometry="BTE hearing aid, 2 mics, 10mm spacing, linear array",
        )

        # Inject the coordination plan into downstream PRIM context so beam and NR stay aligned.
        enriched_scene_str = (
            f"{scene_str}\n"
            f"[Coordination Plan] {plan.beam_nr_coordination}\n"
            f"[Aggressiveness Budget] {plan.aggressiveness_budget}"
        )

        # Phase 2: run the three PRIMs with independent fallback handling.
        # If beam or NR fails, use a PRIM-level fallback rather than collapsing the whole composite.
        beam_used_fallback = False
        try:
            beam_result = self.gen_beam(
                scene_understanding=enriched_scene_str,
                mic_geometry="BTE hearing aid, 2 mics, 10mm spacing, linear array",
            )
        except Exception as e:
            beam_used_fallback = True
            beam_result = dspy.Prediction(
                target_azimuth_deg=0.0,
                beam_width_deg=60.0,
                null_directions_json="[]",
                reasoning=f"[FALLBACK] gen_beam failed: {str(e)[:100]}. "
                "Using safe defaults: front-facing, wide beam.",
            )

        nr_used_fallback = False
        try:
            nr_result = self.gen_nr(
                scene_understanding=enriched_scene_str,
                user_preferences=user_prefs_str,
            )
        except Exception as e:
            nr_used_fallback = True
            nr_result = dspy.Prediction(
                method="wiener",
                aggressiveness=0.5,
                preserve_bands_json='["low-frequency environmental"]',
                reasoning=f"[FALLBACK] gen_nr failed: {str(e)[:100]}. "
                "Using safe defaults: wiener, moderate aggressiveness.",
            )

        # Deterministic primitive: gain generation should always succeed.
        gain_result = prim_generate_gain_params(audiogram_json, scene_str)

        # Phase 3: the integrator always runs, even if beam or NR used fallback outputs.
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
            user_preferences=user_prefs_str,
        )

        # NR aggressiveness may be fine-tuned by the integrator.
        try:
            final_nr_agg = float(integration.adjusted_nr_aggressiveness)
            final_nr_agg = max(0.0, min(1.0, final_nr_agg))
        except (ValueError, TypeError):
            final_nr_agg = float(nr_result.aggressiveness)

        combined_reasoning = (
            f"[Plan] {plan.planning_reasoning}\n"
            f"[Beam] {beam_result.reasoning}"
            f"{' FALLBACK' if beam_used_fallback else ''}\n"
            f"[NR] {nr_result.reasoning}"
            f"{' FALLBACK' if nr_used_fallback else ''}\n"
            f"[Gain] deterministic NAL-NL2, compression={gain_result['compression_ratio']}\n"
            f"[Integration] {integration.integration_reasoning}"
        )

        # Discount the final confidence if any fallback was used.
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
            nr_used_fallback=nr_used_fallback,
        )
