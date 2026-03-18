import json
import dspy


def create_acoustic_feedback_metric(gold, pred, trace=None,
                                     pred_name=None, pred_trace=None):
    """
    GEPA 的 feedback metric — 這是整個系統的「可靠性契約」

    回傳邏輯：
    - 不管有沒有 pred_name，score 都以 module-level（全管線）為準
    - pred_name 存在時，回傳針對該 predictor 的文字 feedback
      （GEPA 用 feedback 做 reflection，不用 predictor-level score）
    """
    import json

    # ===== 先算 module-level score（全管線）=====
    module_score = 0.0
    module_feedback = []

    has_dsp = hasattr(pred, 'dsp_params') and pred.dsp_params is not None
    if has_dsp:
        module_score += 0.25
        module_feedback.append("OK: DSP params generated.")
    else:
        module_feedback.append("FAIL: No DSP params produced.")

    has_scene_conf = (hasattr(pred, 'scene')
                      and hasattr(pred.scene, 'confidence')
                      and float(pred.scene.confidence) > 0.3)
    if has_scene_conf:
        module_score += 0.25
        module_feedback.append(f"OK: Scene confidence={pred.scene.confidence}.")
    else:
        conf = getattr(getattr(pred, 'scene', None), 'confidence', 'N/A')
        module_feedback.append(
            f"WEAK: Scene confidence={conf} <= 0.3. "
            "Improve perceptual description clarity so scene reasoning has stronger inputs."
        )

    has_reasoning = (hasattr(pred, 'strategy')
                     and hasattr(pred.strategy, 'combined_reasoning')
                     and len(str(pred.strategy.combined_reasoning)) > 30)
    if has_reasoning:
        module_score += 0.25
        module_feedback.append("OK: Strategy reasoning present.")
    else:
        module_feedback.append("WEAK: Strategy reasoning too short or missing.")

    has_percept = (hasattr(pred, 'percept')
                   and hasattr(pred.percept, 'confidence')
                   and float(pred.percept.confidence) > 0.3)
    if has_percept:
        module_score += 0.25
    else:
        module_feedback.append("WEAK: Perceptual description confidence low.")

    module_score = min(module_score, 1.0)

    # ===== 如果 GEPA 要求 predictor-level feedback =====
    if pred_name:
        pred_feedback = []

        if "describe_noise" in pred_name:
            try:
                sources = json.loads(pred.noise_sources_json)
                if len(sources) > 0:
                    pred_feedback.append(f"Good: identified {len(sources)} noise source(s).")
                    if hasattr(gold, 'snr_db') and gold.snr_db > 20:
                        if any(s.get('severity') == 'severe' for s in sources):
                            pred_feedback.append(
                                "PHYSICS VIOLATION: SNR > 20dB but noise rated severe. "
                                "High SNR means noise is low relative to signal."
                            )
                else:
                    pred_feedback.append("No noise sources identified — identify at least one.")
            except Exception:
                pred_feedback.append("Failed to parse noise_sources_json as valid JSON.")

        elif "describe_speech" in pred_name:
            if hasattr(pred, 'n_speakers'):
                pred_feedback.append(f"Detected {pred.n_speakers} speaker(s).")
            if hasattr(pred, 'intelligibility'):
                pred_feedback.append(f"Intelligibility: {pred.intelligibility}.")
            if not pred_feedback:
                pred_feedback.append("Ensure n_speakers and intelligibility are filled.")

        elif "describe_env" in pred_name:
            if hasattr(pred, 'environment_type') and len(str(pred.environment_type)) > 5:
                pred_feedback.append(f"Environment: {pred.environment_type}.")
            else:
                pred_feedback.append("Environment type too vague — be specific (e.g. 'indoor market').")

        elif "reason_scene" in pred_name:
            if hasattr(pred, 'challenges_json'):
                try:
                    challenges = json.loads(pred.challenges_json)
                    if len(challenges) > 0:
                        for c in challenges:
                            if 'physical_cause' not in c or len(str(c.get('physical_cause', ''))) < 10:
                                pred_feedback.append(
                                    f"Challenge '{c.get('challenge','')}' lacks physical cause. "
                                    "Every challenge must trace to a physical mechanism."
                                )
                        if not pred_feedback:
                            pred_feedback.append("Good: all challenges have physical causes.")
                    else:
                        pred_feedback.append("No challenges identified — a noisy scene always has challenges.")
                except Exception:
                    pred_feedback.append("challenges_json is not valid JSON.")
            if hasattr(pred, 'preservation_notes_json'):
                try:
                    notes = json.loads(pred.preservation_notes_json)
                    if len(notes) == 0:
                        pred_feedback.append(
                            "No preservation notes. Which sounds help the user navigate?"
                        )
                except Exception:
                    pass

        elif "gen_beam" in pred_name:
            if hasattr(pred, 'beam_width_deg'):
                bw = float(pred.beam_width_deg)
                if bw < 20:
                    pred_feedback.append(
                        f"PHYSICS VIOLATION: beam_width {bw}° < 20°. "
                        "A 2-mic BTE with 10mm spacing cannot form beams < 20°."
                    )
                else:
                    pred_feedback.append(f"Beam width {bw}° respects physical minimum.")
            if hasattr(pred, 'reasoning') and len(str(pred.reasoning)) < 30:
                pred_feedback.append("Provide detailed reasoning for direction choice.")

        elif "gen_nr" in pred_name:
            if hasattr(pred, 'aggressiveness'):
                agg = float(pred.aggressiveness)
                if hasattr(gold, 'user_prefs_natural') and gold.user_prefs_natural and agg > 0.8:
                    pred_feedback.append(
                        f"User prefers 'natural' but aggressiveness={agg} > 0.8. "
                        "Natural preference typically maps to 0.3-0.5."
                    )

        # ★ 新增：Routing Predictor feedback ★

        elif "aggregate_router" in pred_name:
            # L4 聚合路由的 feedback
            if hasattr(pred, 'noise_weight'):
                try:
                    nw = float(pred.noise_weight)
                    sw = float(pred.speech_weight)
                    ew = float(pred.env_weight)
                    total = nw + sw + ew
                    if total < 0.5 or total > 2.0:
                        pred_feedback.append(
                            f"Weights sum={total:.2f} — should be close to 1.0. "
                            "Normalize so downstream can interpret as relative importance."
                        )
                    if hasattr(gold, 'snr_db') and gold.snr_db > 20 and nw > 0.5:
                        pred_feedback.append(
                            f"SNR={gold.snr_db}dB (quiet) but noise_weight={nw:.2f} > 0.5. "
                            "In quiet scenes, speech and environment should dominate."
                        )
                    if hasattr(gold, 'snr_db') and gold.snr_db < 5 and nw < 0.2:
                        pred_feedback.append(
                            f"SNR={gold.snr_db}dB (very noisy) but noise_weight={nw:.2f} < 0.2. "
                            "In noisy scenes, noise description is critical for downstream."
                        )
                except (ValueError, TypeError):
                    pred_feedback.append("Could not parse weights as floats.")
            if hasattr(pred, 'overall_confidence'):
                try:
                    conf = float(pred.overall_confidence)
                    if conf > 0.95:
                        pred_feedback.append(
                            "Confidence > 0.95 is overconfident for LLM-based perception."
                        )
                    if conf < 0.1:
                        pred_feedback.append(
                            "Confidence < 0.1 is too pessimistic — some information was extracted."
                        )
                except (ValueError, TypeError):
                    pass

        elif "scene_router" in pred_name:
            # L5 場景路由的 feedback
            if hasattr(pred, 'should_resolve'):
                if hasattr(pred, 'history_length'):
                    try:
                        hl = int(pred.history_length) if not isinstance(pred.history_length, int) else pred.history_length
                    except (ValueError, TypeError):
                        hl = -1
                    if hl == 0 and pred.should_resolve:
                        pred_feedback.append(
                            "No history exists (history_length=0) but should_resolve=True. "
                            "Cannot resolve contradictions without history."
                        )
            if hasattr(pred, 'adjusted_confidence'):
                try:
                    ac = float(pred.adjusted_confidence)
                    if hasattr(pred, 'current_scene_confidence'):
                        orig = float(pred.current_scene_confidence)
                        if abs(ac - orig) > 0.4:
                            pred_feedback.append(
                                f"Adjusted confidence ({ac:.2f}) deviates > 0.4 from "
                                f"original ({orig:.2f}). Large adjustments should have "
                                "strong justification."
                            )
                except (ValueError, TypeError):
                    pass
            if not pred_feedback:
                pred_feedback.append("Scene routing decision looks reasonable.")

        elif "strategy_planner" in pred_name:
            # L6 前置規劃的 feedback
            if hasattr(pred, 'primary_challenge'):
                challenge = str(pred.primary_challenge).strip()
                valid_challenges = {'directional_noise', 'diffuse_noise',
                                    'reverberation', 'quiet'}
                if challenge not in valid_challenges:
                    pred_feedback.append(
                        f"primary_challenge='{challenge}' is not one of "
                        f"{valid_challenges}. Use an exact match."
                    )
                if hasattr(gold, 'snr_db'):
                    if gold.snr_db > 20 and challenge in ('directional_noise', 'diffuse_noise'):
                        pred_feedback.append(
                            f"SNR={gold.snr_db}dB is high but challenge='{challenge}'. "
                            "High SNR means noise is not the primary issue."
                        )
                    if gold.snr_db < 5 and challenge == 'quiet':
                        pred_feedback.append(
                            f"SNR={gold.snr_db}dB is very low but challenge='quiet'. "
                            "This is a noisy scene."
                        )
            if hasattr(pred, 'aggressiveness_budget'):
                budget = str(pred.aggressiveness_budget).strip()
                if hasattr(gold, 'user_prefs_natural') and gold.user_prefs_natural:
                    if budget == 'aggressive':
                        # 除非使用者不滿意，否則偏好自然的人不該用 aggressive
                        if hasattr(gold, 'user_action') and str(gold.user_action) == 'none':
                            pred_feedback.append(
                                "User prefers natural sound but budget='aggressive'. "
                                "Only use aggressive if user explicitly dissatisfied."
                            )
            if hasattr(pred, 'beam_nr_coordination'):
                coord = str(pred.beam_nr_coordination)
                if len(coord) < 20:
                    pred_feedback.append(
                        "Coordination plan is too short. Provide specific guidance "
                        "for how beam and NR should work together."
                    )
            if not pred_feedback:
                pred_feedback.append("Strategy planning looks reasonable.")

        elif "strategy_integrator" in pred_name:
            # L6 後置整合的 feedback
            if hasattr(pred, 'adjusted_nr_aggressiveness'):
                try:
                    agg = float(pred.adjusted_nr_aggressiveness)
                    if agg < 0 or agg > 1:
                        pred_feedback.append(
                            f"adjusted_nr_aggressiveness={agg} out of [0,1] range."
                        )
                    if hasattr(gold, 'user_prefs_natural') and gold.user_prefs_natural:
                        if agg > 0.8:
                            pred_feedback.append(
                                f"User prefers natural but final aggressiveness={agg:.2f}. "
                                "Should be 0.3-0.6 for natural preference."
                            )
                except (ValueError, TypeError):
                    pred_feedback.append(
                        "Could not parse adjusted_nr_aggressiveness as float."
                    )
            if hasattr(pred, 'overall_confidence'):
                try:
                    conf = float(pred.overall_confidence)
                    if conf > 0.95:
                        pred_feedback.append(
                            "Strategy confidence > 0.95 is overconfident."
                        )
                except (ValueError, TypeError):
                    pass
            if not pred_feedback:
                pred_feedback.append("Strategy integration looks reasonable.")

        elif "pipeline_router" in pred_name:
            # 頂層管線路由的 feedback
            if hasattr(pred, 'execution_depth'):
                depth = str(pred.execution_depth).strip().lower()
                valid_depths = {'fast', 'medium', 'full'}
                if depth not in valid_depths:
                    pred_feedback.append(
                        f"execution_depth='{depth}' is not one of {valid_depths}."
                    )
                if hasattr(gold, 'user_action'):
                    if str(gold.user_action) != 'none' and depth != 'full':
                        pred_feedback.append(
                            f"User took action '{gold.user_action}' but depth='{depth}'. "
                            "User actions should trigger full pipeline execution."
                        )
                if hasattr(pred, 'frames_since_full_update'):
                    try:
                        fsfu = int(pred.frames_since_full_update) if not isinstance(
                            pred.frames_since_full_update, int
                        ) else pred.frames_since_full_update
                        if fsfu > 50 and depth == 'fast':
                            pred_feedback.append(
                                f"frames_since_full={fsfu} > 50 but depth='fast'. "
                                "Run at least medium to prevent stale scene data."
                            )
                    except (ValueError, TypeError):
                        pass
            if not pred_feedback:
                pred_feedback.append("Pipeline routing decision looks reasonable.")

        if pred_feedback:
            return dspy.Prediction(
                score=module_score,
                feedback="\n".join(pred_feedback)
            )

    return dspy.Prediction(
        score=module_score,
        feedback="\n".join(module_feedback)
    )
