"""
Per-layer constraint metrics for ASIR evaluation.

This module does not use exact-match grading. Instead, it performs
constraint-satisfaction checks. Each `check_*` function returns
`{check_name: (passed: bool, detail: str)}`.

Field coupling:
- `pred.percept.*` comes from `FullPerceptualDescription`
- `pred.scene.*` comes from `SceneWithHistory`
- `pred.strategy.*` comes from `GenerateFullStrategy`
- `pred.dsp_params.*` comes from `comp_strategy_to_dsp_params`
- `pred.execution_depth` comes from `PipelineRoutingSig`
"""

import json


def _check_noise_severity(noise_desc: str, high: bool) -> bool:
    """Parse noise JSON and check severity field values."""
    try:
        data = json.loads(noise_desc)
        sources = data if isinstance(data, list) else data.get("noise_sources", [])
        if not sources:
            return not high

        high_severities = {
            "high",
            "severe",
            "very high",
            "extreme",
            "intense",
            "significant",
            "strong",
            "heavy",
            "loud",
        }
        low_severities = {
            "low",
            "minimal",
            "mild",
            "slight",
            "negligible",
            "none",
            "quiet",
            "soft",
            "faint",
        }
        for src in sources:
            sev = str(src.get("severity", "")).lower().strip()
            if high and sev in high_severities:
                return True
            if not high and sev in low_severities:
                return True
        return False
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False


def _safe_str(obj, default=""):
    try:
        return str(obj).lower()
    except Exception:
        return default


def _has_any_keyword(text, keywords):
    text_lower = _safe_str(text)
    return any(k.lower() in text_lower for k in keywords)


def check_l4_perceptual(example, pred):
    """Check whether the L4 description is consistent with injected features."""
    results = {}
    percept = getattr(pred, "percept", None)

    if percept is None:
        return {"l4_available": (False, "No perceptual description produced")}

    # Noise description vs. SNR: low SNR should sound noisy.
    noise_desc = _safe_str(getattr(percept, "noise_description", ""))
    snr = float(example.snr_db)
    if snr < 5:
        passed = _check_noise_severity(noise_desc, high=True)
        if not passed:
            noisy_words = [
                "loud",
                "noisy",
                "high",
                "significant",
                "severe",
                "multiple",
                "strong",
                "intense",
                "heavy",
                "crowded",
                "busy",
            ]
            passed = _has_any_keyword(noise_desc, noisy_words)
        results["noise_consistent"] = (
            passed,
            f"SNR={snr}dB -> noise description should mention loudness. Got: {noise_desc[:100]}",
        )
    elif snr > 20:
        passed = _check_noise_severity(noise_desc, high=False)
        if not passed:
            quiet_words = ["low", "quiet", "minimal", "mild", "slight", "soft", "faint"]
            passed = _has_any_keyword(noise_desc, quiet_words)
        results["noise_consistent"] = (
            passed,
            f"SNR={snr}dB -> noise should be mild. Got: {noise_desc[:100]}",
        )

    speech_desc = _safe_str(getattr(percept, "speech_description", ""))
    results["speech_present"] = (
        len(speech_desc) > 10,
        f"Speech description length={len(speech_desc)}",
    )

    env_desc = _safe_str(getattr(percept, "environment_description", ""))
    results["env_reasonable"] = (
        len(env_desc) > 10,
        f"Environment description length={len(env_desc)}",
    )

    conf = getattr(percept, "confidence", None)
    if conf is not None:
        try:
            c = float(conf)
            results["confidence_calibrated"] = (0.1 <= c <= 0.95, f"L4 confidence={c:.2f}")
        except (ValueError, TypeError):
            results["confidence_calibrated"] = (False, f"Cannot parse confidence: {conf}")

    return results


def check_l5_scene(example, pred):
    """Check whether L5 scene understanding reflects the acoustic features."""
    results = {}
    scene = getattr(pred, "scene", None)

    if scene is None:
        return {"l5_available": (False, "No scene understanding produced")}

    scene_text = _safe_str(getattr(scene, "situation", ""))

    # Noise-level consistency: noisy scenes should be described as noisy.
    expect_noisy = getattr(example, "expect_noisy", None)
    if expect_noisy is not None:
        if expect_noisy:
            noisy_words = [
                "noisy",
                "loud",
                "noise",
                "crowded",
                "busy",
                "chaotic",
                "multiple speaker",
                "multi-talker",
                "challenging",
                "voices",
            ]
            results["noise_level_consistent"] = (
                _has_any_keyword(scene_text, noisy_words),
                f"SNR={example.snr_db}dB, {example.n_active_sources} sources "
                f"-> scene should describe noisy conditions. Got: {scene_text[:100]}",
            )
        else:
            extreme_noise = [
                "extremely noisy",
                "deafening",
                "unbearable",
                "extremely loud",
                "overwhelming noise",
            ]
            results["noise_level_consistent"] = (
                not _has_any_keyword(scene_text, extreme_noise),
                f"SNR={example.snr_db}dB -> should not be described as extremely noisy. "
                f"Got: {scene_text[:100]}",
            )

    # High RT60 should be reflected in the scene description.
    expect_reverberant = getattr(example, "expect_reverberant", None)
    if expect_reverberant is not None and expect_reverberant:
        reverb_words = [
            "reverb",
            "echo",
            "hall",
            "resonan",
            "large space",
            "spacious",
            "cathedral",
            "reverberant",
            "echoic",
            "resonant",
        ]
        results["reverb_consistent"] = (
            _has_any_keyword(scene_text, reverb_words),
            f"RT60={example.rt60_s}s -> scene should mention reverb. Got: {scene_text[:100]}",
        )

    # Many active sources should be acknowledged.
    n_sources = int(example.n_active_sources)
    if n_sources >= 4:
        multi_words = [
            "multiple",
            "several",
            "many",
            "group",
            "crowd",
            "conversation",
            "speaker",
            "talker",
            "voices",
            "busy",
            "gathering",
            "public space",
            "speech",
            "discussion",
        ]
        results["multi_source_aware"] = (
            _has_any_keyword(scene_text, multi_words),
            f"n_sources={n_sources} -> scene should mention multiple sources. Got: {scene_text[:100]}",
        )

    conf = getattr(scene, "confidence", None)
    if conf is not None:
        try:
            c = float(conf)
            results["scene_confidence"] = (0.1 <= c <= 0.95, f"L5 confidence={c:.2f}")
        except (ValueError, TypeError):
            pass

    return results


def check_l6_strategy(example, pred):
    """Check whether L6 strategy reasoning is coherent and useful."""
    results = {}
    strategy = getattr(pred, "strategy", None)

    if strategy is None:
        return {"l6_available": (False, "No strategy produced")}

    # NR aggressiveness vs. scene noise.
    expect_strong_nr = getattr(example, "expect_strong_nr", None)
    nr_agg = getattr(strategy, "nr_aggressiveness", None)
    if nr_agg is None:
        nr_agg = getattr(strategy, "adjusted_nr_aggressiveness", None)

    if expect_strong_nr is not None and nr_agg is not None:
        try:
            agg = float(nr_agg)
            if expect_strong_nr:
                results["nr_matches_scene"] = (
                    agg >= 0.4,
                    f"Noisy scene -> NR agg={agg:.2f}, expected >= 0.4",
                )
            else:
                results["nr_matches_scene"] = (
                    agg < 0.7,
                    f"Quiet scene -> NR agg={agg:.2f}, expected < 0.7",
                )
        except (ValueError, TypeError):
            pass

    reasoning = _safe_str(getattr(strategy, "combined_reasoning", ""))
    results["strategy_has_reasoning"] = (
        len(reasoning) > 50,
        f"Reasoning length={len(reasoning)}",
    )

    return results


def check_dsp_output(example, pred):
    """
    Check whether generated DSP parameters obey hearing-aid and physics constraints.

    The target behavior is: given an acoustic scene plus hearing-loss profile,
    generate DSP parameters that are meaningful for that situation.
    """
    results = {}
    strategy = getattr(pred, "strategy", None)
    dsp = getattr(pred, "dsp_params", None)

    if strategy is None:
        return {"dsp_available": (False, "No strategy output")}

    # 1. Gain should track the severity of hearing loss.
    gain_gpf = getattr(strategy, "gain_per_frequency", None)
    audiogram_str = getattr(example, "audiogram_json", None)
    if gain_gpf and audiogram_str:
        try:
            audiogram = json.loads(str(audiogram_str))
            max_loss_freq = max(audiogram, key=lambda k: audiogram[k])
            min_loss_freq = min(audiogram, key=lambda k: audiogram[k])
            if audiogram[max_loss_freq] != audiogram[min_loss_freq]:
                results["gain_matches_loss"] = (
                    gain_gpf[max_loss_freq] > gain_gpf[min_loss_freq],
                    f"gain@{max_loss_freq}Hz(loss={audiogram[max_loss_freq]}dB)="
                    f"{gain_gpf[max_loss_freq]:.1f} should > "
                    f"gain@{min_loss_freq}Hz(loss={audiogram[min_loss_freq]}dB)="
                    f"{gain_gpf[min_loss_freq]:.1f}",
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # 2. Severe hearing loss should trigger high gain.
    expect_high_gain = getattr(example, "expect_high_gain", None)
    if expect_high_gain and gain_gpf:
        try:
            max_gain = max(float(v) for v in gain_gpf.values())
            results["high_gain_for_severe_loss"] = (
                max_gain >= 20,
                f"Severe hearing loss -> max gain should be >= 20dB. Got: {max_gain:.1f}dB",
            )
        except (ValueError, TypeError):
            pass

    # 3. NR aggressiveness should track noise level.
    nr_agg = getattr(strategy, "nr_aggressiveness", None)
    if nr_agg is not None:
        snr = float(example.snr_db)
        try:
            agg = float(nr_agg)
            if snr < 5:
                results["nr_matches_noise"] = (
                    agg >= 0.3,
                    f"SNR={snr}dB -> NR agg={agg:.2f}, expected >= 0.3",
                )
            elif snr > 20:
                results["nr_matches_noise"] = (
                    agg <= 0.6,
                    f"SNR={snr}dB -> NR agg={agg:.2f}, expected <= 0.6",
                )
        except (ValueError, TypeError):
            pass

    # 4. Beam focus should match the task.
    beam_width = getattr(strategy, "beam_width_deg", None)
    expect_focus = getattr(example, "expect_beam_focus", None)
    if beam_width is not None and expect_focus is not None:
        try:
            bw = float(beam_width)
            if expect_focus:
                results["beam_appropriate"] = (
                    bw < 90,
                    f"Expected focused beam (< 90°) but width={bw:.0f}°",
                )
            else:
                results["beam_appropriate"] = (
                    bw >= 45,
                    f"Expected wide beam (>= 45°) but width={bw:.0f}°",
                )
        except (ValueError, TypeError):
            pass

    # 4b. Tightened v0.9 constraint: beam-width target range.
    beam_width = getattr(strategy, "beam_width_deg", None)
    expect_bw_range = getattr(example, "expect_beam_width_range", None)
    if beam_width is not None and expect_bw_range is not None:
        try:
            bw = float(beam_width)
            bw_min, bw_max = float(expect_bw_range[0]), float(expect_bw_range[1])
            results["beam_width_targeted"] = (
                bw_min <= bw <= bw_max,
                f"beam_width={bw:.0f}°, target range=[{bw_min:.0f}°, {bw_max:.0f}°]",
            )
        except (ValueError, TypeError, IndexError):
            pass

    # 4c. Tightened v0.9 constraint: point the beam toward the front.
    azimuth = getattr(strategy, "target_azimuth_deg", None)
    expect_front = getattr(example, "expect_beam_azimuth_front", None)
    if azimuth is not None and expect_front:
        try:
            az = float(azimuth)
            results["beam_direction_front"] = (
                -30 <= az <= 30,
                f"azimuth={az:.0f}°, expected approximately 0° (front, tolerance ±30°)",
            )
        except (ValueError, TypeError):
            pass

    # 4d. Tightened v0.9 constraint: NR target range.
    nr_agg_val = getattr(strategy, "nr_aggressiveness", None)
    if nr_agg_val is None:
        nr_agg_val = getattr(strategy, "adjusted_nr_aggressiveness", None)
    expect_nr_range = getattr(example, "expect_nr_range", None)
    if nr_agg_val is not None and expect_nr_range is not None:
        try:
            agg = float(nr_agg_val)
            nr_min, nr_max = float(expect_nr_range[0]), float(expect_nr_range[1])
            results["nr_in_target_range"] = (
                nr_min <= agg <= nr_max,
                f"NR agg={agg:.2f}, target range=[{nr_min:.1f}, {nr_max:.1f}]",
            )
        except (ValueError, TypeError, IndexError):
            pass

    # 4e. Tightened v0.9 constraint: the noise mask must show attenuation.
    expect_mask_active = getattr(example, "expect_noise_mask_active", None)
    if dsp is not None and expect_mask_active:
        mask = getattr(dsp, "noise_mask", None)
        if mask is not None and isinstance(mask, (list, tuple)) and len(mask) > 0:
            try:
                mask_vals = [float(v) for v in mask]
                has_attenuation = any(v < 0.95 for v in mask_vals)
                results["noise_mask_active"] = (
                    has_attenuation,
                    f"noise_mask: min={min(mask_vals):.3f}, max={max(mask_vals):.3f}, "
                    f"expected some bins < 0.95",
                )
            except (ValueError, TypeError):
                pass

    # 5. Compression ratio should stay within a reasonable range.
    cr = getattr(strategy, "compression_ratio", None)
    if cr is not None:
        try:
            results["compression_reasonable"] = (
                1.0 <= float(cr) <= 4.0,
                f"compression_ratio={float(cr):.2f}",
            )
        except (ValueError, TypeError):
            pass

    # 6. DSP parameter structure must be complete.
    if dsp is not None:
        has_beam = hasattr(dsp, "beam_weights") and dsp.beam_weights is not None
        has_mask = hasattr(dsp, "noise_mask") and dsp.noise_mask is not None
        has_filter = hasattr(dsp, "filter_coeffs") and dsp.filter_coeffs is not None
        results["dsp_structure_complete"] = (
            has_beam and has_mask and has_filter,
            f"beam={has_beam}, mask={has_mask}, filter={has_filter}",
        )

    return results


def check_l7_routing(example, pred):
    """Check whether pipeline routing decisions are reasonable."""
    results = {}
    depth = _safe_str(getattr(pred, "execution_depth", "unknown"))
    user_action = str(getattr(example, "user_action", "none"))

    if user_action != "none":
        results["action_triggers_full"] = (
            depth == "full",
            f"User action='{user_action}' but depth='{depth}', expected 'full'",
        )

    if depth in ("fast", "medium", "full"):
        results["depth_valid"] = (True, f"depth='{depth}'")
    else:
        results["depth_valid"] = (False, f"depth='{depth}' not in (fast/medium/full)")

    # Preference persistence check, mainly relevant for integration evaluation.
    expect_pref_updated = getattr(example, "expect_preference_updated", None)
    current_prefs = getattr(pred, "current_preferences", None)
    if expect_pref_updated is not None and current_prefs is not None:
        default_prefs = {
            "noise_tolerance": "medium",
            "processing_preference": "natural",
            "environment_awareness": "moderate",
        }
        prefs_changed = False
        if isinstance(current_prefs, dict):
            for key, default_val in default_prefs.items():
                if key in current_prefs and current_prefs[key] != default_val:
                    prefs_changed = True
                    break
            new_keys = set(current_prefs.keys()) - set(default_prefs.keys()) - {
                "known_situations"
            }
            if new_keys:
                prefs_changed = True

        if expect_pref_updated:
            results["preference_updated"] = (
                prefs_changed,
                f"User action='{user_action}' should update preferences. "
                f"Changed: {prefs_changed}, prefs: {current_prefs}",
            )
        else:
            results["preference_stable"] = (
                not prefs_changed,
                f"User action='{user_action}' is a command, not feedback. "
                f"Preferences should be unchanged. Changed: {prefs_changed}",
            )

    return results


def compute_score(check_results):
    """Convert a check-results dict into a 0-1 score."""
    if not check_results:
        return 0.0
    passed = sum(1 for v, _ in check_results.values() if v)
    return passed / len(check_results)
