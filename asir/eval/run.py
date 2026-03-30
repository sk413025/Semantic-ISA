"""
ASIR Semantic Layer Evaluation (L4-L7)

Usage:
  PYTHONUTF8=1 python -X utf8 -m asir.eval
  PYTHONUTF8=1 python -X utf8 -m asir.eval --program programs/gepa_xxx/program.json

Results are logged to MLflow for history and comparison.
"""
import sys
import os
import json

import mlflow

from asir.eval.examples import create_eval_examples
from asir.eval.metrics import (
    check_l4_perceptual, check_l5_scene,
    check_l6_strategy, check_dsp_output, check_l7_routing,
    compute_score,
)


def _load_env():
    """Load .env file if present."""
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())


def build_features(ex):
    """Convert eval-example physics into `AcousticFeatures` for direct L4 injection."""
    from asir.types import AcousticFeatures

    snr = float(ex.snr_db)
    energy = float(ex.energy_db)
    n_src = int(ex.n_active_sources)
    pattern = str(ex.temporal_pattern)

    if snr < 5 or n_src > 3:
        dist = "broadband"
        centroid_bins = 400
    else:
        dist = "low-frequency dominant"
        centroid_bins = 150

    mfcc = (
        f"Energy: {10**(energy/10):.0f}, "
        f"Spectral centroid: {centroid_bins} bins, "
        f"Rolloff: 300 bins, "
        f"Distribution: {dist}"
    )

    return AcousticFeatures(
        mfcc_summary=mfcc,
        snr_db=snr,
        rt60_s=float(ex.rt60_s),
        pitch_hz=None,
        n_active_sources=n_src,
        spectral_centroid_hz=float(centroid_bins * 16000 / (2 * 512)),
        energy_db=energy,
        temporal_pattern=pattern,
    )


def _safe_float(val):
    if val is None:
        return None
    try:
        return round(float(val), 4)
    except (ValueError, TypeError):
        return None


def _build_trace(pred):
    """Extract a structured trace dict for console output and MLflow logging."""
    trace = {}

    percept = getattr(pred, 'percept', None)
    if percept:
        trace["L4"] = {
            "noise_description": str(getattr(percept, 'noise_description', '')),
            "speech_description": str(getattr(percept, 'speech_description', '')),
            "environment_description": str(getattr(percept, 'environment_description', '')),
            "confidence": _safe_float(getattr(percept, 'confidence', None)),
        }

    scene = getattr(pred, 'scene', None)
    if scene:
        trace["L5"] = {
            "situation": str(getattr(scene, 'situation', '')),
            "confidence": _safe_float(getattr(scene, 'confidence', None)),
        }

    strategy = getattr(pred, 'strategy', None)
    if strategy:
        trace["L6"] = {
            "beam_azimuth_deg": _safe_float(getattr(strategy, 'target_azimuth_deg', None)),
            "beam_width_deg": _safe_float(getattr(strategy, 'beam_width_deg', None)),
            "nr_aggressiveness": _safe_float(getattr(strategy, 'nr_aggressiveness', None)),
            "compression_ratio": _safe_float(getattr(strategy, 'compression_ratio', None)),
            "reasoning": str(getattr(strategy, 'combined_reasoning', '')),
        }

    dsp = getattr(pred, 'dsp_params', None)
    if dsp:
        mask = getattr(dsp, 'noise_mask', None)
        if mask and isinstance(mask, (list, tuple)) and len(mask) > 0:
            mask_vals = [float(v) for v in mask]
            trace["DSP"] = {
                "noise_mask_min": round(min(mask_vals), 4),
                "noise_mask_max": round(max(mask_vals), 4),
                "noise_mask_mean": round(sum(mask_vals) / len(mask_vals), 4),
                "noise_mask_len": len(mask_vals),
            }

    trace["L7"] = {
        "execution_depth": str(getattr(pred, 'execution_depth', '?')),
    }
    prefs = getattr(pred, 'current_preferences', None)
    if prefs:
        trace["L7"]["preferences"] = prefs

    return trace


def _serialize_checks(check_results):
    return {
        name: {"passed": passed, "detail": detail}
        for name, (passed, detail) in check_results.items()
    }


def _print_trace(scenario, trace, failures_for_scenario):
    """Print the reasoning trace for one scenario."""
    print(f"\n    ┌─ Trace: {scenario}")

    l4 = trace.get("L4", {})
    if l4:
        print(f"    │ L4 noise:  {l4['noise_description'][:150]}")
        print(f"    │ L4 speech: {l4['speech_description'][:100]}")
        print(f"    │ L4 env:    {l4['environment_description'][:100]}")
        print(f"    │ L4 conf:   {l4['confidence']}")

    l5 = trace.get("L5", {})
    if l5:
        print(f"    │ L5 scene:  {l5['situation'][:200]}")
        print(f"    │ L5 conf:   {l5['confidence']}")

    l6 = trace.get("L6", {})
    if l6:
        print(f"    │ L6 beam:   azimuth={l6['beam_azimuth_deg']}°, width={l6['beam_width_deg']}°")
        print(f"    │ L6 NR:     agg={l6['nr_aggressiveness']}")
        print(f"    │ L6 comp:   ratio={l6['compression_ratio']}")
        print(f"    │ L6 reason: {l6['reasoning'][:200]}")

    dsp_t = trace.get("DSP", {})
    if dsp_t:
        print(f"    │ DSP mask:  min={dsp_t['noise_mask_min']:.3f}, "
              f"max={dsp_t['noise_mask_max']:.3f}, len={dsp_t['noise_mask_len']}")

    l7 = trace.get("L7", {})
    if l7.get("preferences"):
        print(f"    │ L7 prefs:  {l7['preferences']}")
    print(f"    │ L7 depth:  {l7.get('execution_depth', '?')}")

    if failures_for_scenario:
        print(f"    │")
        print(f"    │ Failed checks ({len(failures_for_scenario)}):")
        for f in failures_for_scenario:
            print(f"    │   [{f['layer']}] {f['check']}: {f['detail'][:80]}")

    print(f"    └─")


def run_eval(program=None):
    """L4-L7 semantic evaluation using composites directly, bypassing the harness."""
    _load_env()

    import dspy
    from asir.composites import (
        FullPerceptualDescription, SceneWithHistory,
        GenerateFullStrategy, comp_strategy_to_dsp_params,
    )
    from asir.routing.pipeline import PipelineRoutingSig

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Set it in .env or environment.")
        return None

    print("=" * 60)
    print("  ASIR Evaluation — L4-L7 Semantic Layers")
    print("=" * 60)

    fast_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    strong_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    dspy.configure(lm=fast_lm)

    if program is not None:
        harness = getattr(program, 'harness', program)
        perceptual_desc = harness.perceptual_desc
        scene_understanding = harness.scene_understanding
        strategy_gen = harness.strategy_gen
        pipeline_router = harness.pipeline_router
        program_label = "optimized"
    else:
        perceptual_desc = FullPerceptualDescription()
        scene_understanding = SceneWithHistory()
        strategy_gen = GenerateFullStrategy()
        pipeline_router = dspy.ChainOfThought(PipelineRoutingSig)
        program_label = "baseline"

    examples = create_eval_examples()
    print(f"\n  {len(examples)} scenarios ({program_label})\n")

    layer_scores = {"L4": [], "L5": [], "L6": [], "DSP": [], "L7": []}
    failures = []
    scenario_details = {}

    for i, ex in enumerate(examples):
        scenario = ex.scenario
        print(f"  [{i+1}/{len(examples)}] {scenario}...", end=" ", flush=True)

        try:
            features = build_features(ex)

            with dspy.context(lm=fast_lm):
                percept = perceptual_desc(
                    acoustic_features=features,
                    user_context=str(ex.user_profile),
                )

            with dspy.context(lm=strong_lm):
                scene = scene_understanding(
                    percept=percept,
                    user_profile=str(ex.user_profile),
                    recent_scenes=[],
                )

            prefs = {"noise_tolerance": "medium", "processing_preference": "natural"}
            user_action = str(getattr(ex, 'user_action', 'none'))

            # L7 preference feedback: when user_action exists, update preferences.
            if user_action != "none":
                from asir.primitives.intent import ParseIntentSig, UpdatePreferencesSig
                with dspy.context(lm=strong_lm):
                    pref_update = dspy.ChainOfThought(UpdatePreferencesSig)(
                        current_preferences=json.dumps(prefs, ensure_ascii=False),
                        user_feedback=user_action,
                        current_scene=scene.situation,
                        feedback_history="",
                    )
                try:
                    updated = json.loads(pref_update.updated_preferences_json)
                    prefs.update(updated)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

            prefs_str = json.dumps(prefs, ensure_ascii=False)
            with dspy.context(lm=strong_lm):
                strategy = strategy_gen(
                    scene=scene,
                    user_prefs_str=prefs_str,
                    audiogram_json=str(ex.audiogram_json),
                )

            dsp_params = comp_strategy_to_dsp_params(strategy)

            with dspy.context(lm=fast_lm):
                routing = pipeline_router(
                    signal_change_magnitude=1.0,
                    last_scene_confidence=0.5,
                    last_strategy_confidence=0.5,
                    user_action=user_action,
                    frames_since_full_update=10,
                )

            pred = dspy.Prediction(
                percept=percept,
                scene=scene,
                strategy=strategy,
                dsp_params=dsp_params,
                execution_depth=str(routing.execution_depth).strip().lower(),
                current_preferences=prefs if isinstance(prefs, dict) else None,
            )

            l4 = check_l4_perceptual(ex, pred)
            l5 = check_l5_scene(ex, pred)
            l6 = check_l6_strategy(ex, pred)
            dsp = check_dsp_output(ex, pred)
            l7 = check_l7_routing(ex, pred)

            s4 = compute_score(l4)
            s5 = compute_score(l5)
            s6 = compute_score(l6)
            sd = compute_score(dsp)
            s7 = compute_score(l7)

            layer_scores["L4"].append(s4)
            layer_scores["L5"].append(s5)
            layer_scores["L6"].append(s6)
            layer_scores["DSP"].append(sd)
            layer_scores["L7"].append(s7)

            print(f"L4={s4:.0%} L5={s5:.0%} L6={s6:.0%} DSP={sd:.0%} L7={s7:.0%}")

            trace = _build_trace(pred)
            all_checks = {
                "L4": _serialize_checks(l4),
                "L5": _serialize_checks(l5),
                "L6": _serialize_checks(l6),
                "DSP": _serialize_checks(dsp),
                "L7": _serialize_checks(l7),
            }

            scenario_failures = []
            for layer_name, checks in [
                ("L4", l4), ("L5", l5), ("L6", l6), ("DSP", dsp), ("L7", l7),
            ]:
                for check_name, (passed, detail) in checks.items():
                    if not passed:
                        entry = {
                            "scenario": scenario,
                            "layer": layer_name,
                            "check": check_name,
                            "detail": detail,
                        }
                        scenario_failures.append(entry)
                        failures.append(entry)

            _print_trace(scenario, trace, scenario_failures)

            scenario_details[scenario] = {
                "scores": {"L4": s4, "L5": s5, "L6": s6, "DSP": sd, "L7": s7},
                "trace": trace,
                "checks": all_checks,
            }

        except Exception as e:
            print(f"ERROR: {e}")
            for layer in layer_scores:
                layer_scores[layer].append(0.0)
            failures.append({
                "scenario": scenario, "layer": "pipeline",
                "check": "execution", "detail": str(e),
            })

    # === Summary ===
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    summary = {}
    for layer, scores in layer_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            summary[layer] = round(avg, 3)
            above_half = sum(1 for s in scores if s >= 0.5)
            print(f"  {layer:>4}: {avg:.1%}  ({above_half}/{len(scores)} scenarios >= 50%)")

    if failures:
        print(f"\n  {len(failures)} constraint violations:")
        for f in failures[:15]:
            print(f"    [{f['layer']}] {f['scenario']}.{f['check']}: {f['detail'][:80]}")

    # === MLflow logging ===
    mlflow.set_experiment("asir-eval")
    with mlflow.start_run(run_name=f"semantic_{program_label}"):
        mlflow.set_tag("eval_type", "semantic")
        mlflow.set_tag("program", program_label)
        for layer, avg in summary.items():
            mlflow.log_metric(f"{layer}_avg", avg)
        mlflow.log_metric("num_failures", len(failures))
        result_data = {
            "program": program_label,
            "summary": summary,
            "scenarios": scenario_details,
            "failures": failures,
        }
        mlflow.log_dict(result_data, "eval_results.json")
    print(f"\n  Results logged to MLflow (experiment: asir-eval)")

    return layer_scores


def main():
    program = None
    for i, arg in enumerate(sys.argv):
        if arg == "--program" and i + 1 < len(sys.argv):
            program_path = sys.argv[i + 1]
            print(f"  Loading program from: {program_path}")
            import dspy
            from asir.gepa.training import GEPATrainableHarness
            _load_env()
            api_key = os.environ.get("OPENAI_API_KEY")
            fast_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
            program = GEPATrainableHarness(fast_lm=fast_lm, strong_lm=fast_lm)
            program.load(program_path)
            break

    run_eval(program=program)


if __name__ == "__main__":
    main()
