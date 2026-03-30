"""
ASIR Integration Evaluation — End-to-End with Real Audio

Usage:
  PYTHONUTF8=1 python -X utf8 -m asir.eval --integration
  PYTHONUTF8=1 python -X utf8 -m asir.eval --integration --program programs/gepa_xxx/program.json

Prerequisites:
  1. Generate test audio: python -m asir.eval.generate_audio
  2. OPENAI_API_KEY in .env

Results are logged to MLflow for history and comparison.
"""
import os
import sys
import json
from pathlib import Path

import mlflow

from asir.eval.examples import create_eval_examples
from asir.eval.metrics import (
    check_l4_perceptual, check_l5_scene,
    check_l6_strategy, check_dsp_output, check_l7_routing,
    compute_score,
)

BASE_DIR = Path(__file__).parent.parent.parent
SCENARIO_DIR = Path(__file__).parent / "audio" / "scenarios"


def _load_env():
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())


def _load_scenario_wav(scenario_name):
    from asir.primitives.signal import prim_load_audio
    wav_path = SCENARIO_DIR / f"{scenario_name}.wav"
    if not wav_path.exists():
        return None
    signal = prim_load_audio(str(wav_path))
    if isinstance(signal, tuple):
        return signal[0]
    return signal


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
    print(f"\n    ┌─ Trace: {scenario}")

    l4 = trace.get("L4", {})
    if l4:
        print(f"    │ L4 noise:  {l4['noise_description'][:150]}")
        print(f"    │ L4 speech: {l4['speech_description'][:100]}")
        print(f"    │ L4 conf:   {l4['confidence']}")

    l5 = trace.get("L5", {})
    if l5:
        print(f"    │ L5 scene:  {l5['situation'][:200]}")
        print(f"    │ L5 conf:   {l5['confidence']}")

    l6 = trace.get("L6", {})
    if l6:
        print(f"    │ L6 beam:   azimuth={l6['beam_azimuth_deg']}°, width={l6['beam_width_deg']}°")
        print(f"    │ L6 NR:     agg={l6['nr_aggressiveness']}")
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


# ===== Test A: L3 Feature Extraction =====

def test_feature_extraction(examples):
    from asir.primitives.features import comp_extract_full_features

    print("\n  Test A: L3 Feature Extraction on Real Audio")
    print("  " + "-" * 50)

    results = []
    for ex in examples:
        name = ex.scenario
        signal = _load_scenario_wav(name)
        if signal is None:
            print(f"  [{name}] SKIP — WAV not found")
            results.append({"scenario": name, "status": "skip"})
            continue

        try:
            features = comp_extract_full_features(signal)
            target_snr = float(ex.snr_db)
            actual_snr = features.snr_db

            checks = {}
            snr_diff = abs(actual_snr - target_snr)
            checks["snr_reasonable"] = (
                snr_diff < 20,
                f"target={target_snr:.1f}dB, actual={actual_snr:.1f}dB, diff={snr_diff:.1f}dB"
            )
            checks["energy_valid"] = (
                20 < features.energy_db < 100,
                f"energy={features.energy_db:.1f}dB"
            )
            checks["temporal_valid"] = (
                features.temporal_pattern in ("stationary", "modulated", "impulsive"),
                f"pattern={features.temporal_pattern}"
            )
            checks["sources_valid"] = (
                1 <= features.n_active_sources <= 10,
                f"n_sources={features.n_active_sources}"
            )
            checks["mfcc_present"] = (
                len(features.mfcc_summary) > 10,
                f"mfcc_len={len(features.mfcc_summary)}"
            )

            score = compute_score(checks)
            status = "PASS" if score >= 0.8 else "WARN" if score >= 0.5 else "FAIL"
            print(f"  [{name}] {status} {score:.0%} — "
                  f"SNR: {target_snr:.0f}→{actual_snr:.1f}dB, "
                  f"energy={features.energy_db:.0f}dB, "
                  f"pattern={features.temporal_pattern}")

            for check_name, (passed, detail) in checks.items():
                if not passed:
                    print(f"    ✗ {check_name}: {detail}")

            results.append({
                "scenario": name, "status": status, "score": score,
                "snr_target": target_snr, "snr_actual": actual_snr,
                "energy_db": features.energy_db,
            })

        except Exception as e:
            print(f"  [{name}] ERROR — {e}")
            results.append({"scenario": name, "status": "error", "error": str(e)})

    return results


# ===== Test B: End-to-End Pipeline =====

def test_end_to_end(examples, harness=None):
    import dspy
    from asir.harness import AcousticSemanticHarness

    _load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n  Test B: SKIP — OPENAI_API_KEY not set")
        return [], {}, {}

    print("\n  Test B: End-to-End Pipeline with Real Audio")
    print("  " + "-" * 50)

    if harness is None:
        fast_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
        strong_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
        dspy.configure(lm=fast_lm)
        harness = AcousticSemanticHarness(
            fast_lm=fast_lm, strong_lm=strong_lm, enable_multimodal=True,
        )

    e2e_failures = []
    layer_scores = {"L4": [], "L5": [], "L6": [], "DSP": [], "L7": []}
    scenario_details = {}

    for i, ex in enumerate(examples):
        name = ex.scenario
        signal = _load_scenario_wav(name)
        if signal is None:
            print(f"  [{i+1}/{len(examples)}] {name}... SKIP (no WAV)")
            continue

        print(f"  [{i+1}/{len(examples)}] {name}...", end=" ", flush=True)

        try:
            user_action = str(getattr(ex, 'user_action', 'none'))
            result = harness(
                raw_signal=signal,
                user_action=user_action,
                user_profile=str(ex.user_profile),
                audiogram_json=str(ex.audiogram_json),
            )

            pred = dspy.Prediction(
                percept=result.percept,
                scene=result.scene,
                strategy=result.strategy,
                dsp_params=result.dsp_params,
                execution_depth=str(
                    getattr(result, 'execution_depth', 'full')
                ).strip().lower(),
                current_preferences=getattr(result, 'current_preferences', None),
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

            for k, v in [("L4", s4), ("L5", s5), ("L6", s6), ("DSP", sd), ("L7", s7)]:
                layer_scores[k].append(v)

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
                            "scenario": name, "layer": layer_name,
                            "check": check_name, "detail": detail[:100],
                        }
                        scenario_failures.append(entry)
                        e2e_failures.append(entry)

            _print_trace(name, trace, scenario_failures)

            scenario_details[name] = {
                "scores": {"L4": s4, "L5": s5, "L6": s6, "DSP": sd, "L7": s7},
                "trace": trace,
                "checks": all_checks,
            }

        except Exception as e:
            print(f"ERROR — {e}")
            e2e_failures.append({
                "scenario": name, "layer": "pipeline",
                "check": "execution", "detail": str(e)[:200],
            })

        # Reset harness state so preferences do not accumulate across scenarios.
        harness.current_preferences = {
            "noise_tolerance": "medium",
            "processing_preference": "natural",
            "environment_awareness": "moderate",
            "known_situations": ["wet market: enhance the front, preserve environmental awareness"]
        }
        harness.feedback_history = []

    if any(v for v in layer_scores.values()):
        print(f"\n  End-to-End Summary:")
        for layer, scores in layer_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                print(f"    {layer:>4}: {avg:.1%}  "
                      f"({sum(1 for s in scores if s >= 0.5)}/{len(scores)} >= 50%)")

    return e2e_failures, layer_scores, scenario_details


# ===== Main =====

def run_integration(harness=None):
    print("=" * 60)
    print("  ASIR Integration Evaluation — Real Audio")
    print("=" * 60)

    examples = create_eval_examples()
    available = sum(
        1 for ex in examples
        if (SCENARIO_DIR / f"{ex.scenario}.wav").exists()
    )
    print(f"\n  {available}/{len(examples)} scenario WAVs found in {SCENARIO_DIR}")

    if available == 0:
        print("\n  No scenario WAVs found. Generate them first:")
        print("  PYTHONUTF8=1 python -X utf8 -m asir.eval.generate_audio")
        return

    feat_results = test_feature_extraction(examples)
    e2e_failures, e2e_layer_scores, e2e_scenario_details = test_end_to_end(
        examples, harness=harness,
    )

    # === MLflow logging ===
    summary = {}
    if e2e_layer_scores:
        summary = {
            layer: round(sum(scores) / len(scores), 3)
            for layer, scores in e2e_layer_scores.items() if scores
        }

    mlflow.set_experiment("asir-eval")
    with mlflow.start_run(run_name="integration"):
        mlflow.set_tag("eval_type", "integration")
        for layer, avg in summary.items():
            mlflow.log_metric(f"{layer}_avg", avg)
        mlflow.log_metric("num_failures", len(e2e_failures))
        result_data = {
            "summary": summary,
            "scenarios": e2e_scenario_details,
            "feature_extraction": feat_results,
            "failures": e2e_failures,
        }
        mlflow.log_dict(result_data, "integration_results.json")
    print(f"\n  Results logged to MLflow (experiment: asir-eval)")


def main():
    harness = None
    for i, arg in enumerate(sys.argv):
        if arg == "--program" and i + 1 < len(sys.argv):
            program_path = sys.argv[i + 1]
            print(f"  Loading program from: {program_path}")
            import dspy
            from asir.gepa.training import GEPATrainableHarness
            _load_env()
            api_key = os.environ.get("OPENAI_API_KEY")
            fast_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
            strong_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
            dspy.configure(lm=fast_lm)
            program = GEPATrainableHarness(fast_lm=fast_lm, strong_lm=strong_lm)
            program.load(program_path)
            harness = program.harness
            break

    run_integration(harness=harness)


if __name__ == "__main__":
    main()

