"""
ASIR Integration Evaluation — End-to-End with Real Audio

Tests:
  A. L3 Feature Extraction: scenario WAV → AcousticFeatures → validate ranges
  B. End-to-End Pipeline: scenario WAV → harness.forward() → validate output

Usage:
  PYTHONUTF8=1 python -X utf8 -m asir.eval --integration
  PYTHONUTF8=1 python -X utf8 -m asir.eval --integration --verbose
  PYTHONUTF8=1 python -X utf8 -m asir.eval --integration --program programs/gepa_xxx/program.json

Prerequisites:
  1. Generate test audio first:
     PYTHONUTF8=1 python -X utf8 -m asir.eval.generate_audio
  2. Needs OPENAI_API_KEY in .env for L4-L7 (Test B)
"""
import os
import sys
import json
from pathlib import Path

from asir.eval.examples import create_eval_examples
from asir.eval.metrics import (
    check_l4_perceptual, check_l5_scene,
    check_l6_strategy, check_dsp_output, check_l7_routing,
    compute_score,
)

BASE_DIR = Path(__file__).parent.parent.parent
SCENARIO_DIR = BASE_DIR / "examples" / "audio" / "scenarios"


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
    """Load scenario WAV as RawSignal."""
    from asir.primitives.signal import prim_load_audio
    wav_path = SCENARIO_DIR / f"{scenario_name}.wav"
    if not wav_path.exists():
        return None
    signal = prim_load_audio(str(wav_path))
    # prim_load_audio returns (RawSignal, Optional[dspy.Audio]) or just RawSignal
    if isinstance(signal, tuple):
        return signal[0]
    return signal


def _print_trace(scenario, pred, failures_for_scenario):
    """
    ★ v0.9: 印出場景的推理 trace，用於 fail 時的根因分析。
    """
    print(f"\n    ┌─ Trace: {scenario}")

    percept = getattr(pred, 'percept', None)
    if percept:
        noise = str(getattr(percept, 'noise_description', ''))[:150]
        speech = str(getattr(percept, 'speech_description', ''))[:100]
        conf = getattr(percept, 'confidence', '?')
        print(f"    │ L4 noise:  {noise}")
        print(f"    │ L4 speech: {speech}")
        print(f"    │ L4 conf:   {conf}")

    scene = getattr(pred, 'scene', None)
    if scene:
        sit = str(getattr(scene, 'situation', ''))[:200]
        conf = getattr(scene, 'confidence', '?')
        print(f"    │ L5 scene:  {sit}")
        print(f"    │ L5 conf:   {conf}")

    strategy = getattr(pred, 'strategy', None)
    if strategy:
        az = getattr(strategy, 'target_azimuth_deg', '?')
        bw = getattr(strategy, 'beam_width_deg', '?')
        nr = getattr(strategy, 'nr_aggressiveness', '?')
        reasoning = str(getattr(strategy, 'combined_reasoning', ''))[:200]
        print(f"    │ L6 beam:   azimuth={az}°, width={bw}°")
        print(f"    │ L6 NR:     agg={nr}")
        print(f"    │ L6 reason: {reasoning}")

    dsp = getattr(pred, 'dsp_params', None)
    if dsp:
        mask = getattr(dsp, 'noise_mask', None)
        if mask and isinstance(mask, (list, tuple)) and len(mask) > 0:
            mask_vals = [float(v) for v in mask]
            print(f"    │ DSP mask:  min={min(mask_vals):.3f}, "
                  f"max={max(mask_vals):.3f}, len={len(mask_vals)}")

    prefs = getattr(pred, 'current_preferences', None)
    if prefs:
        print(f"    │ L7 prefs:  {prefs}")

    depth = getattr(pred, 'execution_depth', '?')
    print(f"    │ L7 depth:  {depth}")

    if failures_for_scenario:
        print(f"    │")
        print(f"    │ Failed checks ({len(failures_for_scenario)}):")
        for f in failures_for_scenario:
            print(f"    │   [{f['layer']}] {f['check']}: {f['detail'][:80]}")

    print(f"    └─")


# ===== Test A: L3 Feature Extraction =====

def test_feature_extraction(examples):
    """
    Test A: 真實音檔 → comp_extract_full_features() → 特徵值合理嗎？

    驗證 L3 特徵提取在真實音訊上不會崩潰，
    且估計的 SNR 跟混音時設定的 SNR 在合理誤差內。
    """
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

            # SNR sanity: within ±20dB of target (generous, because
            # synthetic noise mixing and simplified SNR estimator differ)
            snr_diff = abs(actual_snr - target_snr)
            checks["snr_reasonable"] = (
                snr_diff < 20,
                f"target={target_snr:.1f}dB, actual={actual_snr:.1f}dB, diff={snr_diff:.1f}dB"
            )

            # Energy in valid range (20-100 dB SPL)
            checks["energy_valid"] = (
                20 < features.energy_db < 100,
                f"energy={features.energy_db:.1f}dB"
            )

            # Temporal pattern is valid
            checks["temporal_valid"] = (
                features.temporal_pattern in ("stationary", "modulated", "impulsive"),
                f"pattern={features.temporal_pattern}"
            )

            # n_active_sources is positive
            checks["sources_valid"] = (
                1 <= features.n_active_sources <= 10,
                f"n_sources={features.n_active_sources}"
            )

            # MFCC summary is non-empty
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

            # Show failures
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

def test_end_to_end(examples, harness=None, verbose=False):
    """
    Test B: 真實音檔 → harness.forward() → 驗證完整 pipeline。

    這是專案宣稱的核心 I/O：
    RawSignal + user_action + audiogram → DSPParameterSet

    驗證：pipeline 不崩潰、DSP 輸出完整、語意推理合理（復用 metrics.py）。

    Args:
        harness: 可選的已初始化 harness（用於 A/B 對比）。
        verbose: 印出所有場景的 trace（不只 fail 的）。
    """
    import dspy
    from asir.harness import AcousticSemanticHarness

    _load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n  Test B: SKIP — OPENAI_API_KEY not set")
        return []

    print("\n  Test B: End-to-End Pipeline with Real Audio")
    print("  " + "-" * 50)

    if harness is None:
        fast_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
        strong_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
        dspy.configure(lm=fast_lm)

        harness = AcousticSemanticHarness(
            fast_lm=fast_lm,
            strong_lm=strong_lm,
            enable_multimodal=False,  # Test without multimodal first
        )

    results = []
    layer_scores = {"L4": [], "L5": [], "L6": [], "DSP": [], "L7": []}

    for i, ex in enumerate(examples):
        name = ex.scenario
        signal = _load_scenario_wav(name)
        if signal is None:
            print(f"  [{i+1}/{len(examples)}] {name}... SKIP (no WAV)")
            continue

        print(f"  [{i+1}/{len(examples)}] {name}...", end=" ", flush=True)

        try:
            user_action = str(getattr(ex, 'user_action', 'none'))
            user_profile = str(ex.user_profile)
            audiogram = str(ex.audiogram_json)

            result = harness(
                raw_signal=signal,
                user_action=user_action,
                user_profile=user_profile,
                audiogram_json=audiogram,
            )

            # Build prediction for metrics (same structure as semantic eval)
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

            # Run constraint checks (reuse metrics.py)
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

            # Collect failures for this scenario
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
                        results.append(entry)

            # ★ v0.9: Trace — 每個場景都印摘要（方便目視確認數值合理性）
            _print_trace(name, pred, scenario_failures)

        except Exception as e:
            print(f"ERROR — {e}")
            results.append({
                "scenario": name, "layer": "pipeline",
                "check": "execution", "detail": str(e)[:200],
            })

        # ★ v0.8: 重置 harness 狀態，避免場景間偏好累積互相影響
        harness.current_preferences = {
            "noise_tolerance": "medium",
            "processing_preference": "natural",
            "environment_awareness": "moderate",
            "known_situations": ["菜市場: 增強正前方, 保留環境感"]
        }
        harness.feedback_history = []

    # Summary
    if any(v for v in layer_scores.values()):
        print(f"\n  End-to-End Summary:")
        for layer, scores in layer_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                print(f"    {layer:>4}: {avg:.1%}  "
                      f"({sum(1 for s in scores if s >= 0.5)}/{len(scores)} >= 50%)")

    return results, layer_scores


def _auto_compare_integration(prev_path, current_scores, current_scenarios):
    """★ v0.9: 自動讀取上一次 integration 結果，印 delta。"""
    if not prev_path.exists():
        return
    try:
        with open(prev_path, 'r', encoding='utf-8') as f:
            prev = json.load(f)
        prev_scores = prev.get("layer_scores", {})
        prev_ts = prev.get("timestamp", "?")
    except (json.JSONDecodeError, KeyError):
        return
    if not prev_scores:
        return

    layers = ["L4", "L5", "L6", "DSP", "L7"]
    print(f"\n  vs Previous Integration ({prev_ts}):")
    for layer in layers:
        prev_vals = prev_scores.get(layer, [])
        curr_vals = current_scores.get(layer, [])
        if not prev_vals or not curr_vals:
            continue
        prev_avg = sum(prev_vals) / len(prev_vals)
        curr_avg = sum(curr_vals) / len(curr_vals)
        delta = curr_avg - prev_avg
        if abs(delta) > 0.005:
            marker = "+" if delta > 0 else ""
            print(f"    {layer:>4}: {prev_avg:.1%} → {curr_avg:.1%}  ({marker}{delta:.1%})")
        else:
            print(f"    {layer:>4}: {curr_avg:.1%}  (unchanged)")


# ===== Main =====

def run_integration(harness=None, verbose=False):
    """Run all integration tests."""
    print("=" * 60)
    print("  ASIR Integration Evaluation — Real Audio")
    print("=" * 60)

    # Check if scenario WAVs exist
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

    # Test A: Feature extraction
    feat_results = test_feature_extraction(examples)

    # Test B: End-to-end pipeline
    e2e_failures, e2e_layer_scores = test_end_to_end(
        examples, harness=harness, verbose=verbose
    )

    # ★ v0.9: 存檔 — 時間戳結果 + 覆蓋 latest + 自動比對
    from datetime import datetime
    scenarios = [ex.scenario for ex in examples]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_data = {
        "timestamp": timestamp,
        "scenarios": scenarios,
        "feature_extraction": feat_results,
        "end_to_end_failures": e2e_failures,
        "layer_scores": {
            k: [round(s, 3) for s in v] for k, v in e2e_layer_scores.items()
        } if e2e_layer_scores else {},
    }

    # 存到 results/ 目錄（時間戳，累積歷史）
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts_path = results_dir / f"integration_{timestamp}.json"
    with open(ts_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # 自動比對上一次 integration 結果
    latest_path = BASE_DIR / "integration_results.json"
    if e2e_layer_scores:
        _auto_compare_integration(latest_path, e2e_layer_scores, scenarios)

    # 覆蓋 latest
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: integration_results.json + {ts_path.name}")


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # ★ v0.9: 支援載入優化後的 program
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

    run_integration(harness=harness, verbose=verbose)


if __name__ == "__main__":
    main()
