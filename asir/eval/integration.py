"""
ASIR Integration Evaluation — End-to-End with Real Audio

Tests:
  A. L3 Feature Extraction: scenario WAV → AcousticFeatures → validate ranges
  B. End-to-End Pipeline: scenario WAV → harness.forward() → validate output

Usage:
  PYTHONUTF8=1 python -X utf8 -m asir.eval --integration

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

def test_end_to_end(examples):
    """
    Test B: 真實音檔 → harness.forward() → 驗證完整 pipeline。

    這是專案宣稱的核心 I/O：
    RawSignal + user_action + audiogram → DSPParameterSet

    驗證：pipeline 不崩潰、DSP 輸出完整、語意推理合理（復用 metrics.py）。
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

            # Collect failures for display
            for layer_name, checks in [
                ("L4", l4), ("L5", l5), ("L6", l6), ("DSP", dsp), ("L7", l7),
            ]:
                for check_name, (passed, detail) in checks.items():
                    if not passed:
                        results.append({
                            "scenario": name, "layer": layer_name,
                            "check": check_name, "detail": detail[:100],
                        })

        except Exception as e:
            print(f"ERROR — {e}")
            results.append({
                "scenario": name, "layer": "pipeline",
                "check": "execution", "detail": str(e)[:200],
            })

    # Summary
    if any(v for v in layer_scores.values()):
        print(f"\n  End-to-End Summary:")
        for layer, scores in layer_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                print(f"    {layer:>4}: {avg:.1%}  "
                      f"({sum(1 for s in scores if s >= 0.5)}/{len(scores)} >= 50%)")

    return results


# ===== Main =====

def run_integration():
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
    e2e_failures = test_end_to_end(examples)

    # Save results
    results_path = BASE_DIR / "integration_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "feature_extraction": feat_results,
            "end_to_end_failures": e2e_failures,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {results_path.name}")


def main():
    run_integration()


if __name__ == "__main__":
    main()
