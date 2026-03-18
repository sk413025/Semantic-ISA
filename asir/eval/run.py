"""
ASIR Evaluation Runner

用法:
  PYTHONUTF8=1 python -X utf8 -m asir.eval          # L1-L3 only (no API key)
  PYTHONUTF8=1 python -X utf8 -m asir.eval --full    # L1-L7 + pipeline (需 API key)

輸出: per-layer scores + detailed failures
"""
import sys
import os
import argparse
import json
import numpy as np

from asir.types import RawSignal
from asir.primitives import (
    prim_sample_audio, prim_fft, prim_estimate_noise_psd,
    prim_beamform, comp_spectral_subtract,
    prim_extract_mfcc, prim_estimate_snr, prim_estimate_rt60,
    comp_extract_full_features, prim_generate_gain_params,
)


def run_l1_l3_eval():
    """L1-L3 確定性層評估，不需要 API key。"""
    print("=" * 60)
    print("  ASIR Evaluation — L1-L3 Deterministic Layers")
    print("=" * 60)

    results = {}
    total_pass = 0
    total_checks = 0

    # --- L1: Physical Sensing ---
    print("\nL1 Physical Sensing:")
    checks = []

    sig = prim_sample_audio(duration_ms=32.0, n_channels=2)
    ok = sig.n_channels == 2 and sig.sample_rate == 16000
    checks.append(("sample_audio shape", ok))

    arr = np.array(sig.samples[0])
    ok = np.all(np.isfinite(arr)) and np.max(np.abs(arr)) < 100
    checks.append(("sample_audio values finite", ok))

    for name, passed in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}")
        total_checks += 1
        if passed:
            total_pass += 1
    results["L1"] = checks

    # --- L2: Signal Processing ---
    print("\nL2 Signal Processing:")
    checks = []

    fft_result = prim_fft(sig)
    ok = "magnitude" in fft_result and len(fft_result["magnitude"]) > 0
    checks.append(("FFT structure", ok))

    bf = prim_beamform(sig, target_azimuth_deg=0.0)
    ok = len(bf) == len(sig.samples[0])
    checks.append(("beamform output length", ok))

    noise_psd = prim_estimate_noise_psd(sig)
    ok = len(noise_psd) == len(fft_result["magnitude"])
    checks.append(("noise PSD length", ok))

    cleaned = comp_spectral_subtract(sig, noise_psd, alpha=1.0)
    ok = len(cleaned) > 0 and np.all(np.isfinite(cleaned))
    checks.append(("spectral subtract finite", ok))

    for name, passed in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}")
        total_checks += 1
        if passed:
            total_pass += 1
    results["L2"] = checks

    # --- L3: Acoustic Features ---
    print("\nL3 Acoustic Features:")
    checks = []

    features = comp_extract_full_features(sig)
    ok = isinstance(features.snr_db, float) and isinstance(features.rt60_s, float)
    checks.append(("feature types", ok))

    ok = features.n_active_sources >= 1
    checks.append(("n_active_sources >= 1", ok))

    ok = features.temporal_pattern in ("stationary", "modulated", "impulsive")
    checks.append(("temporal_pattern valid", ok))

    mfcc = prim_extract_mfcc(sig)
    ok = "Energy" in mfcc
    checks.append(("MFCC summary contains Energy", ok))

    # NAL-NL2 gain
    audiogram = '{"250":30,"500":35,"1000":40,"2000":50,"4000":60}'
    gain = prim_generate_gain_params(audiogram, "test")
    ok = gain["deterministic"] is True
    checks.append(("NAL-NL2 deterministic", ok))

    gpf = gain["gain_per_frequency"]
    ok = gpf["4000"] > gpf["250"]
    checks.append(("gain increases with hearing loss", ok))

    ok = 1.0 <= gain["compression_ratio"] <= 4.0
    checks.append(("compression ratio in [1, 4]", ok))

    for name, passed in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}")
        total_checks += 1
        if passed:
            total_pass += 1
    results["L3"] = checks

    print(f"\n--- L1-L3 Summary: {total_pass}/{total_checks} passed ---")
    return results, total_pass, total_checks


def run_l4_l7_eval():
    """L4-L7 語意層評估，需要 API key。"""
    # Load .env
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

    import dspy
    from asir.harness import AcousticSemanticHarness
    from asir.eval.examples import create_eval_examples
    from asir.eval.metrics import (
        check_l4_perceptual, check_l5_scene, check_l6_strategy,
        check_l7_intent, check_pipeline, compute_score,
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Cannot run L4-L7 eval.")
        print("Set it in .env or environment.")
        return None

    print("\n" + "=" * 60)
    print("  ASIR Evaluation — L4-L7 Semantic Layers")
    print("=" * 60)

    # Configure LMs
    fast_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    strong_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    dspy.configure(lm=fast_lm)

    harness = AcousticSemanticHarness(
        fast_lm=fast_lm,
        strong_lm=strong_lm,
        enable_multimodal=False,  # eval 時不用多模態，省成本
    )

    examples = create_eval_examples()
    print(f"\nRunning {len(examples)} evaluation scenarios...\n")

    layer_scores = {"L4": [], "L5": [], "L6": [], "L7": [], "pipeline": []}
    failures = []

    for i, ex in enumerate(examples):
        scenario = ex.scenario
        print(f"  [{i+1}/{len(examples)}] {scenario}...", end=" ", flush=True)

        try:
            raw_signal = prim_sample_audio(duration_ms=32.0, n_channels=2)
            result = harness(
                raw_signal=raw_signal,
                user_action=str(getattr(ex, 'user_action', 'none')),
                user_profile=str(getattr(ex, 'user_profile', '')),
                audiogram_json=str(getattr(
                    ex, 'audiogram_json',
                    '{"250":30,"500":35,"1000":40,"2000":50,"4000":60}'
                )),
            )

            # Per-layer checks
            l4 = check_l4_perceptual(ex, result)
            l5 = check_l5_scene(ex, result)
            l6 = check_l6_strategy(ex, result)
            l7 = check_l7_intent(ex, result)
            pipe = check_pipeline(ex, result)

            s4 = compute_score(l4)
            s5 = compute_score(l5)
            s6 = compute_score(l6)
            s7 = compute_score(l7)
            sp = compute_score(pipe)

            layer_scores["L4"].append(s4)
            layer_scores["L5"].append(s5)
            layer_scores["L6"].append(s6)
            layer_scores["L7"].append(s7)
            layer_scores["pipeline"].append(sp)

            print(f"L4={s4:.0%} L5={s5:.0%} L6={s6:.0%} L7={s7:.0%} pipe={sp:.0%}")

            # Collect failures
            for layer_name, checks in [("L4", l4), ("L5", l5), ("L6", l6), ("L7", l7)]:
                for check_name, (passed, detail) in checks.items():
                    if not passed:
                        failures.append({
                            "scenario": scenario,
                            "layer": layer_name,
                            "check": check_name,
                            "detail": detail,
                        })

        except Exception as e:
            print(f"ERROR: {e}")
            for layer in layer_scores:
                layer_scores[layer].append(0.0)
            failures.append({
                "scenario": scenario,
                "layer": "pipeline",
                "check": "execution",
                "detail": str(e),
            })

    # Summary
    print("\n" + "=" * 60)
    print("  L4-L7 Evaluation Summary")
    print("=" * 60)
    for layer, scores in layer_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {layer:>8}: {avg:.1%}  ({sum(1 for s in scores if s >= 0.5)}/{len(scores)} scenarios >= 50%)")

    if failures:
        print(f"\n  {len(failures)} constraint violations found:")
        for f in failures[:10]:  # 最多顯示 10 筆
            print(f"    [{f['layer']}] {f['scenario']}.{f['check']}: {f['detail'][:80]}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")

    # Save detailed results
    results_path = os.path.join(os.path.dirname(__file__), '..', '..', 'eval_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "layer_scores": {k: [round(s, 3) for s in v] for k, v in layer_scores.items()},
            "failures": failures,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Detailed results saved to: eval_results.json")

    return layer_scores


def main():
    parser = argparse.ArgumentParser(description="ASIR Evaluation Runner")
    parser.add_argument("--full", action="store_true",
                        help="Run L4-L7 semantic evaluation (needs API key)")
    args = parser.parse_args()

    run_l1_l3_eval()

    if args.full:
        run_l4_l7_eval()
    else:
        print("\nTip: run with --full to evaluate L4-L7 semantic layers (needs OPENAI_API_KEY)")


if __name__ == "__main__":
    main()
