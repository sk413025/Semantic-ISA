"""
ASIR Semantic Layer Evaluation (L4-L7)

用法:
  PYTHONUTF8=1 python -X utf8 -m asir.eval          # needs OPENAI_API_KEY in .env

L1-L3 確定性層的測試請用 pytest:
  PYTHONUTF8=1 python -X utf8 -m pytest tests/ -v

設計原則:
  - 直接注入 AcousticFeatures → L4，繞過 L1-L3 的隨機訊號
  - 每個 example 的物理參數 (SNR, RT60...) 會被 LLM 直接看到
  - constraint checks 測的是物理合理性，不是 keyword exact match

資料流:
  example (snr_db, rt60_s, ...) → build_features() → AcousticFeatures
  → L4 (perceptual_desc) → L5 (scene_understanding) → L6 (strategy_gen)
  → comp_strategy_to_dsp_params → DSPParameterSet
  → check_l4 / check_l5 / check_l6 / check_dsp_output / check_l7_routing
"""
import sys
import os
import json

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
    """
    把 eval example 的物理參數轉成 AcousticFeatures，直接注入 L4。

    這是 v0.5.1 的核心改善：example 定義的 snr_db=3.0 會直接出現在
    LLM 看到的 features string 裡，而不是被隨機噪音覆蓋。
    """
    from asir.types import AcousticFeatures

    snr = float(ex.snr_db)
    energy = float(ex.energy_db)
    n_src = int(ex.n_active_sources)
    pattern = str(ex.temporal_pattern)

    # 根據場景特性生成合理的 MFCC summary
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


def run_eval():
    """L4-L7 semantic evaluation — 直接呼叫 composites，繞過 harness。"""
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
    print("  Design: AcousticFeatures injected directly → L4")
    print("  L1-L3 tests: python -m pytest tests/ -v")

    # Configure LMs
    fast_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    strong_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    dspy.configure(lm=fast_lm)

    # Instantiate composites directly (not through harness)
    perceptual_desc = FullPerceptualDescription()
    scene_understanding = SceneWithHistory()
    strategy_gen = GenerateFullStrategy()
    pipeline_router = dspy.ChainOfThought(PipelineRoutingSig)

    examples = create_eval_examples()
    print(f"\n  {len(examples)} scenarios\n")

    layer_scores = {"L4": [], "L5": [], "L6": [], "DSP": [], "L7": []}
    failures = []

    for i, ex in enumerate(examples):
        scenario = ex.scenario
        print(f"  [{i+1}/{len(examples)}] {scenario}...", end=" ", flush=True)

        try:
            # === Build AcousticFeatures from example params ===
            features = build_features(ex)

            # === L4: Perceptual Description ===
            with dspy.context(lm=fast_lm):
                percept = perceptual_desc(
                    acoustic_features=features,
                    user_context=str(ex.user_profile),
                )

            # === L5: Scene Understanding ===
            with dspy.context(lm=strong_lm):
                scene = scene_understanding(
                    percept=percept,
                    user_profile=str(ex.user_profile),
                    recent_scenes=[],
                )

            # === L6: Strategy Generation ===
            prefs = '{"noise_tolerance":"medium","processing_preference":"natural"}'
            with dspy.context(lm=strong_lm):
                strategy = strategy_gen(
                    scene=scene,
                    user_prefs_str=prefs,
                    audiogram_json=str(ex.audiogram_json),
                )

            # === L6→L2: Translation to DSP params ===
            dsp_params = comp_strategy_to_dsp_params(strategy)

            # === L7: Pipeline Router ===
            user_action = str(getattr(ex, 'user_action', 'none'))
            with dspy.context(lm=fast_lm):
                routing = pipeline_router(
                    signal_change_magnitude=1.0,
                    last_scene_confidence=0.5,
                    last_strategy_confidence=0.5,
                    user_action=user_action,
                    frames_since_full_update=10,
                )

            # === Build unified prediction for metrics ===
            pred = dspy.Prediction(
                percept=percept,
                scene=scene,
                strategy=strategy,
                dsp_params=dsp_params,
                execution_depth=str(routing.execution_depth).strip().lower(),
            )

            # === Per-layer constraint checks ===
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

            # Collect failures
            for layer_name, checks in [
                ("L4", l4), ("L5", l5), ("L6", l6), ("DSP", dsp), ("L7", l7),
            ]:
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
                "scenario": scenario, "layer": "pipeline",
                "check": "execution", "detail": str(e),
            })

    # === Summary ===
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    for layer, scores in layer_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            above_half = sum(1 for s in scores if s >= 0.5)
            print(f"  {layer:>4}: {avg:.1%}  ({above_half}/{len(scores)} scenarios >= 50%)")

    if failures:
        print(f"\n  {len(failures)} constraint violations:")
        for f in failures[:15]:
            print(f"    [{f['layer']}] {f['scenario']}.{f['check']}: {f['detail'][:80]}")
        if len(failures) > 15:
            print(f"    ... and {len(failures) - 15} more")

    # Save detailed results
    results_path = os.path.join(os.path.dirname(__file__), '..', '..', 'eval_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "scenarios": [ex.scenario for ex in examples],
            "layer_scores": {k: [round(s, 3) for s in v] for k, v in layer_scores.items()},
            "failures": failures,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: eval_results.json")

    # Exit code: fail if any layer avg < 40%
    all_avg = {k: sum(v)/len(v) for k, v in layer_scores.items() if v}
    if any(avg < 0.4 for avg in all_avg.values()):
        print("\n  ⚠ Some layers below 40% — check failures above")
        return layer_scores
    return layer_scores


def main():
    run_eval()


if __name__ == "__main__":
    main()
