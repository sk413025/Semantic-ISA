"""
ASIR Semantic Layer Evaluation (L4-L7)

用法:
  PYTHONUTF8=1 python -X utf8 -m asir.eval          # needs OPENAI_API_KEY in .env
  PYTHONUTF8=1 python -X utf8 -m asir.eval --verbose # 顯示完整 trace

設計原則:
  - 直接注入 AcousticFeatures → L4，繞過 L1-L3 的隨機訊號
  - 每個 example 的物理參數 (SNR, RT60...) 會被 LLM 直接看到
  - constraint checks 測的是物理合理性，不是 keyword exact match

資料流:
  example (snr_db, rt60_s, ...) → build_features() → AcousticFeatures
  → L4 (perceptual_desc) → L5 (scene_understanding) → L6 (strategy_gen)
  → comp_strategy_to_dsp_params → DSPParameterSet
  → check_l4 / check_l5 / check_l6 / check_dsp_output / check_l7_routing

★ v0.9: 支援 --verbose (完整 trace) + --program (載入優化後 program)
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


def _print_trace(scenario, pred, failures_for_scenario):
    """
    ★ v0.9: 印出場景的推理 trace，用於 fail 時的根因分析。

    只在有 failure 或 --verbose 時呼叫。
    顯示 L4→L5→L6 的推理鏈 + DSP 關鍵數值，方便定位是哪一層導致問題。
    """
    print(f"\n    ┌─ Trace: {scenario}")

    # L4 Perceptual
    percept = getattr(pred, 'percept', None)
    if percept:
        noise = str(getattr(percept, 'noise_description', ''))[:150]
        speech = str(getattr(percept, 'speech_description', ''))[:100]
        env = str(getattr(percept, 'environment_description', ''))[:100]
        conf = getattr(percept, 'confidence', '?')
        print(f"    │ L4 noise:  {noise}")
        print(f"    │ L4 speech: {speech}")
        print(f"    │ L4 env:    {env}")
        print(f"    │ L4 conf:   {conf}")

    # L5 Scene
    scene = getattr(pred, 'scene', None)
    if scene:
        sit = str(getattr(scene, 'situation', ''))[:200]
        conf = getattr(scene, 'confidence', '?')
        print(f"    │ L5 scene:  {sit}")
        print(f"    │ L5 conf:   {conf}")

    # L6 Strategy
    strategy = getattr(pred, 'strategy', None)
    if strategy:
        az = getattr(strategy, 'target_azimuth_deg', '?')
        bw = getattr(strategy, 'beam_width_deg', '?')
        nr = getattr(strategy, 'nr_aggressiveness', '?')
        cr = getattr(strategy, 'compression_ratio', '?')
        reasoning = str(getattr(strategy, 'combined_reasoning', ''))[:200]
        print(f"    │ L6 beam:   azimuth={az}°, width={bw}°")
        print(f"    │ L6 NR:     agg={nr}")
        print(f"    │ L6 comp:   ratio={cr}")
        print(f"    │ L6 reason: {reasoning}")

    # DSP params
    dsp = getattr(pred, 'dsp_params', None)
    if dsp:
        mask = getattr(dsp, 'noise_mask', None)
        if mask and isinstance(mask, (list, tuple)) and len(mask) > 0:
            mask_vals = [float(v) for v in mask]
            print(f"    │ DSP mask:  min={min(mask_vals):.3f}, "
                  f"max={max(mask_vals):.3f}, len={len(mask_vals)}")

    # Execution depth
    depth = getattr(pred, 'execution_depth', '?')
    print(f"    │ L7 depth:  {depth}")

    # Failed checks summary
    if failures_for_scenario:
        print(f"    │")
        print(f"    │ Failed checks ({len(failures_for_scenario)}):")
        for f in failures_for_scenario:
            print(f"    │   [{f['layer']}] {f['check']}: {f['detail'][:80]}")

    print(f"    └─")


def _auto_compare(prev_results_path, current_scores, current_scenarios):
    """
    ★ v0.9: 自動讀取上一次的 eval 結果，印出 delta 比較。

    每次 eval 跑完都會呼叫。如果找到上一次結果，就印出逐 layer 的平均分數變化。
    讓每次跑 eval 都能看到「跟上次比有沒有變好/變差」。
    """
    if not os.path.exists(prev_results_path):
        return

    try:
        with open(prev_results_path, 'r', encoding='utf-8') as f:
            prev = json.load(f)
        prev_scores = prev.get("layer_scores", {})
        prev_program = prev.get("program", "?")
        prev_ts = prev.get("timestamp", "?")
    except (json.JSONDecodeError, KeyError):
        return

    if not prev_scores:
        return

    layers = ["L4", "L5", "L6", "DSP", "L7"]
    has_delta = False

    print(f"\n  vs Previous Run ({prev_program}, {prev_ts}):")
    for layer in layers:
        prev_vals = prev_scores.get(layer, [])
        curr_vals = current_scores.get(layer, [])
        if not prev_vals or not curr_vals:
            continue
        prev_avg = sum(prev_vals) / len(prev_vals)
        curr_avg = sum(curr_vals) / len(curr_vals)
        delta = curr_avg - prev_avg

        if abs(delta) > 0.005:
            has_delta = True
            marker = "+" if delta > 0 else ""
            print(f"    {layer:>4}: {prev_avg:.1%} → {curr_avg:.1%}  ({marker}{delta:.1%})")
        else:
            print(f"    {layer:>4}: {curr_avg:.1%}  (unchanged)")

    # 逐場景 delta（只印有變化的）
    prev_scenario_list = prev.get("scenarios", [])
    if prev_scenario_list == current_scenarios and has_delta:
        print(f"\n  Per-scenario changes:")
        for i, scenario in enumerate(current_scenarios):
            changes = []
            for layer in layers:
                prev_vals = prev_scores.get(layer, [])
                curr_vals = current_scores.get(layer, [])
                if i < len(prev_vals) and i < len(curr_vals):
                    d = curr_vals[i] - prev_vals[i]
                    if abs(d) > 0.01:
                        marker = "+" if d > 0 else ""
                        changes.append(f"{layer}{marker}{d:.0%}")
            if changes:
                print(f"    {scenario}: {', '.join(changes)}")


def run_eval(program=None, verbose=False):
    """
    L4-L7 semantic evaluation — 直接呼叫 composites，繞過 harness。

    Args:
        program: 可選的優化後 Module（GEPATrainableHarness 或 AcousticSemanticHarness）。
                 若提供，用其中的 composites 取代預設實例。
        verbose: 印出所有場景的完整 trace（不只是 fail 的場景）。
    """
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

    # ★ v0.9: 支援載入優化後的 composites
    if program is not None:
        # 從 GEPATrainableHarness 或 AcousticSemanticHarness 取出 composites
        harness = getattr(program, 'harness', program)
        perceptual_desc = harness.perceptual_desc
        scene_understanding = harness.scene_understanding
        strategy_gen = harness.strategy_gen
        pipeline_router = harness.pipeline_router
        program_label = "optimized"
        print(f"  Program: {program_label}")
    else:
        # Instantiate composites directly (not through harness)
        perceptual_desc = FullPerceptualDescription()
        scene_understanding = SceneWithHistory()
        strategy_gen = GenerateFullStrategy()
        pipeline_router = dspy.ChainOfThought(PipelineRoutingSig)
        program_label = "baseline"

    examples = create_eval_examples()
    print(f"\n  {len(examples)} scenarios ({program_label})")
    if verbose:
        print("  Verbose mode: showing full trace for all scenarios\n")
    else:
        print()

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

            # Collect failures for this scenario
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

            # ★ v0.9: Trace — 每個場景都印摘要（方便目視確認數值合理性）
            _print_trace(scenario, pred, scenario_failures)

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

    # ★ v0.9: 存檔 — 時間戳結果 + 覆蓋 latest
    from datetime import datetime
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    scenarios = [ex.scenario for ex in examples]
    result_data = {
        "program": program_label if program is not None else "baseline",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "scenarios": scenarios,
        "layer_scores": {k: [round(s, 3) for s in v] for k, v in layer_scores.items()},
        "failures": failures,
    }

    # 存到 results/ 目錄（時間戳，累積歷史）
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts_path = os.path.join(results_dir, f"eval_{result_data['timestamp']}.json")
    with open(ts_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # 也存到 eval_results.json（覆蓋，方便快速查看最新）
    latest_path = os.path.join(base_dir, 'eval_results.json')

    # ★ v0.9: A/B 自動對比 — 讀取上一次結果，印 delta
    _auto_compare(latest_path, layer_scores, scenarios)

    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: eval_results.json + {os.path.basename(ts_path)}")

    # Exit code: fail if any layer avg < 40%
    all_avg = {k: sum(v)/len(v) for k, v in layer_scores.items() if v}
    if any(avg < 0.4 for avg in all_avg.values()):
        print("\n  ⚠ Some layers below 40% — check failures above")
        return layer_scores
    return layer_scores


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # ★ v0.9: 支援載入優化後的 program
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

    run_eval(program=program, verbose=verbose)


if __name__ == "__main__":
    main()
