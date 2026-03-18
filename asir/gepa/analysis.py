"""
GEPA Optimization Analysis — 優化結果分析工具

用法：由 compiler.py 在 GEPA 編譯後自動呼叫。
也可獨立使用：
  from asir.gepa.analysis import print_gepa_summary
  print_gepa_summary(optimized_module)
"""


def print_gepa_summary(optimized):
    """印出 GEPA 優化結果摘要（從 detailed_results 讀取）。"""
    print("\n" + "=" * 60)
    print("  GEPA 優化結果摘要")
    print("=" * 60)

    if not hasattr(optimized, 'detailed_results'):
        print("  (track_stats 未啟用，無詳細結果)")
        return

    dr = optimized.detailed_results

    # 基本統計
    total_calls = getattr(dr, 'total_metric_calls', '?')
    num_candidates = getattr(dr, 'num_candidates', '?')
    num_val_evals = getattr(dr, 'num_full_val_evals', '?')
    num_val_inst = getattr(dr, 'num_val_instances', '?')

    print(f"  Total metric calls:    {total_calls}")
    print(f"  Candidates explored:   {num_candidates}")
    print(f"  Full val evaluations:  {num_val_evals}")
    print(f"  Val instances:         {num_val_inst}")

    # Pareto front
    pareto = getattr(dr, 'objective_pareto_front', None)
    if pareto:
        print(f"  Pareto front size:     {len(pareto)}")

    # Per-objective best
    per_obj_best = getattr(dr, 'per_objective_best_candidates', None)
    if per_obj_best and isinstance(per_obj_best, dict):
        print(f"\n  Per-objective best candidates:")
        for obj_name, candidates in per_obj_best.items():
            n = len(candidates) if hasattr(candidates, '__len__') else '?'
            print(f"    {obj_name}: {n} candidate(s)")

    # Val aggregate subscores
    subscores = getattr(dr, 'val_aggregate_subscores', None)
    if subscores and isinstance(subscores, dict):
        print(f"\n  Val aggregate subscores:")
        for key, val in subscores.items():
            if isinstance(val, float):
                print(f"    {key}: {val:.4f}")
            else:
                print(f"    {key}: {val}")

    # Run dir
    run_dir = getattr(dr, 'run_dir', None)
    if run_dir:
        print(f"\n  Artifacts dir: {run_dir}")

    # Seed
    seed = getattr(dr, 'seed', None)
    if seed is not None:
        print(f"  Seed: {seed}")


def print_instruction_diffs(seed_instructions, optimized):
    """
    印出每個 predictor 的 instruction diff（優化前 vs 優化後）。

    Args:
        seed_instructions: dict {predictor_name: original_instruction_str}
        optimized: 優化後的 dspy.Module
    """
    print("\n" + "=" * 60)
    print("  Predictor Instruction Diffs")
    print("  (只顯示有變化的 predictors)")
    print("=" * 60)

    changed = 0
    unchanged = 0

    for name, pred in optimized.named_predictors():
        sig = pred.signature
        after = str(sig.instructions) if hasattr(sig, 'instructions') else str(sig)
        before = seed_instructions.get(name, "")

        if before.strip() == after.strip():
            unchanged += 1
            continue

        changed += 1
        print(f"\n  ★ {name}")
        print(f"    BEFORE: {before[:200]}{'...' if len(before) > 200 else ''}")
        print(f"    AFTER:  {after[:200]}{'...' if len(after) > 200 else ''}")

        # 長度變化
        len_diff = len(after) - len(before)
        sign = "+" if len_diff > 0 else ""
        print(f"    Length: {len(before)} → {len(after)} ({sign}{len_diff})")

    print(f"\n  Summary: {changed} changed, {unchanged} unchanged "
          f"(total {changed + unchanged} predictors)")
