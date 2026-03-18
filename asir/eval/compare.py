"""
ASIR A/B Comparison — 比較兩個 program 的 eval 分數

用法:
  # baseline vs 優化後
  PYTHONUTF8=1 python -X utf8 -m asir.eval --compare programs/gepa_xxx/program.json

  # 兩個版本互相比較
  PYTHONUTF8=1 python -X utf8 -m asir.eval --compare programs/gepa_v1/program.json programs/gepa_v2/program.json

設計:
  - 跑同一組 eval examples，對兩個 program 各跑一次
  - 產出並排比較表，逐場景、逐 layer 標示改進/退步
  - 結果存到 compare_results.json
"""
import sys
import os
import json

from asir.eval.run import run_eval, _load_env


def _load_program(program_path):
    """載入 saved program，回傳 GEPATrainableHarness instance。"""
    import dspy
    from asir.gepa.training import GEPATrainableHarness

    _load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    fast_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    dspy.configure(lm=fast_lm)

    program = GEPATrainableHarness(fast_lm=fast_lm, strong_lm=fast_lm)
    program.load(program_path)
    return program


def _print_comparison_table(scenarios, scores_a, scores_b, label_a, label_b):
    """印出 A/B 對比表。"""
    layers = ["L4", "L5", "L6", "DSP", "L7"]

    # Header
    print(f"\n  {'Scenario':<25}", end="")
    for layer in layers:
        print(f" {layer:>10}", end="")
    print()

    print(f"  {'':<25}", end="")
    for _ in layers:
        print(f" {'A → B':>10}", end="")
    print()

    print("  " + "-" * (25 + 11 * len(layers)))

    # Per-scenario rows
    layer_deltas = {l: [] for l in layers}

    for i, scenario in enumerate(scenarios):
        print(f"  {scenario:<25}", end="")
        for layer in layers:
            sa = scores_a[layer][i] if i < len(scores_a.get(layer, [])) else 0.0
            sb = scores_b[layer][i] if i < len(scores_b.get(layer, [])) else 0.0
            delta = sb - sa

            if delta > 0.01:
                marker = "+"
            elif delta < -0.01:
                marker = "-"
            else:
                marker = " "

            print(f" {sa:.0%}→{sb:.0%}{marker}", end="")
            layer_deltas[layer].append(delta)
        print()

    # Summary row
    print("  " + "-" * (25 + 11 * len(layers)))
    print(f"  {'AVG':<25}", end="")
    for layer in layers:
        avg_a = sum(scores_a.get(layer, [0])) / max(len(scores_a.get(layer, [0])), 1)
        avg_b = sum(scores_b.get(layer, [0])) / max(len(scores_b.get(layer, [0])), 1)
        delta = avg_b - avg_a
        if delta > 0.01:
            marker = "+"
        elif delta < -0.01:
            marker = "-"
        else:
            marker = " "
        print(f" {avg_a:.0%}→{avg_b:.0%}{marker}", end="")
    print()

    # Overall delta summary
    print(f"\n  Legend: A = {label_a}, B = {label_b}")
    print(f"  + = B improved, - = B regressed, (blank) = same")

    improved = sum(1 for l in layers for d in layer_deltas[l] if d > 0.01)
    regressed = sum(1 for l in layers for d in layer_deltas[l] if d < -0.01)
    same = sum(1 for l in layers for d in layer_deltas[l] if abs(d) <= 0.01)
    print(f"\n  Cells: {improved} improved, {regressed} regressed, {same} same")


def run_comparison(path_a=None, path_b=None):
    """
    跑 A/B comparison。

    Args:
        path_a: Program A 的路徑（None = baseline，不載入任何 program）
        path_b: Program B 的路徑（必須提供至少一個）
    """
    from asir.eval.examples import create_eval_examples

    print("=" * 60)
    print("  ASIR A/B Comparison")
    print("=" * 60)

    # Determine labels
    label_a = os.path.basename(os.path.dirname(path_a)) if path_a else "baseline"
    label_b = os.path.basename(os.path.dirname(path_b)) if path_b else "baseline"
    print(f"  A: {label_a} {'(' + path_a + ')' if path_a else '(default composites)'}")
    print(f"  B: {label_b} {'(' + path_b + ')' if path_b else '(default composites)'}")

    # Load programs
    program_a = _load_program(path_a) if path_a else None
    program_b = _load_program(path_b) if path_b else None

    examples = create_eval_examples()
    scenarios = [ex.scenario for ex in examples]

    # Run A
    print(f"\n{'='*60}")
    print(f"  Running A: {label_a}")
    print(f"{'='*60}")
    scores_a = run_eval(program=program_a, verbose=False)

    # Run B
    print(f"\n{'='*60}")
    print(f"  Running B: {label_b}")
    print(f"{'='*60}")
    scores_b = run_eval(program=program_b, verbose=False)

    if scores_a is None or scores_b is None:
        print("\n  ERROR: One or both eval runs failed.")
        return

    # Comparison table
    print(f"\n{'='*60}")
    print(f"  A/B Comparison: {label_a} vs {label_b}")
    print(f"{'='*60}")
    _print_comparison_table(scenarios, scores_a, scores_b, label_a, label_b)

    # Save results
    results_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'compare_results.json'
    )
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "a": {"label": label_a, "path": path_a, "scores": scores_a},
            "b": {"label": label_b, "path": path_b, "scores": scores_b},
            "scenarios": scenarios,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Comparison saved to: compare_results.json")


def main():
    """CLI entry point for --compare."""
    # Parse --compare args: expect 1 or 2 paths after --compare
    args = sys.argv[1:]
    paths = []
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--compare":
            continue
        if arg in ("--verbose", "-v"):
            continue
        if arg == "--program":
            skip_next = True
            continue
        if not arg.startswith("-"):
            paths.append(arg)

    if len(paths) == 0:
        print("Usage: python -m asir.eval --compare <program_b.json>")
        print("       python -m asir.eval --compare <program_a.json> <program_b.json>")
        sys.exit(1)
    elif len(paths) == 1:
        # baseline vs optimized
        run_comparison(path_a=None, path_b=paths[0])
    else:
        # two programs
        run_comparison(path_a=paths[0], path_b=paths[1])


if __name__ == "__main__":
    main()
