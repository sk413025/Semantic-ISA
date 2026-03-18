import json
import os
from datetime import datetime

import dspy
from asir.gepa.metric import create_acoustic_feedback_metric
from asir.gepa.training import create_training_examples, GEPATrainableHarness


def compile_with_gepa(save_program=True):
    """
    ★★★ 用 GEPA 編譯整個管線 ★★★

    GEPA 做的事情類比 LLVM 的 optimization passes：
    1. 它看到每個 Primitive 的失敗案例
    2. 用 reflection_lm 分析失敗原因
    3. 提出改進的 prompt（= 改進的指令）
    4. 在 Pareto frontier 上保留最佳候選

    GEPA 的 feedback function 就是 IR 的可靠性規格。
    物理約束以文字回饋的形式注入 GEPA 的反思過程。

    Args:
        save_program: 是否自動存檔優化後的 program（預設 True）。
                      存到 programs/gepa_{timestamp}/program.json。

    Returns:
        optimized: 優化後的 GEPATrainableHarness（dspy.Module）
    """

    # 配置 — 全部用 gpt-4o-mini 降低成本，可替換為 gpt-4o
    task_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)
    reflection_lm = dspy.LM("openai/gpt-4o-mini", temperature=1.0, max_tokens=8000)
    dspy.configure(lm=task_lm)

    # 創建可訓練的 Harness（用 wrapper 橋接 example fields）
    trainable = GEPATrainableHarness(
        fast_lm=task_lm,
        strong_lm=task_lm
    )

    # ★ v0.9: 記錄優化前的 instructions（用於 diff 分析）
    seed_instructions = {}
    for name, pred in trainable.named_predictors():
        sig = pred.signature
        seed_instructions[name] = (
            str(sig.instructions) if hasattr(sig, 'instructions') else str(sig)
        )

    # 訓練集 — 包含 dissatisfied example 以觸發 L7 predictors
    examples = create_training_examples()
    # examples: [market, home, tv, park, market_dissatisfied]
    # trainset 要包含 dissatisfied 讓 parse_intent/update_prefs 有 trace
    train_set = [examples[0], examples[2], examples[4]]  # market, tv, dissatisfied
    val_set = [examples[1], examples[3]]                  # home, park

    # ★ v0.9: 時間戳 + 日誌目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", "gepa", timestamp)

    print("\n" + "=" * 70)
    print("★★★ GEPA 編譯開始 ★★★")
    print(f"  訓練集: {len(train_set)} examples")
    for ex in train_set:
        print(f"    - {ex.scenario} (user_action={ex.user_action})")
    print(f"  驗證集: {len(val_set)} examples")
    for ex in val_set:
        print(f"    - {ex.scenario}")
    print(f"  Task LM: openai/gpt-4o-mini")
    print(f"  Reflection LM: openai/gpt-4o-mini")
    print(f"  Log dir: {log_dir}")
    print("  Predictors to optimize:")
    for name, _ in trainable.named_predictors():
        print(f"    - {name}")
    print("=" * 70)

    # ★ Phase 4: 嘗試載入 MultiModalInstructionProposer
    #   若有多模態 Signature（dspy.Audio/dspy.Image），GEPA 的反思迴圈
    #   可以同時看到音訊/圖片 + 錯誤 feedback，更精準地改進 prompt。
    instruction_proposer = None
    try:
        from dspy.teleprompt.gepa.instruction_proposal import (
            MultiModalInstructionProposer,
        )
        instruction_proposer = MultiModalInstructionProposer()
        print("  ★ Phase 4: MultiModalInstructionProposer 已啟用")
    except ImportError:
        print("  ★ Phase 4: MultiModalInstructionProposer 不可用，使用預設 proposer")

    import mlflow
    print(f"  ★ MLflow version {mlflow.__version__}")

    # ★ GEPA 編譯 — 限制預算在 ~5-10 分鐘
    gepa_kwargs = dict(
        metric=create_acoustic_feedback_metric,
        max_metric_calls=100,
        num_threads=2,
        track_stats=True,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        warn_on_score_mismatch=False,
        # ★ v0.9: 追蹤配置
        log_dir=log_dir,
        track_best_outputs=True,
        use_mlflow=True,
    )
    if instruction_proposer is not None:
        gepa_kwargs["instruction_proposer"] = instruction_proposer

    optimizer = dspy.GEPA(**gepa_kwargs)

    optimized = optimizer.compile(
        trainable,
        trainset=train_set,
        valset=val_set,
    )

    print("\n" + "=" * 70)
    print("★★★ GEPA 編譯完成 ★★★")
    print("=" * 70)

    # 印出 instruction diffs（MLflow 也有記錄，這裡只做 console 摘要）
    changed = 0
    for name, pred in optimized.named_predictors():
        sig = pred.signature
        after = str(sig.instructions) if hasattr(sig, 'instructions') else str(sig)
        before = seed_instructions.get(name, "")
        if before.strip() != after.strip():
            changed += 1
            print(f"\n  ★ {name}")
            print(f"    BEFORE: {before[:150]}{'...' if len(before) > 150 else ''}")
            print(f"    AFTER:  {after[:150]}{'...' if len(after) > 150 else ''}")
    print(f"\n  Instruction changes: {changed}/{len(seed_instructions)} predictors")

    # ★ v0.9: 存檔優化後的 program + metadata
    save_dir = None
    if save_program:
        save_dir = os.path.join("programs", f"gepa_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        program_path = os.path.join(save_dir, "program.json")
        optimized.save(program_path)
        print(f"\n  Program saved to: {program_path}")

        # Metadata
        metadata = {
            "timestamp": timestamp,
            "gepa_config": {
                "max_metric_calls": 100,
                "num_threads": 2,
                "task_lm": "openai/gpt-4o-mini",
                "reflection_lm": "openai/gpt-4o-mini",
                "train_scenarios": [ex.scenario for ex in train_set],
                "val_scenarios": [ex.scenario for ex in val_set],
            },
            "predictor_count": len(list(optimized.named_predictors())),
        }
        if hasattr(optimized, 'detailed_results'):
            dr = optimized.detailed_results
            metadata["total_metric_calls"] = getattr(dr, 'total_metric_calls', None)
            metadata["num_candidates"] = getattr(dr, 'num_candidates', None)
            metadata["num_full_val_evals"] = getattr(dr, 'num_full_val_evals', None)

        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"  Metadata saved to: {metadata_path}")

    # 用優化後的模型跑一次驗證
    print("\n" + "-" * 70)
    print(">>> 用優化後的管線執行驗證場景")
    dspy.configure(lm=task_lm)
    try:
        val_example = val_set[0]
        val_result = optimized(
            scenario=val_example.scenario,
            snr_db=val_example.snr_db,
            rt60_s=val_example.rt60_s,
            n_active_sources=val_example.n_active_sources,
            energy_db=val_example.energy_db,
            temporal_pattern=val_example.temporal_pattern,
            user_profile=val_example.user_profile,
            user_action=val_example.user_action,
        )
        print(f"\n  場景: {val_example.scenario}")
        print(f"  [Layer 4] Percept confidence: {val_result.percept.confidence}")
        print(f"  [Layer 5] Scene: {val_result.scene.situation[:200]}")
        print(f"  [Layer 6] Strategy: beam={val_result.strategy.target_azimuth_deg}°, "
              f"NR={val_result.strategy.nr_aggressiveness}")
        print(f"  [Layer 2] DSP params generated: {val_result.dsp_params is not None}")
    except Exception as e:
        print(f"  驗證執行時發生錯誤: {e}")

    return optimized
