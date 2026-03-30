import json
import os
from datetime import datetime

import dspy

from asir.gepa.metric import create_acoustic_feedback_metric
from asir.gepa.training import GEPATrainableHarness, create_training_examples


def compile_with_gepa(save_program=True):
    """
    Compile the end-to-end ASIR pipeline with GEPA.

    GEPA inspects failures, reflects on their causes, proposes better
    instructions, and keeps the best candidates.
    """

    task_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)
    reflection_lm = dspy.LM("openai/gpt-4o-mini", temperature=1.0, max_tokens=8000)
    dspy.configure(lm=task_lm)

    trainable = GEPATrainableHarness(fast_lm=task_lm, strong_lm=task_lm)

    seed_instructions = {}
    for name, pred in trainable.named_predictors():
        sig = pred.signature
        seed_instructions[name] = (
            str(sig.instructions) if hasattr(sig, "instructions") else str(sig)
        )

    examples = create_training_examples()
    train_set = [examples[0], examples[2], examples[4]]
    val_set = [examples[1], examples[3]]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", "gepa", timestamp)

    print("\n" + "=" * 70)
    print("*** GEPA compilation started ***")
    print(f"  Train set: {len(train_set)} examples")
    for ex in train_set:
        print(f"    - {ex.scenario} (user_action={ex.user_action})")
    print(f"  Validation set: {len(val_set)} examples")
    for ex in val_set:
        print(f"    - {ex.scenario}")
    print("  Task LM: openai/gpt-4o-mini")
    print("  Reflection LM: openai/gpt-4o-mini")
    print(f"  Log dir: {log_dir}")
    print("  Predictors to optimize:")
    for name, _ in trainable.named_predictors():
        print(f"    - {name}")
    print("=" * 70)

    instruction_proposer = None
    try:
        from dspy.teleprompt.gepa.instruction_proposal import (
            MultiModalInstructionProposer,
        )

        instruction_proposer = MultiModalInstructionProposer()
        print("  MultiModalInstructionProposer enabled")
    except ImportError:
        print("  MultiModalInstructionProposer unavailable, using default proposer")

    import mlflow

    print(f"  MLflow version {mlflow.__version__}")

    gepa_kwargs = dict(
        metric=create_acoustic_feedback_metric,
        max_metric_calls=100,
        num_threads=2,
        track_stats=True,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        warn_on_score_mismatch=False,
        log_dir=log_dir,
        track_best_outputs=True,
        use_mlflow=True,
    )
    if instruction_proposer is not None:
        gepa_kwargs["instruction_proposer"] = instruction_proposer

    optimizer = dspy.GEPA(**gepa_kwargs)
    optimized = optimizer.compile(trainable, trainset=train_set, valset=val_set)

    print("\n" + "=" * 70)
    print("*** GEPA compilation finished ***")
    print("=" * 70)

    changed = 0
    for name, pred in optimized.named_predictors():
        sig = pred.signature
        after = str(sig.instructions) if hasattr(sig, "instructions") else str(sig)
        before = seed_instructions.get(name, "")
        if before.strip() != after.strip():
            changed += 1
            print(f"\n  * {name}")
            print(f"    BEFORE: {before[:150]}{'...' if len(before) > 150 else ''}")
            print(f"    AFTER:  {after[:150]}{'...' if len(after) > 150 else ''}")
    print(f"\n  Instruction changes: {changed}/{len(seed_instructions)} predictors")

    if save_program:
        save_dir = os.path.join("programs", f"gepa_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        program_path = os.path.join(save_dir, "program.json")
        optimized.save(program_path)
        print(f"\n  Program saved to: {program_path}")

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
        if hasattr(optimized, "detailed_results"):
            dr = optimized.detailed_results
            metadata["total_metric_calls"] = getattr(dr, "total_metric_calls", None)
            metadata["num_candidates"] = getattr(dr, "num_candidates", None)
            metadata["num_full_val_evals"] = getattr(dr, "num_full_val_evals", None)

        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"  Metadata saved to: {metadata_path}")

    print("\n" + "-" * 70)
    print(">>> Running one validation scenario with the optimized pipeline")
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
        print(f"\n  Scenario: {val_example.scenario}")
        print(f"  [Layer 4] Percept confidence: {val_result.percept.confidence}")
        print(f"  [Layer 5] Scene: {val_result.scene.situation[:200]}")
        print(
            f"  [Layer 6] Strategy: beam={val_result.strategy.target_azimuth_deg}°, "
            f"NR={val_result.strategy.nr_aggressiveness}"
        )
        print(f"  [Layer 2] DSP params generated: {val_result.dsp_params is not None}")
    except Exception as e:
        print(f"  Validation run raised an error: {e}")

    return optimized
