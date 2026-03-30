# AGENTS.md — Cross-Tool Agent Instructions

**Do not over-engineer.** If one line solves the problem, do not write three.
Avoid abstractions, tools, or frameworks that only exist for hypothetical future
needs. Start with the simplest solution and add complexity only when the simple
version is proven insufficient. Before adding a feature, check whether DSPy or
MLflow already provides it so you do not rebuild something that already exists.

This file provides context for AI coding agents such as Claude Code or Codex.

## Repository Purpose

ASIR (Acoustic Semantic Instruction Register) is a 7-layer semantic IR for
hearing aids. It uses DSPy modules plus GEPA optimization to convert raw audio
into optimized DSP parameters through LLM-powered reasoning.

**Use case:** hearing devices need to adapt DSP dynamically in complex
soundscapes such as wet markets or multi-speaker conversations. Fixed-rule DSP
does not understand semantics; ASIR uses LLM reasoning to interpret the scene
before choosing DSP parameters.

**I/O** (`harness.forward()`):
- Input: `RawSignal` (2-ch PCM) + `user_action` + `audiogram_json` + `user_profile`
- Output: `DSPParameterSet` (`beam_weights`, `noise_mask`, `filter_coeffs`,
  `compression_ratio`) + `execution_depth` + semantic intermediate results

## File Organization

- Every `.py` file is under 500 lines.
- One concept per file.
- Imports are explicit.
- Types, primitives, routing, and composites stay clearly separated.

## Critical Invariants

1. L1-L3 are **deterministic** (`numpy` only, no LLM calls).
2. L4-L7 use **`dspy.ChainOfThought`** with typed Signatures.
3. Routing predictors (Method A) are the primary GEPA optimization targets.
4. `harness.py` is the only file that orchestrates cross-layer execution.
5. `gepa/metric.py` encodes all physics and audiology constraints as text feedback.

## Running the Project

```bash
# Full demo (L1-L7 semantic reasoning + user feedback, requires OPENAI_API_KEY)
PYTHONUTF8=1 python -X utf8 -m examples.run_demo

# Deterministic layers only (no API key required)
PYTHONUTF8=1 python -X utf8 -m examples.run_demo --l1-l3
```

## Testing and Evaluation

**Run in this order:**

```bash
# Step 0: generate 10 scenario test WAV files (run once)
PYTHONUTF8=1 python -X utf8 -m asir.eval.generate_audio

# Step 1: L1-L3 + scenario consistency pytest (no API key, 60 tests, ~10s)
#   Verifies deterministic pipeline + 10 scenario WAV loads +
#   1:1 alignment between eval examples and WAV files
#   No LLM reasoning, so no reasoning trace is produced
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_deterministic.py -v

# Step 2: L4-L7 semantic reasoning pytest (needs OPENAI_API_KEY, 52 tests, ~2 min)
#   Verifies 10 scenarios × 5 layers + the wet-market E9/E10 pair
#   Reasoning traces are logged to MLflow (experiment: asir-eval-pytest)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_semantic.py -v

# Step 3: end-to-end integration pytest (needs OPENAI_API_KEY, 50 tests, ~10 min)
#   Verifies real audio -> full harness -> per-layer metrics checks
#   Reasoning traces are logged to MLflow (experiment: asir-integration-pytest)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_integration.py -v

# Or use standalone eval runners for the same logic with richer console traces
PYTHONUTF8=1 python -X utf8 -m asir.eval
PYTHONUTF8=1 python -X utf8 -m asir.eval --integration
```

**All LLM reasoning outputs are recorded to MLflow.** Inspect history and
compare runs with `mlflow ui`.

### Where to Inspect Reasoning Traces

| What you ran | Where the trace lives | How to inspect it |
|---|---|---|
| `test_deterministic.py` | none (L1-L3 is deterministic) | console output |
| `test_semantic.py` | MLflow `asir-eval-pytest` → `pytest_eval_results.json` | `mlflow ui` |
| `test_integration.py` | MLflow `asir-integration-pytest` → `pytest_integration_results.json` | `mlflow ui` |
| `run_demo` | MLflow `asir-demo` → `demo_trace.json` | `mlflow ui` |
| `asir.eval` | MLflow `asir-eval` → `eval_results.json` | `mlflow ui` or `download_artifacts("eval_results.json")` |
| `asir.eval --integration` | MLflow `asir-eval` → `integration_results.json` | `mlflow ui` |

After running eval, read the structured trace in the MLflow artifact to judge
whether each layer's reasoning is sensible. Do not look at scores only.

**Agent interpretation policy**
- After eval finishes, use `mlflow.search_runs()` plus
  `download_artifacts("eval_results.json")` to read structured traces.
- Check each scenario layer by layer: whether L4 noise descriptions reflect SNR,
  whether L5 scene interpretations are plausible, whether L6 NR matches the
  noise level, and whether L7 updates preferences correctly.
- Separate **LLM reasoning drift** (a GEPA optimization target, not necessarily a
  code bug) from **program logic bugs** (which require code changes).
- Do not just report "2 failures." Explain why they failed and whether action is
  needed.

### Trace: Per-Scenario Reasoning Chain

Every scenario prints an L4 → L5 → L6 → DSP → L7 reasoning chain automatically.
The same trace is also stored in MLflow artifacts.

Even if every check passes, outputs may still be unreasonable
(for example, `NR=0.7` in a quiet scene). Use the trace to confirm that the
reasoning matches the acoustic conditions.

### GEPA Optimization

```bash
PYTHONUTF8=1 OPENAI_API_KEY=sk-xxx python -X utf8 -m examples.run_demo --gepa
```

GEPA automatically saves optimized programs to `programs/gepa_{timestamp}/`,
and MLflow records the full optimization process.

### Loading an Optimized Program

```bash
PYTHONUTF8=1 python -X utf8 -m asir.eval --program programs/gepa_xxx/program.json
PYTHONUTF8=1 python -X utf8 -m asir.eval --integration --program programs/gepa_xxx/program.json
```

### Eval Scenarios and README Alignment

The 10 evaluation scenarios are defined in `asir/eval/examples.py`. Two of them
map directly to the flagship README examples:

| Scenario | README section | Validation focus |
|---|---|---|
| `wet_market_vendor` | "Example Scenario: Wet Market Conversation" | extreme noise at SNR=0 dB → strong NR + forward beam focus |
| `market_too_muffled` | "User feedback: too muffled" | `user_action="too muffled"` → reduce NR + persist preferences |

If you change the README scenario description, update the eval scenario too, and
vice versa.

### Interpreting Eval Results

Eval scores vary by about ±5% because of LLM nondeterminism.

**Known GEPA optimization targets (not bugs)**
- L6 `nr_matches_scene` — NR aggressiveness sometimes falls below the `0.4`
  threshold.
- DSP `beam_appropriate` — the LLM sometimes chooses a narrow beam in quiet
  scenes where a wider beam would be more appropriate.

**A real bug signature:** the same check fails every run for a logic reason.

### Key Behavior: `user_action` and Preferences

`harness.py` calls `UpdatePreferencesSig` for any `user_action != "none"`.
The LLM decides semantically whether preferences should change; there is no
hard-coded keyword rule.

## Test Audio Files

`asir/eval/audio/` keeps evaluation code and data in one place:

```text
asir/eval/audio/
├── audio1.wav, audio2.wav     # Gemini TTS samples (24kHz mono)
├── speech/                    # clean speech clips (16kHz mono)
├── noise/                     # optional DEMAND dataset noise clips
└── scenarios/                 # mixed scenario audio (stereo 16kHz, 5s each)
```

Generator: `asir/eval/generate_audio.py`

WAV files are tracked with Git LFS.

## Common Tasks

- **Add a new Signature**: create it in `asir/primitives/` and export it in `__init__.py`
- **Add routing logic**: create it in `asir/routing/` and wire it into the composite
- **Modify GEPA feedback**: edit `asir/gepa/metric.py`
- **Change pipeline flow**: edit `asir/harness.py`
- **Add a new eval scenario**: update `examples.py` and `generate_audio.py`, then rerun generation and eval
- **Debug an eval failure**: inspect trace output or MLflow artifacts with `mlflow ui`

## Artifact Directories

```text
mlruns/               # MLflow tracking (eval + GEPA results)
runs/                 # GEPA optimization logs
programs/             # saved optimized programs
```
