# AGENTS.md — Cross-Tool Agent Instructions

This file provides context for AI coding agents (Claude Code, Codex, etc.).

## Repository Purpose

ASIR (Acoustic Semantic Instruction Register) is a 7-layer semantic IR
for hearing aids. It uses DSPy modules + GEPA optimizer to convert raw
audio into optimized DSP parameters through LLM-powered reasoning.

## File Organization

- Every `.py` file is < 500 lines
- One concept per file
- All imports are explicit (no wildcard imports)
- Types, primitives, routing, composites are cleanly separated

## Critical Invariants

1. L1-L3 are **deterministic** (numpy only, no LLM calls)
2. L4-L7 use **dspy.ChainOfThought** with typed Signatures
3. Routing predictors (Method A) are the primary GEPA optimization targets
4. `harness.py` is the only file that orchestrates cross-layer execution
5. `gepa/metric.py` encodes all physics/audiology constraints as text feedback

## Running the Project

```bash
# Always use UTF-8 on Windows
PYTHONUTF8=1 python -X utf8 -m examples.run_demo
```

## Testing Changes

After modifying any file, verify the deterministic layers still work:
```bash
PYTHONUTF8=1 python -X utf8 -c "from asir.primitives import comp_extract_full_features, prim_sample_audio; f = comp_extract_full_features(prim_sample_audio()); print(f'SNR={f.snr_db}, RT60={f.rt60_s}')"
```

## Common Tasks

- **Add new Signature**: Create in `asir/primitives/`, export in `__init__.py`
- **Add routing logic**: Create in `asir/routing/`, wire in composite
- **Modify GEPA feedback**: Edit `asir/gepa/metric.py`
- **Change pipeline flow**: Edit `asir/harness.py`
