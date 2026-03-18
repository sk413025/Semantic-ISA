# ASIR — Acoustic Semantic IR for Hearing Aids

7-layer semantic instruction set architecture using DSPy + GEPA.
Converts raw microphone signals into optimized hearing aid DSP parameters
via an LLM-powered pipeline with learnable routing decisions.

## Quick Start

```bash
# Deterministic layers only (no API key)
PYTHONUTF8=1 python -X utf8 -m examples.run_demo

# Full pipeline (needs OPENAI_API_KEY)
PYTHONUTF8=1 OPENAI_API_KEY=sk-xxx python -X utf8 -m examples.run_demo --full

# Full + GEPA optimization
PYTHONUTF8=1 OPENAI_API_KEY=sk-xxx python -X utf8 -m examples.run_demo --gepa
```

**Windows note**: Always use `PYTHONUTF8=1 python -X utf8` to avoid cp1252 encoding errors.

## Architecture Overview

```
Input:  RawSignal (2-ch mic array, 16kHz, 32ms frames)
Output: DSPParameterSet (beam weights, NR mask, gain filter, compression)

L1 Physical Sensing    → prim_sample_audio()           [deterministic]
L2 Signal Processing   → FFT, beamform, spectral sub   [deterministic]
L3 Acoustic Features   → MFCC, SNR, RT60 extraction    [deterministic]
── SEMANTIC BOUNDARY ──
L4 Perceptual Desc     → 3 LLM prims + routing         [fast_lm + multimodal]
L5 Scene Understanding → LLM + history + routing        [strong_lm + spectrogram]
── SEMANTIC BOUNDARY ──
L6 Strategy Generation → planner → beam+NR+gain → integrator  [strong_lm]
L7 Intent & Preference → parse user actions, update prefs      [strong_lm]

Pipeline Router decides per-frame depth: fast / medium / full
```

## Package Structure

```
asir/
├── types/          # Pydantic data types for each layer (L1-L7)
├── primitives/     # Atomic operations: deterministic DSP + LLM Signatures
├── routing/        # GEPA-optimizable routing Signatures (Method A)
├── composites/     # dspy.Module orchestrators combining primitives+routing
├── multimodal/     # dspy.Audio / dspy.Image generation (Phase 1-2)
├── gepa/           # GEPA metric, training examples, compiler
├── harness.py      # Top-level AcousticSemanticHarness (the "OS")
└── architecture.py # ASCII architecture diagram
examples/
├── run_demo.py     # Entry point: deterministic demo / full pipeline / GEPA
└── audio/          # Test WAV files for agent testing (audio1.wav, audio2.wav)
docs/               # Research papers (PDF) — theoretical background
```

## Key Concepts

- **PRIM**: Atomic operation (deterministic function or LLM Signature)
- **ROUTING**: Learnable decision predictor (Method A) — GEPA-optimizable
- **COMP**: dspy.Module that orchestrates PRIMs + ROUTINGs
- **GEPA**: Genetic Evolution Programming Architecture — optimizes LLM prompts via reflective mutation on Pareto frontiers

## Dependencies

- `dspy>=2.6` (with GEPA support)
- `numpy`, `matplotlib`, `scipy`
- LLM API key (OpenAI recommended: `gpt-4o-mini`)

## Adding a New Layer or Primitive

1. Define types in `asir/types/`
2. Define Signature in `asir/primitives/`
3. If routing needed, add to `asir/routing/`
4. Wire into composite in `asir/composites/`
5. Add GEPA feedback rules in `asir/gepa/metric.py`
6. Update `asir/harness.py` forward() to call the new component
