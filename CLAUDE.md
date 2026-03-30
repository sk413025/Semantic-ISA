# ASIR — Acoustic Semantic IR for Hearing Aids

**Do not over-engineer.** If one line solves the problem, do not write three.
Avoid abstractions, tools, or frameworks that exist only for hypothetical future
needs. Start with the simplest workable solution and add complexity only when it
is justified. Before adding functionality, search the web and inspect DSPy and
MLflow internals to confirm whether the capability already exists.

This repository implements a seven-layer semantic instruction architecture using
DSPy plus GEPA. It converts raw microphone signals into optimized hearing-aid
DSP parameters through an LLM-powered pipeline with learnable routing.

## Use Case and I/O

**Use case:** hearing aids need to adapt DSP parameters dynamically in complex
acoustic scenes such as wet markets or multi-speaker conversations. Fixed-rule
processing cannot infer what the user actually wants to hear, so ASIR adds
semantic reasoning before selecting DSP behavior.

**Input** (`harness.forward()`):
- `raw_signal: RawSignal` — 2-ch PCM, 16 kHz, 32 ms/frame
- `user_action: str` — user action such as `"too_noisy"`, `"focus_front"`, or `"none"`
- `audiogram_json: str` — audiogram, for example `{"250":30,"500":35,...}`
- `user_profile: str` — user description

**Output** (`dspy.Prediction`):
- `dsp_params: DSPParameterSet` — beam weights, noise mask, filter coefficients, compression
- `execution_depth: str` — `"fast"`, `"medium"`, or `"full"`
- `scene_description`, `strategy_summary` — semantic intermediate outputs

## Quick Start

```bash
# Full L1-L7 demo — wet market -> semantic reasoning -> "too muffled" feedback -> preference update
PYTHONUTF8=1 OPENAI_API_KEY=sk-xxx python -X utf8 -m examples.run_demo

# Deterministic only (no API key)
PYTHONUTF8=1 python -X utf8 -m examples.run_demo --l1-l3

# Full pipeline + GEPA optimization
PYTHONUTF8=1 OPENAI_API_KEY=sk-xxx python -X utf8 -m examples.run_demo --gepa
```

**Windows note:** always use `PYTHONUTF8=1 python -X utf8` to avoid cp1252
encoding errors.

## Architecture Overview

```text
Input:  RawSignal (2-ch mic array, 16kHz, 32ms frames)
Output: DSPParameterSet (beam weights, NR mask, gain filter, compression)

L1 Physical Sensing    -> prim_sample_audio()                    [deterministic]
L2 Signal Processing   -> FFT, beamform, spectral subtraction    [deterministic]
L3 Acoustic Features   -> MFCC, SNR, RT60 extraction             [deterministic]
── SEMANTIC BOUNDARY ──
L4 Perceptual Desc     -> 3 LLM primitives + routing             [fast_lm + multimodal]
L5 Scene Understanding -> LLM + history + routing                [strong_lm + spectrogram]
── SEMANTIC BOUNDARY ──
L6 Strategy Generation -> planner -> beam + NR + gain -> integrator  [strong_lm]
L7 Intent & Preference -> parse user actions, update prefs            [strong_lm]

PipelineRouter chooses per-frame depth: fast / medium / full
```

## Package Structure

```text
asir/
├── types/             # Pydantic data types for each layer (L1-L7)
├── primitives/        # Atomic ops: deterministic DSP + LLM Signatures
├── routing/           # GEPA-optimizable routing Signatures (Method A)
├── composites/        # dspy.Module orchestrators combining primitives + routing
├── multimodal/        # dspy.Audio / dspy.Image generation
├── gepa/              # GEPA metric, training examples, compiler
├── eval/              # Evaluation framework: examples, metrics, runners
│   ├── audio/         # Test WAV files (Git LFS), scenarios/ + speech/
│   ├── examples.py    # 10 eval scenarios (physical parameters + constraints)
│   ├── run.py         # Semantic eval runner (L4-L7)
│   ├── integration.py # Integration eval (real audio -> full pipeline)
│   └── generate_audio.py
├── harness.py         # Top-level AcousticSemanticHarness
└── architecture.py    # ASCII architecture diagram
tests/
├── test_deterministic.py  # L1-L3 + eval scenario consistency
├── test_semantic.py       # L4-L7 semantic reasoning quality
└── test_integration.py    # End-to-end real-audio harness tests
examples/
└── run_demo.py
docs/                       # Development PDFs
```

## Key Concepts

- **PRIM**: atomic operation (deterministic function or LLM Signature)
- **ROUTING**: learnable decision predictor (Method A), optimized by GEPA
- **COMP**: `dspy.Module` that orchestrates PRIMs and ROUTINGs
- **GEPA**: Pareto-frontier prompt optimization using reflective mutation

## Domain Glossary

Audiology:
- **Audiogram** = hearing threshold per frequency in dB HL
- **NAL-NL2** = prescription formula that maps audiogram values to gain
- **SNHL** = sensorineural hearing loss

Signal processing:
- **Beamforming** = spatial filtering across microphones (`beam_weights`)
- **Spectral Subtraction** = frequency-domain noise reduction
- **Noise Mask** = per-frequency 0-1 gain mask
- **SNR** = signal-to-noise ratio in dB
- **RT60** = seconds needed for reverberation to decay by 60 dB
- **MFCC** = speech/audio features inspired by human hearing
- **Compression** = dynamic range compression (`compression_ratio`)

## Evaluation

```bash
# L1-L3 + scenario consistency pytest (no API key, 60 tests)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_deterministic.py -v

# L4-L7 semantic reasoning pytest (needs OPENAI_API_KEY, 52 tests)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_semantic.py -v

# End-to-end integration pytest (needs OPENAI_API_KEY, 50 tests)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_integration.py -v

# Standalone eval runners
PYTHONUTF8=1 python -X utf8 -m asir.eval
PYTHONUTF8=1 python -X utf8 -m asir.eval --integration
PYTHONUTF8=1 python -X utf8 -m asir.eval --program programs/gepa_xxx/program.json
```

All eval and GEPA outputs are logged to MLflow. Inspect them with `mlflow ui`.

Each scenario automatically prints an L4 → L7 reasoning trace and stores the
same trace in MLflow artifacts.

L4-L7 evaluation data flow:
- `examples.py` defines physical scenario parameters
- `build_features()` converts them into `AcousticFeatures` for L4
- composites call L4 -> L5 -> L6 directly, bypassing harness L1-L3
- `metrics.py` checks physical and semantic constraints rather than exact keywords
- the constraint-field mapping lives at the top of `metrics.py`

## Dependencies

- `dspy>=2.6` (with GEPA support)
- `numpy`, `matplotlib`, `scipy`
- `mlflow>=2.0`
- LLM API key (recommended: OpenAI with `gpt-4o-mini`)

## Adding a New Layer or Primitive

1. Define types in `asir/types/`
2. Define the Signature in `asir/primitives/`
3. Add routing logic in `asir/routing/` if needed
4. Wire it into a composite in `asir/composites/`
5. Add GEPA feedback rules in `asir/gepa/metric.py`
6. Update `asir/harness.py` so `forward()` calls the new component
