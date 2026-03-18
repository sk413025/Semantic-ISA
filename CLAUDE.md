# ASIR — Acoustic Semantic IR for Hearing Aids

**不要過度工程化。** 能用一行解決就不要寫三行。不要加「以後可能用到」的抽象、工具、框架。先做最簡單的方案，只在證明不夠時才加複雜度。加功能前先 web search 並深入 DSPy 和 MLflow 底層確認是否已內建支援——不要重複造輪子。

7-layer semantic instruction set architecture using DSPy + GEPA.
Converts raw microphone signals into optimized hearing aid DSP parameters
via an LLM-powered pipeline with learnable routing decisions.

## Use Case & I/O

**情境**：助聽器在複雜聲學場景（菜市場、多人對話等）需要動態調整 DSP 參數。
傳統固定規則無法理解「使用者想聽什麼」，ASIR 用 LLM 做語意推理來決定。

**Input** (`harness.forward()`):
- `raw_signal: RawSignal` — 2-ch PCM, 16kHz, 32ms/frame
- `user_action: str` — 使用者動作 (`"太吵了"`, `"focus_front"`, `"none"`)
- `audiogram_json: str` — 聽力圖 (`{"250":30, "500":35, ...}`)
- `user_profile: str` — 使用者描述

**Output** (`dspy.Prediction`):
- `dsp_params: DSPParameterSet` — beam_weights, noise_mask, filter_coeffs, compression
- `execution_depth: str` — `"fast"` / `"medium"` / `"full"`
- `scene_description`, `strategy_summary` — 語意中間結果

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
├── eval/           # Evaluation framework: examples, metrics, runner
│   ├── audio/      # Test WAV files (Git LFS), scenarios/ + speech/
│   ├── examples.py # 10 eval scenarios (物理參數 + 約束)
│   ├── run.py      # Semantic eval runner (L4-L7)
│   ├── integration.py  # Integration eval (real audio → full pipeline)
│   └── generate_audio.py  # Scenario WAV generator
├── harness.py      # Top-level AcousticSemanticHarness (the "OS")
└── architecture.py # ASCII architecture diagram
tests/
├── test_deterministic.py  # L1-L3 + eval 場景一致性 (60 tests, no API key)
├── test_semantic.py       # L4-L7 語意推理品質 (52 tests, needs API key)
└── test_integration.py    # 端對端: 真實音檔 → 完整 harness (50 tests, needs API key)
examples/
└── run_demo.py     # Entry point: uses asir/eval/audio/ scenario WAVs
docs/               # Development documents (PDF)
```

## Key Concepts

- **PRIM**: Atomic operation (deterministic function or LLM Signature)
- **ROUTING**: Learnable decision predictor (Method A) — GEPA-optimizable
- **COMP**: dspy.Module that orchestrates PRIMs + ROUTINGs
- **GEPA**: Genetic Evolution Programming Architecture — optimizes LLM prompts via reflective mutation on Pareto frontiers

## Domain Glossary (quick reference)

Audiology: **Audiogram** = 各頻率聽力閾值圖 (dB HL)；**NAL-NL2** = 根據 audiogram 算增益的處方公式；**SNHL** = 感音神經性聽損（老年聽損主因）

Signal Processing: **Beamforming** = 多麥克風空間濾波（`beam_weights`）；**Spectral Subtraction** = 頻譜減噪；**Noise Mask** = 逐頻率 0-1 增益遮罩；**SNR** = 訊噪比 (dB)；**RT60** = 迴響衰減 60dB 所需秒數；**MFCC** = 模仿人耳的頻率特徵；**Compression** = 動態範圍壓縮（`compression_ratio`）

## Evaluation

```bash
# L1-L3 + eval 場景一致性 pytest (no API key, 60 tests)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_deterministic.py -v

# L4-L7 語意推理 pytest (needs OPENAI_API_KEY, 52 tests)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_semantic.py -v

# Integration 端對端 pytest (needs OPENAI_API_KEY, 50 tests)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_integration.py -v

# Standalone eval runners (same logic, more detailed console output)
PYTHONUTF8=1 python -X utf8 -m asir.eval
PYTHONUTF8=1 python -X utf8 -m asir.eval --integration
PYTHONUTF8=1 python -X utf8 -m asir.eval --program programs/gepa_xxx/program.json
```

**所有 eval/GEPA 結果記錄到 MLflow。** 查看歷史與比較：`mlflow ui`

每個場景自動印出 L4→L7 推理鏈 trace + 存到 MLflow artifacts。

L4-L7 eval 資料流：
- `examples.py` 定義場景物理參數 → `build_features()` 轉成 `AcousticFeatures` 注入 L4
- composites (L4→L5→L6) 直接呼叫，繞過 harness 的 L1-L3
- `metrics.py` 檢查物理約束（不是 keyword exact match）
- Constraint 欄位對應見 `metrics.py` 頂部 mapping 表

## Dependencies

- `dspy>=2.6` (with GEPA support)
- `numpy`, `matplotlib`, `scipy`
- `mlflow>=2.0`
- LLM API key (OpenAI recommended: `gpt-4o-mini`)

## Adding a New Layer or Primitive

1. Define types in `asir/types/`
2. Define Signature in `asir/primitives/`
3. If routing needed, add to `asir/routing/`
4. Wire into composite in `asir/composites/`
5. Add GEPA feedback rules in `asir/gepa/metric.py`
6. Update `asir/harness.py` forward() to call the new component
