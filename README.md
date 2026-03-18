# ASIR — Acoustic Semantic Instruction Register

> **⚠️ Research Prototype** — 這是研究用原型，不是 production-ready 套件。
> API 隨時可能改動，breaking changes 不另行通知。

## 這個 repo 是給 coding agent 看的

本專案的文件結構專門為 AI coding agents（Claude Code, Codex 等）最佳化：

| 檔案 | 用途 |
|------|------|
| `CLAUDE.md` | Agent 進入點 — 快速理解架構、跑法、package 結構 |
| `AGENTS.md` | 跨工具 agent 指令 — invariants、測試方式、常見任務 |
| `pyproject.toml` | 依賴與 metadata |

**設計原則**：每個 `.py` < 500 行、一個檔案一個概念、explicit imports only。

## What is ASIR?

7-layer semantic instruction set architecture for hearing aids.
用 DSPy modules + GEPA optimizer，把麥克風原始訊號轉換成最佳化的助聽器 DSP 參數。

```
Raw Audio (2-ch, 16kHz) → [L1-L3 deterministic DSP] → [L4-L7 LLM reasoning] → DSP Parameters
```

### 架構總覽

```
L1 Physical Sensing    → prim_sample_audio()           [deterministic, numpy]
L2 Signal Processing   → FFT, beamform, spectral sub   [deterministic, numpy]
L3 Acoustic Features   → MFCC, SNR, RT60 extraction    [deterministic, numpy]
── SEMANTIC BOUNDARY ──
L4 Perceptual Desc     → 3 LLM prims + routing         [fast_lm + multimodal]
L5 Scene Understanding → LLM + history + routing        [strong_lm + spectrogram]
── SEMANTIC BOUNDARY ──
L6 Strategy Generation → planner → beam+NR+gain → integrator  [strong_lm]
L7 Intent & Preference → parse user actions, update prefs      [strong_lm]

Pipeline Router 決定每幀跑多深：fast / medium / full
```

完整的 ASCII 架構圖見 `asir/architecture.py`。

## Project Structure

```
asir/
├── types/          # Pydantic data types (L1-L7)
├── primitives/     # 原子操作：deterministic DSP + LLM Signatures
├── routing/        # GEPA-optimizable routing predictors (Method A)
├── composites/     # dspy.Module 組合器
├── multimodal/     # dspy.Audio / dspy.Image 生成
├── gepa/           # GEPA metric, training examples, compiler
├── harness.py      # Top-level orchestrator (the "OS")
└── architecture.py # ASCII 架構圖
examples/
├── run_demo.py     # Entry point
└── audio/          # 測試用 WAV 檔 (audio1.wav, audio2.wav)
docs/               # 研究論文 PDF（理論背景）
tests/              # (placeholder)
```

## Quick Start

```bash
# L1-L3 deterministic demo（不需要 API key）
PYTHONUTF8=1 python -X utf8 -m examples.run_demo

# Full L1-L7 pipeline（需要 OPENAI_API_KEY）
PYTHONUTF8=1 OPENAI_API_KEY=sk-xxx python -X utf8 -m examples.run_demo --full

# Full + GEPA optimization
PYTHONUTF8=1 OPENAI_API_KEY=sk-xxx python -X utf8 -m examples.run_demo --gepa
```

> **Windows 必須**加 `PYTHONUTF8=1 python -X utf8`，否則會遇到 cp1252 encoding error。

## Core Concepts

| 概念 | 說明 |
|------|------|
| **PRIM** | 原子操作 — deterministic function 或 LLM Signature |
| **ROUTING** | 可學習的路由決策（Method A）— GEPA 最佳化目標 |
| **COMP** | `dspy.Module` 組合器，串接 PRIMs + ROUTINGs |
| **GEPA** | Genetic Evolution Programming Architecture — 透過 reflective mutation 在 Pareto frontier 上最佳化 LLM prompt |
| **Harness** | 頂層 orchestrator，管理 semantic linker / runtime / scheduler / persistent store |

### 14 個 predictors

- 9 PRIMs: `describe_noise`, `describe_speech`, `describe_env`, `reason_scene`, `resolve_contradictions`, `gen_beam`, `gen_nr`, `parse_intent`, `update_preferences`
- 5 ROUTERs: `aggregate_router`, `scene_router`, `strategy_planner`, `strategy_integrator`, `pipeline_router`

## 研究背景 / Research Papers

`docs/` 目錄下的 PDF：

1. `01-LLM聲學研究意義` — Why LLMs matter for acoustic research
2. `02-語意ISA` — Semantic ISA 概念
3. `03-語意ISA-HarnessEngineering` — Harness engineering 設計
4. `04-語音評估的語意ISA相關研究` — Speech assessment 相關研究
5. `05-Semantic IR——從物理層到意圖層` — From physical layer to intent layer
6. `06-Acoustic Semantic IR (ASIR)` — Complete 7-layer specification

## Tech Stack

- **DSPy** ≥ 2.6 — LLM programming framework with GEPA support
- **NumPy** / **SciPy** — deterministic signal processing (L1-L3)
- **Matplotlib** — spectrogram 生成
- **Python** ≥ 3.10
- Recommended LM: `gpt-4o-mini`（fast_lm）+ `gpt-4o`（strong_lm）

## License

Research use only. 未定正式 license。
