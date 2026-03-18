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

**問題**：傳統助聽器用固定規則做訊號處理（波束成形、降噪、增益）。
在複雜聲學場景（菜市場、多人對話、迴響空間），固定規則無法理解「使用者想聽什麼」。

**ASIR 的做法**：在傳統 DSP 之上加一層 LLM 語意推理。
系統先用 numpy 做 deterministic 訊號分析（L1-L3），
再用 LLM 理解場景語意、規劃處理策略（L4-L7），
最後把語意決策翻譯回具體的 DSP 參數。

### 使用情境

```
助聽器麥克風 → ASIR → DSP 硬體
                ↑
            使用者偏好/動作
```

一個 72 歲的聽損使用者戴著助聽器走進菜市場。ASIR：
1. 收到麥克風陣列的原始音訊（2 通道 PCM）
2. L1-L3：分析出 SNR=-2dB、RT60=0.8s、能量集中在低頻 → 吵雜迴響環境
3. L4：LLM 描述「正前方有人說話，背景是嘈雜的市場噪音和風扇聲」
4. L5：LLM 判斷「使用者在菜市場跟攤販對話」，信心度 0.85
5. L6：LLM 規劃策略 →「波束集中正前方 45°、啟用強降噪、NAL-NL2 增益」
6. 翻譯成 DSP 參數：beam_weights, noise_mask, filter_coeffs, compression 設定
7. L7：使用者說「太悶了」→ 更新偏好 → 下一幀降低降噪強度

### 系統輸入 / 輸出

**輸入（每幀）**：

| 參數 | 型別 | 說明 |
|------|------|------|
| `raw_signal` | `RawSignal` | 2-ch PCM, 16kHz, 32ms/frame |
| `user_action` | `str` | 使用者動作，如 `"太吵了"`, `"focus_front"`, `"none"` |
| `audiogram_json` | `str` | 聽力圖 JSON，如 `{"250":30, "500":35, ...}` (dB HL) |
| `user_profile` | `str` | 使用者描述，如 `"72歲男性，雙耳中度感音神經性聽損"` |

**輸出**：

| 欄位 | 型別 | 說明 |
|------|------|------|
| `dsp_params` | `DSPParameterSet` | beam_weights, noise_mask, filter_coeffs, compression_ratio, attack_ms, release_ms |
| `execution_depth` | `str` | 本幀執行深度：`"fast"` / `"medium"` / `"full"` |
| `scene_description` | `str` | L5 的場景理解文字（如有） |
| `strategy_summary` | `str` | L6 的策略摘要文字（如有） |

```python
# 程式碼層面
from asir.harness import AcousticSemanticHarness
from asir.types import RawSignal

harness = AcousticSemanticHarness(fast_lm=..., strong_lm=...)
result = harness.forward(
    raw_signal=raw_signal,
    user_action="focus_front",
    audiogram_json='{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
    user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
)
print(result.dsp_params)  # → DSPParameterSet(...)
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
