# ASIR — Acoustic Semantic Instruction Register

> **Research Prototype** — 研究用原型，API 隨時可能改動。

用 LLM 讓助聽器「聽懂」場景，動態調整 DSP 參數。

```
麥克風陣列 ──→ [ 訊號分析 → LLM 語意推理 → 策略規劃 ] ──→ DSP 參數
  2-ch PCM       L1-L3 numpy    L4-L5 場景理解   L6 策略     beam_weights
  16kHz                          L7 使用者偏好               noise_mask
                                                              gain, compression
                        ↑
                  使用者：「太悶了」「focus_front」
```

## 為什麼需要這個？

傳統助聽器用固定規則做降噪和增益。但在菜市場、多人對話、迴響大廳這些場景，
固定規則不知道「使用者想聽前面那個人說話」還是「想聽到整個環境」。

ASIR 在確定性 DSP 之上加一層 LLM 推理：先分析聲學特徵，再用語言模型理解場景語意，最後把語意決策翻譯回 DSP 硬體能執行的參數。

## 場景範例：菜市場

一個 72 歲聽損使用者戴著助聽器走進菜市場：

1. **收音** — 麥克風陣列收到 2 通道 PCM 原始音訊
2. **訊號分析（L1-L3）** — 算出 SNR=-2dB（噪音蓋過語音）、RT60=0.8s（中度迴響）、MFCC 特徵
3. **場景描述（L4）** — LLM：「正前方有人說話，背景是嘈雜市場噪音和風扇聲」
4. **場景理解（L5）** — LLM：「使用者在菜市場跟攤販對話」，信心度 0.85
5. **策略規劃（L6）** — LLM：「波束集中正前方 45°、啟用強降噪、NAL-NL2 增益」
6. **翻譯** — 語意策略 → `DSPParameterSet`（beam_weights, noise_mask, filter_coeffs, compression）
7. **使用者回饋（L7）** — 使用者說「太悶了」→ 更新偏好 → 下一幀降低降噪強度

## Quick Start

```bash
# L1-L3 deterministic demo（不需要 API key）
PYTHONUTF8=1 python -X utf8 -m examples.run_demo

# Full L1-L7 pipeline（需要 OPENAI_API_KEY）
PYTHONUTF8=1 OPENAI_API_KEY=sk-xxx python -X utf8 -m examples.run_demo --full

# Full + GEPA prompt optimization
PYTHONUTF8=1 OPENAI_API_KEY=sk-xxx python -X utf8 -m examples.run_demo --gepa
```

> Windows 必須加 `PYTHONUTF8=1 python -X utf8`，否則會遇到 cp1252 encoding error。

## Tech Stack

| | |
|---|---|
| **DSPy** ≥ 2.6 | LLM programming framework（Signature + Module + GEPA optimizer） |
| **NumPy / SciPy** | L1-L3 確定性訊號處理 |
| **Matplotlib** | 頻譜圖生成（dspy.Image） |
| **Python** ≥ 3.10 | |
| Recommended LM | `gpt-4o-mini`（fast_lm）+ `gpt-4o`（strong_lm） |

## 研究論文

`docs/` 目錄下的理論背景：

1. `01-LLM聲學研究意義` — Why LLMs matter for acoustic research
2. `02-語意ISA` — Semantic ISA 概念
3. `03-語意ISA-HarnessEngineering` — Harness engineering 設計
4. `04-語音評估的語意ISA相關研究` — Speech assessment 相關研究
5. `05-Semantic IR——從物理層到意圖層` — Physical layer to intent layer
6. `06-Acoustic Semantic IR (ASIR)` — Complete 7-layer specification

---

<details>
<summary><strong>領域術語 Glossary</strong>（點擊展開）</summary>

### 聽力學 (Audiology)

| 術語 | 說明 |
|------|------|
| **Audiogram（聽力圖）** | 各頻率（250–8000 Hz）的聽力閾值圖。本系統以 JSON 輸入，如 `{"250":30, "1000":40}` 表示該頻率損失幾 dB |
| **NAL-NL2** | 澳洲國家聲學實驗室的**增益處方公式** — 根據 audiogram 算出每個頻率該放大多少 dB。業界最廣泛使用的處方之一。`prim_generate_gain_params()` 實作簡化版 |
| **感音神經性聽損 (SNHL)** | 內耳或聽神經損傷造成的聽損，老年聽損最常見類型。高頻損失較嚴重、需要非線性放大 |
| **dB HL** | 聽力圖單位。0 = 正常聽力，30 = 輕度，50 = 中度，70 = 重度 |

### 訊號處理 (Signal Processing)

| 術語 | 說明 |
|------|------|
| **Beamforming（波束成形）** | 多麥克風空間濾波 — 增強特定方向、抑制其他方向。輸出 `beam_weights` |
| **Spectral Subtraction** | 從帶噪頻譜減去噪音頻譜 → 降噪。`comp_spectral_subtract()` |
| **Noise Mask** | 逐頻率增益遮罩，0–1。0 = 完全抑制，1 = 完全保留 |
| **SNR** | 訊噪比 (dB)。正 = 語音 > 噪音，負 = 噪音 > 語音 |
| **RT60** | 迴響衰減 60dB 的時間（秒）。0.3s = 小房間，>1s = 大廳 |
| **MFCC** | 模仿人耳頻率感知的特徵向量，語音辨識常用。L3 提取 |
| **FFT** | 時域 → 頻域轉換，所有頻域處理的基礎 |
| **Compression** | 動態範圍壓縮。`compression_ratio` 2:1 = 輸入 +2dB → 輸出 +1dB |
| **PCM** | 未壓縮數位音訊格式。`RawSignal.samples` 就是 PCM |

### LLM / DSPy

| 術語 | 說明 |
|------|------|
| **DSPy** | Stanford 的 LLM programming framework — Signature 定義 I/O，Module 組合，Optimizer 改進 prompt |
| **Signature** | LLM 任務的 I/O schema。如 `DescribeNoiseSig`：features → noise description |
| **ChainOfThought** | DSPy 推理模組 — 自動加入 step-by-step reasoning |
| **GEPA** | Genetic Evolution Programming Architecture — 演化策略最佳化 prompt，在 Pareto frontier 做 reflective mutation |
| **Method A** | 用 learnable predictor（非 hardcoded if-else）做路由決策，GEPA 主要最佳化目標 |

</details>

---

**Coding agents**: 技術細節（架構、I/O spec、package 結構、開發指南）見 [`CLAUDE.md`](CLAUDE.md)，invariants 和測試方式見 [`AGENTS.md`](AGENTS.md)。

License: Research use only. 未定正式 license。
