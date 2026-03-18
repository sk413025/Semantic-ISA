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

1. **收音** — 麥克風陣列收到 2 通道原始音訊
2. **訊號分析（L1-L3）** — 算出 SNR、RT60、MFCC（見下方架構圖的 L1-L3 註解）
3. **場景描述（L4）** — LLM：「正前方有人說話，背景是嘈雜市場噪音和風扇聲」
4. **場景理解（L5）** — LLM：「使用者在菜市場跟攤販對話」，信心度 0.85
5. **策略規劃（L6）** — LLM：「波束集中正前方 45°、啟用強降噪、NAL-NL2 增益」
6. **翻譯** — 語意策略 → DSP 參數（beam_weights, noise_mask, compression...）
7. **使用者回饋（L7）** — 「太悶了」→ 更新偏好 → 下一幀降低降噪強度

## 七層架構與領域術語

術語直接標注在它被使用的層級，不用另外查表：

```
L1 Physical Sensing                            [deterministic, numpy]
    輸入: PCM 原始音訊 (未壓縮浮點數, 每 sample 一個值)
    麥克風陣列 --> RawSignal (2-ch, 16kHz, 32ms/frame)

L2 Signal Processing                           [deterministic, numpy]
    FFT            -- 時域-->頻域轉換, 所有頻域處理的前置步驟
    Beamforming    -- 多麥克風空間濾波, 增強某方向 --> beam_weights
    Spectral Sub.  -- 頻譜減去噪音估計 --> 降噪

L3 Acoustic Features                           [deterministic, numpy]
    SNR   -- 訊噪比 (dB), 負值=噪音蓋過語音, 菜市場約 -5~5 dB
    RT60  -- 迴響衰減 60dB 的秒數, 0.3s=小房間, >1s=大廳
    MFCC  -- 模仿人耳頻率感知的特徵向量

====================== SEMANTIC BOUNDARY ======================

L4 Perceptual Description              [DSPy ChainOfThought, fast_lm]
    3 個 Signature 分別描述噪音/語音/環境
    aggregate_router -- learnable 路由, 非 hardcoded if-else
                        (Method A: GEPA 主要最佳化目標)

L5 Scene Understanding                [DSPy ChainOfThought, strong_lm]
    綜合 L4 描述 + 場景歷史 --> 場景判斷
    scene_router -- 決定是否啟動矛盾解決

====================== SEMANTIC BOUNDARY ======================

L6 Strategy Generation                [DSPy ChainOfThought, strong_lm]
    strategy_planner --> gen_beam + gen_nr + gain --> integrator
    NAL-NL2      -- 根據 Audiogram 算每頻率放大量的處方公式
      Audiogram  = 各頻率聽力閾值 {"250":30, "1000":40} (dB HL)
    輸出:
      Noise Mask   -- 逐頻率 0-1 遮罩 (0=抑制, 1=保留)
      Compression  -- 動態壓縮 (ratio 2:1 = +2dB輸入 --> +1dB輸出)

L7 Intent & Preference                [DSPy ChainOfThought, strong_lm]
    解析使用者動作 ("太吵了" / "focus_front")
    SNHL   -- 感音神經性聽損, 老年聽損主因, 高頻損失較嚴重
    dB HL  -- 聽力圖單位: 0=正常, 30=輕度, 50=中度, 70=重度
    更新偏好 --> 影響下一幀的 L4-L6 策略

--------------------------------------------------------------
DSPy           = Stanford LLM framework (Signature + Module + Optimizer)
GEPA           = 演化策略在 Pareto frontier 最佳化 LLM prompt
ChainOfThought = DSPy 推理模組, 自動加入 step-by-step reasoning
Method A       = 用 learnable predictor 做路由, GEPA 最佳化目標
```

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

## 開發文件

`docs/` 目錄下的開發背景與設計文件：

1. `01-LLM聲學研究意義` — 為什麼用 LLM 做聲學處理
2. `02-語意ISA` — Semantic ISA 概念設計
3. `03-語意ISA-HarnessEngineering` — Harness engineering 設計
4. `04-語音評估的語意ISA相關研究` — 語音評估相關背景
5. `05-Semantic IR——從物理層到意圖層` — 從物理層到意圖層的架構演進
6. `06-Acoustic Semantic IR (ASIR)` — 完整七層規格書

---

**Coding agents**: 技術細節（I/O spec、package 結構、開發指南）見 [`CLAUDE.md`](CLAUDE.md)，invariants 和測試方式見 [`AGENTS.md`](AGENTS.md)。

License: Research use only. 未定正式 license。
