# AGENTS.md — Cross-Tool Agent Instructions

**不要過度工程化。** 能用一行解決就不要寫三行。不要加「以後可能用到」的抽象、工具、框架。先做最簡單的方案，只在證明不夠時才加複雜度。加功能前先查 DSPy 和 MLflow 是否已內建支援——不要重複造輪子。

This file provides context for AI coding agents (Claude Code, Codex, etc.).

## Repository Purpose

ASIR (Acoustic Semantic Instruction Register) is a 7-layer semantic IR
for hearing aids. It uses DSPy modules + GEPA optimizer to convert raw
audio into optimized DSP parameters through LLM-powered reasoning.

**使用情境**：助聽器在複雜場景（菜市場、多人對話）需動態調整 DSP。
傳統固定規則不懂語意，ASIR 用 LLM 理解場景後再決定 DSP 參數。

**I/O** (`harness.forward()`):
- Input: `RawSignal`(2-ch PCM) + `user_action` + `audiogram_json` + `user_profile`
- Output: `DSPParameterSet`(beam_weights, noise_mask, filter_coeffs, compression) + execution_depth + 語意中間結果

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
# 完整 demo（L1-L7 語意推理 + 使用者回饋，需要 OPENAI_API_KEY）
PYTHONUTF8=1 python -X utf8 -m examples.run_demo

# 只跑確定性層（不需要 API key）
PYTHONUTF8=1 python -X utf8 -m examples.run_demo --l1-l3
```

## Testing & Evaluation

**依序執行：**

```bash
# Step 0: 合成 10 個場景測試音檔（只需跑一次，存在 asir/eval/audio/scenarios/）
PYTHONUTF8=1 python -X utf8 -m asir.eval.generate_audio

# Step 1: L1-L3 + 場景一致性 pytest（no API key, 60 tests, ~10s）
#   驗證: 確定性管線 + 10 場景 WAV 載入 + eval examples 與 WAV 檔 1:1 對應
#   ★ 無 LLM 推理，不產生推理 trace
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_deterministic.py -v

# Step 2: L4-L7 語意推理 pytest（needs OPENAI_API_KEY, 52 tests, ~2min）
#   驗證: 10 場景 × 5 層 + 菜市場 E9/E10 比較
#   ★ 推理 trace 記錄到 MLflow (experiment: asir-eval-pytest)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_semantic.py -v

# Step 3: Integration 端對端 pytest（needs OPENAI_API_KEY, 50 tests, ~10min）
#   驗證: 真實音檔 → 完整 harness → 每層 metrics check
#   ★ 推理 trace 記錄到 MLflow (experiment: asir-integration-pytest)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_integration.py -v

# 或者用 standalone eval runners（同樣邏輯，更詳細的 console trace 輸出）
PYTHONUTF8=1 python -X utf8 -m asir.eval
PYTHONUTF8=1 python -X utf8 -m asir.eval --integration
```

**所有 LLM 推理結果記錄到 MLflow。** 查看歷史與比較：`mlflow ui`

### 跑完之後去哪裡看推理過程

| 跑了什麼 | 推理 trace 在哪 | 怎麼看 |
|----------|----------------|--------|
| `test_deterministic.py` | 無（L1-L3 確定性，無 LLM） | 看 console output |
| `test_semantic.py` | MLflow `asir-eval-pytest` → `pytest_eval_results.json` | `mlflow ui` |
| `test_integration.py` | MLflow `asir-integration-pytest` → `pytest_integration_results.json` | `mlflow ui` |
| `run_demo` | MLflow `asir-demo` → `demo_trace.json` | `mlflow ui` |
| `asir.eval` | MLflow `asir-eval` → `eval_results.json` | `mlflow ui` 或 `download_artifacts("eval_results.json")` |
| `asir.eval --integration` | MLflow `asir-eval` → `integration_results.json` | `mlflow ui` |

**跑完 eval 後，讀 MLflow artifact 中的 trace 判讀每層推理是否合理，不要只看分數。**

**Agent 自動判讀規範：**
- eval 跑完後，用 `mlflow.search_runs()` + `download_artifacts("eval_results.json")` 讀取結構化 trace
- 逐場景檢查：L4 噪音描述是否反映 SNR、L5 場景是否合理、L6 NR 是否匹配噪音程度、L7 偏好是否正確更新
- 判斷失敗原因：**LLM 推理偏差**（→ GEPA 優化目標，不改 code）vs **程式邏輯 bug**（→ 需要修 code）
- 不要只報「2 failures」——要說明失敗的具體原因和是否需要行動

### Trace — 每場景推理鏈

每個場景自動印出 L4→L5→L6→DSP→L7 推理鏈（不需要額外 flag）。
Trace 包含完整推理內容，同時存到 MLflow artifacts。

即使 check 全部 pass，數值可能仍不合理（如安靜場景 NR=0.7）。
Trace 讓人（或 agent）目視確認推理是否符合場景特性。

### GEPA 優化

```bash
PYTHONUTF8=1 OPENAI_API_KEY=sk-xxx python -X utf8 -m examples.run_demo --gepa
```

GEPA 完成後自動存檔到 `programs/gepa_{timestamp}/`，MLflow 記錄完整優化過程。

### 載入優化後的 Program

```bash
PYTHONUTF8=1 python -X utf8 -m asir.eval --program programs/gepa_xxx/program.json
PYTHONUTF8=1 python -X utf8 -m asir.eval --integration --program programs/gepa_xxx/program.json
```

### Eval Scenarios and README Alignment

10 個 eval 場景定義在 `asir/eval/examples.py`，其中兩個直接對應 README 的旗艦展示：

| Scenario | 對應 README | 驗證重點 |
|----------|-----------|---------|
| `wet_market_vendor` | "場景範例：菜市場" | SNR=0dB 極吵 → 強降噪 + 波束聚焦 |
| `market_too_muffled` | "使用者回饋：太悶了" | user_action="太悶了" → 降低降噪 + 偏好持久化 |

**如果你改了 README 的場景描述，eval 也要對應更新（反之亦然）。**

### Interpreting Eval Results

eval 分數受 LLM 非確定性影響，每次跑會有 ±5% 波動。

**已知 GEPA 優化目標（不是 bug）：**
- L6 `nr_matches_scene` — NR aggressiveness 有時低於 0.4 threshold
- DSP `beam_appropriate` — 安靜場景 LLM 有時給窄波束（應該給寬波束）

**真正需要修的 bug 特徵：** 同一個 check **每次跑都失敗**，且原因是程式邏輯錯誤。

### Key Behavior: user_action and Preferences

`harness.py` 對任何 `user_action != "none"` 都呼叫 `UpdatePreferencesSig`。
LLM 語意決定偏好要不要改，沒有硬編碼 keyword 判斷。

## Test Audio Files

`asir/eval/audio/` — eval 的 code 和資料在同一個目錄下：
```
asir/eval/audio/
├── audio1.wav, audio2.wav     # Gemini TTS samples (24kHz mono)
├── speech/                    # Clean speech clips (16kHz mono)
├── noise/                     # (Optional) DEMAND dataset noise clips
└── scenarios/                 # Mixed scenario audio (stereo 16kHz, 5s each)
```

Generator: `asir/eval/generate_audio.py`
WAV 檔案使用 Git LFS 追蹤。

## Common Tasks

- **Add new Signature**: Create in `asir/primitives/`, export in `__init__.py`
- **Add routing logic**: Create in `asir/routing/`, wire in composite
- **Modify GEPA feedback**: Edit `asir/gepa/metric.py`
- **Change pipeline flow**: Edit `asir/harness.py`
- **Add new eval scenario**: 改 `examples.py` + `generate_audio.py`，跑 generate + eval
- **Debug eval failure**: 看 trace 輸出或 `mlflow ui` 查看歷史 artifacts

## Artifact Directories

```
mlruns/               # MLflow 追蹤（eval + GEPA 結果，mlflow ui 查看）
runs/                 # GEPA 優化過程日誌
programs/             # 優化後的 program 存檔
```
