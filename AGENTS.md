# AGENTS.md — Cross-Tool Agent Instructions

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
# Always use UTF-8 on Windows
PYTHONUTF8=1 python -X utf8 -m examples.run_demo
```

## Testing & Evaluation

**Run in this order:**
```bash
# Step 0: Generate scenario WAVs (only if missing)
PYTHONUTF8=1 python -X utf8 -m asir.eval.generate_audio

# Step 1: L1-L3 pytest (deterministic, no API key, ~7s)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_deterministic.py -v

# Step 2: L4-L7 semantic eval (needs OPENAI_API_KEY, ~2min)
PYTHONUTF8=1 python -X utf8 -m asir.eval

# Step 3: Integration eval — real audio → full pipeline (~10min)
PYTHONUTF8=1 python -X utf8 -m asir.eval --integration
```

**所有 eval 結果記錄到 MLflow。** 查看歷史與比較：`mlflow ui`

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

`examples/audio/` structure:
```
examples/audio/
├── audio1.wav, audio2.wav     # Gemini TTS samples (24kHz mono)
├── speech/                    # Clean speech clips (16kHz mono)
├── noise/                     # (Optional) DEMAND dataset noise clips
└── scenarios/                 # Mixed scenario audio (stereo 16kHz, 5s each)
```

Generator: `asir/eval/generate_audio.py`

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
