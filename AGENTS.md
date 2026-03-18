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

Three tools, each does one job. **Run in this order:**
```bash
# Step 0: Generate scenario WAVs (only if missing or after adding new scenarios)
PYTHONUTF8=1 python -X utf8 -m asir.eval.generate_audio

# Step 1: L1-L3 pytest (deterministic, no API key, ~7s)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_deterministic.py -v

# Step 2: L4-L7 semantic eval (needs OPENAI_API_KEY, ~2min)
PYTHONUTF8=1 python -X utf8 -m asir.eval

# Step 3: Integration eval — real audio → full pipeline (needs OPENAI_API_KEY, ~10min)
PYTHONUTF8=1 python -X utf8 -m asir.eval --integration
```

- Integration eval **需要 scenario WAVs 存在**。如果 `examples/audio/scenarios/` 裡缺音檔，先跑 Step 0
- L4-L7 eval 直接注入 AcousticFeatures 到 L4，繞過 L1-L3 隨機訊號
- Integration eval 用真實音檔走完整 harness pipeline (L1→L7)
- Constraint 欄位的對應見 `asir/eval/metrics.py` 頂部 mapping 表
- 輸出: `eval_results.json` (semantic) / `integration_results.json` (integration)

### Trace — 每場景推理鏈摘要 (v0.9+)

每個場景跑完都會自動印出 L4→L5→L6→DSP→L7 的推理鏈摘要，**不需要額外 flag**：

```bash
# 跑 eval 就會看到每個場景的 trace
PYTHONUTF8=1 python -X utf8 -m asir.eval
PYTHONUTF8=1 python -X utf8 -m asir.eval --integration
```

Trace 輸出包含：L4 noise/speech/env description、L5 situation、L6 beam/NR/reasoning、DSP mask stats、L7 depth + preferences。

**為什麼每場景都印？** 即使所有 check pass，數值可能仍不合理（如安靜場景 NR=0.7）。
Trace 讓人（或 agent）目視確認每一層的推理是否符合場景特性。

如果有 check fail，trace 尾端會額外列出失敗的 check 清單。

### GEPA 優化 + Program 存檔 (v0.9+)

```bash
# 跑 GEPA 優化（~5-10 分鐘）
PYTHONUTF8=1 OPENAI_API_KEY=sk-xxx python -X utf8 -m examples.run_demo --gepa
```

GEPA 完成後會自動：
1. 印出 **優化摘要**（metric calls、candidates、Pareto front）
2. 印出 **instruction diffs**（每個 predictor 優化前後的 prompt 變化）
3. 存檔到 `programs/gepa_{timestamp}/program.json` + `metadata.json`
4. 若安裝 MLflow (`pip install mlflow`)，自動記錄到 MLflow

### A/B 對比 — baseline vs 優化後 (v0.9+)

```bash
# baseline vs 優化後的 program
PYTHONUTF8=1 python -X utf8 -m asir.eval --compare programs/gepa_xxx/program.json

# 兩個版本互相比較
PYTHONUTF8=1 python -X utf8 -m asir.eval --compare programs/v1/program.json programs/v2/program.json
```

會跑同一組 eval examples 各一次，產出並排比較表（逐場景、逐 layer、標示 +/-）。

### 自動 Delta 對比 (v0.9+)

每次跑 `python -m asir.eval` 或 `--integration` 都會**自動比對上一次結果**，印出逐 layer delta：
```
  vs Previous Run (baseline, 20260317_120000):
      L4: 75% → 80%  (+5%)
      L6: 90% → 85%  (-5%)
```

結果同時存到 `results/` 目錄（時間戳檔名，累積歷史）+ `eval_results.json`（覆蓋，最新一筆）。

### 載入優化後的 Program 跑 eval

```bash
# Semantic eval 用優化後的 program
PYTHONUTF8=1 python -X utf8 -m asir.eval --program programs/gepa_xxx/program.json

# Integration eval 用優化後的 program
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

eval 分數受 LLM 非確定性影響，每次跑會有 ±5% 波動。以下是已知的狀態：

**已知的 GEPA 優化目標（不是 bug，是 LLM 需要學習的）：**
- L4 `noise_consistent` — LLM 有時不在 JSON severity 欄位寫 "high"，即使 SNR<5dB
- L6 `nr_matches_scene` — NR aggressiveness 有時低於 0.4 threshold
- DSP `beam_appropriate` — 安靜場景 LLM 有時給窄波束（應該給寬波束）
- L7 `preference_stable` — "focus_front" 等指令型 user_action，LLM 有時會多更新偏好

**真正需要修的 bug 特徵：**
- 同一個 check 在**每次跑都失敗**（不是偶爾）
- 失敗原因是**程式邏輯錯誤**而不是 LLM 輸出品質

### Key Behavior: user_action and Preferences (v0.8+)

`harness.py` 對 **任何 `user_action != "none"`** 都會呼叫 `UpdatePreferencesSig`。
LLM 語意決定偏好要不要改，沒有硬編碼 keyword 判斷。

這表示：
- 中文回饋（"太悶了"、"太吵了"）會正確觸發偏好更新
- 指令型動作（"focus_front"）也會經過 LLM 判斷，但理想上不該改偏好
- 偏好更新後會持久化到 `self.current_preferences`，影響後續每一幀的策略生成

## Test Audio Files

`examples/audio/` structure:
```
examples/audio/
├── audio1.wav, audio2.wav     # Gemini TTS samples (24kHz mono, for prim_load_audio tests)
├── speech/                    # Clean speech clips (16kHz mono)
│   ├── speech_1.wav           # "你好，今天晚餐想吃什麼？"
│   ├── speech_2.wav           # "請問這個多少錢？"
│   └── speech_3.wav           # "不好意思，請問到捷運站怎麼走？"
├── noise/                     # (Optional) DEMAND dataset noise clips
│   └── PRESTO_ch01.wav, ...   # See "Upgrading to DEMAND" below
└── scenarios/                 # Mixed scenario audio (stereo 16kHz, 5s each)
    ├── restaurant_dinner.wav  # SNR=3dB,  energy=72dB SPL
    ├── church_ceremony.wav    # SNR=12dB, energy=60dB SPL, RT60=2.5s
    ├── quiet_library.wav      # SNR=30dB, energy=40dB SPL
    ├── wet_market_vendor.wav  # SNR=0dB,  energy=78dB SPL (README 旗艦場景)
    ├── market_too_muffled.wav # SNR=0dB,  energy=78dB SPL (「太悶了」回饋)
    └── ...                    # 10 total, one per eval scenario
```

### How scenario audio is generated

Generator: `asir/eval/generate_audio.py`
```bash
PYTHONUTF8=1 python -X utf8 -m asir.eval.generate_audio
```

Pipeline:
1. **Speech** — Gemini TTS (`gemini-2.5-flash-preview-tts`, needs `GEMINI_API_KEY` in `.env`)
   - 3 Chinese daily conversation sentences, 24kHz → resample to 16kHz
   - Falls back to amplitude-modulated harmonic signal if no API key
2. **Noise** — Synthetic with spectral shaping (numpy + scipy.signal)
   - Each scenario maps to a noise type: babble, traffic, car, crowd, quiet, ambient
   - See `NOISE_TYPE_MAP` in generate_audio.py for the mapping
3. **Mixing** — `mix_at_snr(speech, noise, target_snr_db)` controls exact SNR
4. **Reverb** — Exponential decay RIR for RT60 ≥ 0.2s
5. **Energy** — Normalized to match `energy_db` from eval examples
   - Formula: `target_rms = 10^((energy_db - 94) / 20)`
   - So `comp_extract_full_features()` should recover similar energy_db

### Upgrading to DEMAND noise (optional)

[DEMAND](https://zenodo.org/records/1227121) (CC-BY 4.0) provides real-world
16-channel noise recordings. To use them instead of synthetic noise:

1. Download per-environment 16kHz zips from Zenodo:
   - `PRESTO_16k.zip` (restaurant), `TCAR_16k.zip` (car), `STRAFFIC_16k.zip` (traffic)
   - `PCAFETER_16k.zip` (cafeteria), `PSTATION_16k.zip` (station)
   - `OOFFICE_16k.zip` (office), `OHALLWAY_16k.zip` (hallway), `DLIVING_16k.zip` (living)
2. Extract channel 01 WAV from each, trim to 5+ seconds
3. Place in `examples/audio/noise/` as `{NAME}_ch01.wav` (e.g. `PRESTO_ch01.wav`)
4. Re-run: `python -m asir.eval.generate_audio` — it auto-detects DEMAND files

Mapping: see `DEMAND_MAP` in `asir/eval/generate_audio.py`

### Loading audio for testing

```bash
PYTHONUTF8=1 python -X utf8 -c "
from asir.primitives.signal import prim_load_audio
sig, audio_obj = prim_load_audio('examples/audio/scenarios/restaurant_dinner.wav')
print(f'Loaded: {sig.n_channels}ch, {sig.sample_rate}Hz, {sig.duration_ms:.0f}ms')
"
```

## Development Documents

`docs/` contains design documents and development background (PDF):
- `01-LLM聲學研究意義` — Why use LLMs for acoustic processing
- `02-語意ISA` — Semantic ISA concept design
- `03-語意ISA-HarnessEngineering` — Harness engineering design
- `04-語音評估的語意ISA相關研究` — Speech assessment background
- `05-Semantic IR——從物理層到意圖層` — Architecture evolution from physical to intent layer
- `06-Acoustic Semantic IR (ASIR)` — Full 7-layer specification

## Common Tasks

- **Add new Signature**: Create in `asir/primitives/`, export in `__init__.py`
- **Add routing logic**: Create in `asir/routing/`, wire in composite
- **Modify GEPA feedback**: Edit `asir/gepa/metric.py`
- **Change pipeline flow**: Edit `asir/harness.py`
- **Add new eval scenario**: 三個地方都要改，然後重新生成音檔：
  1. `asir/eval/examples.py` — 加 `dspy.Example` 含 scenario name + 聲學參數 + 約束
  2. `asir/eval/generate_audio.py` — 在 `SCENARIOS`, `NOISE_TYPE_MAP`, `DEMAND_MAP` 三個 dict 加對應項
  3. 跑 `python -m asir.eval.generate_audio` 生成新 WAV
  4. 跑 `python -m asir.eval --integration` 驗證
- **跑 GEPA 優化並比較**:
  1. `python -m examples.run_demo --gepa` → 自動存檔到 `programs/`
  2. `python -m asir.eval --compare programs/gepa_xxx/program.json` → A/B 對比
- **Debug eval failure**: `python -m asir.eval --verbose` → 看完整 L4→L6 推理鏈

## Artifact Directories

```
results/                 # eval 歷史結果（gitignored，每次 eval 自動累積）
├── eval_{timestamp}.json           # semantic eval 結果
└── integration_{timestamp}.json    # integration eval 結果

runs/                    # GEPA 優化過程日誌（gitignored）
└── gepa/{timestamp}/    # GEPA log_dir 輸出

programs/                # 優化後的 program 存檔（gitignored）
└── gepa_{timestamp}/
    ├── program.json     # dspy.Module state (可用 Module.load 載入)
    └── metadata.json    # 優化配置 + 統計（metric calls, candidates 等）
```
