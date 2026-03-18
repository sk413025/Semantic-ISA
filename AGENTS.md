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

Three tools, each does one job:
```bash
# L1-L3: pytest (deterministic, no API key)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_deterministic.py -v

# L4-L7: semantic eval — inject features directly (needs OPENAI_API_KEY)
PYTHONUTF8=1 python -X utf8 -m asir.eval

# Integration: real audio → full pipeline (needs OPENAI_API_KEY)
PYTHONUTF8=1 python -X utf8 -m asir.eval --integration
```

- L4-L7 eval 直接注入 AcousticFeatures 到 L4，繞過 L1-L3 隨機訊號
- Integration eval 用真實音檔走完整 harness pipeline (L1→L7)
- Constraint 欄位的對應見 `asir/eval/metrics.py` 頂部 mapping 表
- 輸出: `eval_results.json` (semantic) / `integration_results.json` (integration)

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
