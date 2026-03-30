# ASIR — Acoustic Semantic Instruction Register

> **Research Prototype** — Experimental API, subject to change.

ASIR uses LLM reasoning to help hearing devices interpret acoustic scenes and
adapt DSP parameters dynamically.

```text
Microphone Array ──→ [ Signal Analysis → LLM Semantic Reasoning → Strategy Planning ] ──→ DSP Parameters
  2-ch PCM             L1-L3 numpy        L4-L5 scene understanding    L6 strategy      beam_weights
  16kHz                                      L7 user preferences                          noise_mask
                                                                                         gain, compression
                                  ↑
                         User feedback: "too muffled", "focus_front"
```

## Why This Exists

Traditional hearing aids rely on fixed rules for noise reduction and gain.
Those rules struggle in settings like wet markets, overlapping conversations,
or reverberant halls because they do not understand whether the user wants to
focus on one speaker or preserve overall environmental awareness.

ASIR adds an LLM reasoning layer on top of deterministic DSP. It first extracts
acoustic features, then interprets scene semantics with a language model, and
finally translates the semantic decision back into DSP parameters that real
hardware can execute.

## Example Scenario: Wet Market Conversation

Consider a 72-year-old user with hearing loss walking into a wet market while
wearing hearing aids:

1. **Capture** — the microphone array records 2-channel audio.
2. **Signal analysis (L1-L3)** — the system estimates SNR≈0 dB, RT60≈0.6 s,
   8 active sources, and 78 dB SPL.
3. **Perceptual description (L4)** — the LLM describes noise, speech, and
   environment from numeric features or spectrograms.
4. **Scene understanding (L5)** — the LLM infers scene type and listening
   challenges.
5. **Strategy generation (L6)** — the LLM decides beam direction, noise
   reduction strength, and gain strategy.
6. **Translation** — semantic strategy becomes DSP parameters such as
   `beam_weights`, `noise_mask`, and `compression_ratio`.
7. **User feedback (L7)** — a complaint like "too muffled" updates user
   preferences and adjusts the next frame.

### Demo Snapshot (`run_demo`, `gpt-4o-mini`, `multimodal=True`)

> **L4:** "ambient noise, varied direction, modulated, moderate" +
> "crowded indoor environment, complex, reverberant"
>
> **L5:** "crowded indoor space with multiple overlapping sound sources"
> (`confidence=0.85`)
>
> **L6:** `beam=0°`, `width=60°`, `NR=0.4`, `compression=1.86`
>
> **L7:** user says "too muffled" → NR drops to `0.3`,
> `noise_tolerance: "medium" -> "low"`
>
> In the current prototype, the model can often identify
> "multiple voices + reverberation" from spectrograms, but may still confuse
> a wet market with another crowded indoor environment. A likely next step is
> adding wet-market-specific audio cues such as metal impacts and motors, plus
> further GEPA prompt optimization.

### Design Targets vs Current Prototype

| Dimension | Target | Current state | Improvement path |
|---|---|---|---|
| L4/L5 scene recognition | "wet market conversation with a vendor" | "crowded indoor space" | add wet-market acoustic cues + GEPA |
| L6 noise reduction | strong reduction (`NR > 0.5`) | `NR = 0.4` | optimize prompts with GEPA |
| L7 feedback loop | "too muffled" lowers NR | `0.4 -> 0.3` + preferences updated | already working |

## Seven-Layer Stack and Domain Terms

```text
L1 Physical Sensing                            [deterministic, numpy]
    Input: raw PCM audio (uncompressed floating-point samples)
    Microphone array --> RawSignal (2-ch, 16kHz, 32ms/frame)

L2 Signal Processing                           [deterministic, numpy]
    FFT            -- time-domain to frequency-domain transform
    Beamforming    -- multi-microphone spatial filtering --> beam_weights
    Spectral Sub.  -- subtract estimated noise spectrum --> denoising

L3 Acoustic Features                           [deterministic, numpy]
    SNR   -- signal-to-noise ratio (dB), negative means noise dominates speech
    RT60  -- seconds needed for reverberation to decay by 60 dB
    MFCC  -- features that approximate human auditory frequency perception

====================== SEMANTIC BOUNDARY ======================

L4 Perceptual Description                      [DSPy ChainOfThought, fast_lm]
    Three Signatures describe noise, speech, and environment
    aggregate_router -- learnable routing, not hard-coded if/else
                        (Method A: a primary GEPA target)

L5 Scene Understanding                        [DSPy ChainOfThought, strong_lm]
    Combines L4 descriptions with scene history --> scene judgment
    scene_router -- decides whether contradiction resolution is needed

====================== SEMANTIC BOUNDARY ======================

L6 Strategy Generation                        [DSPy ChainOfThought, strong_lm]
    strategy_planner --> gen_beam + gen_nr + gain --> integrator
    NAL-NL2      -- prescription formula mapping audiogram to per-band gain
    Outputs:
      Noise Mask   -- per-frequency 0-1 mask (0=suppress, 1=preserve)
      Compression  -- dynamic range compression

L7 Intent & Preference                        [DSPy ChainOfThought, strong_lm]
    Interprets user actions ("too noisy", "focus_front")
    SNHL          -- sensorineural hearing loss
    dB HL         -- audiogram unit (0=normal, 30=mild, 50=moderate, 70=severe)
    Updates preferences --> influences the next frame's L4-L6 strategy

--------------------------------------------------------------
DSPy           = Stanford LLM framework (Signature + Module + Optimizer)
GEPA           = Pareto-frontier optimization for LLM prompts
ChainOfThought = DSPy reasoning module with step-by-step reasoning
Method A       = learnable predictors used for routing and GEPA optimization
```

## Quick Start

```bash
# Step 0: generate 10 scenario test WAV files (wet market, restaurant, church, etc.)
# Only needs to be run once.
PYTHONUTF8=1 python -X utf8 -m asir.eval.generate_audio

# Step 1: full demo — wet market → semantic reasoning → DSP →
# "too muffled" feedback → preference update
# Requires OPENAI_API_KEY. Traces are logged to MLflow.
PYTHONUTF8=1 python -X utf8 -m examples.run_demo

# Step 2: deterministic tests (L1-L3 + scenario consistency)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_deterministic.py -v

# Step 3: semantic tests (L4-L7 reasoning quality, requires API key)
PYTHONUTF8=1 python -X utf8 -m pytest tests/test_semantic.py -v
```

> No API key? Run `python -m examples.run_demo --l1-l3` for deterministic layers only.

> On Windows, keep `PYTHONUTF8=1 python -X utf8` to avoid cp1252 encoding issues.

### Test Audio Assets

All evaluation audio lives under `asir/eval/audio/` and is tracked with Git LFS:

```text
asir/eval/audio/
├── scenarios/          # 10 mixed scenario WAV files (stereo 16kHz, 5s)
│   ├── wet_market_vendor.wav
│   ├── market_too_muffled.wav
│   ├── restaurant_dinner.wav
│   └── ...
├── speech/             # 3 clean speech clips (TTS, 16kHz mono)
└── noise/              # optional DEMAND noise dataset
```

**Where these files are used**
- `examples/run_demo.py` — end-to-end demo using the wet-market scenario
- `tests/test_deterministic.py` — L1-L3 pytest coverage on all 10 scenario WAVs
- `tests/test_semantic.py` — semantic evaluation using scenario definitions
- `tests/test_integration.py` — real-audio end-to-end harness validation
- `asir/eval/run.py` — semantic evaluation runner
- `asir/eval/integration.py` — integration evaluation runner

**How they are generated**
- `asir/eval/generate_audio.py` synthesizes different noise types
  (`babble`, `market`, `traffic`, etc.), mixes TTS speech, and adds reverberation.

## Tech Stack

| Component | Role |
|---|---|
| **DSPy** `>= 2.6` | LLM programming framework (`Signature`, `Module`, `GEPA`) |
| **NumPy / SciPy** | deterministic L1-L3 signal processing |
| **Matplotlib** | spectrogram generation for `dspy.Image` |
| **Python** `>= 3.10` | runtime |
| Recommended models | `gpt-4o-mini` (`fast_lm`) + `gpt-4o` (`strong_lm`) |

## Development Documents

The `docs/` directory stores archived design and research notes related to ASIR,
LLM-guided acoustics, harness engineering, speech evaluation, and the full
seven-layer Acoustic Semantic IR design.

---

**Coding agents**: implementation details such as I/O specs, package layout,
and development guidance live in [`CLAUDE.md`](CLAUDE.md). Operational
invariants and testing workflow live in [`AGENTS.md`](AGENTS.md).

License: research use only. No formal license has been assigned yet.
