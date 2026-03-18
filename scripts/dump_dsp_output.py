"""
Dump actual DSP output for wet_market_vendor scenario.
Shows the real values the system produces so README can be corrected.

Usage:
  PYTHONUTF8=1 OPENAI_API_KEY=sk-xxx python -X utf8 scripts/dump_dsp_output.py
"""
import os, sys, json
from pathlib import Path

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

import dspy
import numpy as np
from asir.harness import AcousticSemanticHarness
from asir.primitives.signal import prim_load_audio

SCENARIO_DIR = Path(__file__).parent.parent / "examples" / "audio" / "scenarios"

# Setup LMs
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)

fast_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
strong_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
dspy.configure(lm=fast_lm)

harness = AcousticSemanticHarness(
    fast_lm=fast_lm, strong_lm=strong_lm, enable_multimodal=False)

# Run both market scenarios
for scenario in ["wet_market_vendor", "market_too_muffled"]:
    wav_path = SCENARIO_DIR / f"{scenario}.wav"
    if not wav_path.exists():
        print(f"SKIP {scenario} — WAV not found"); continue

    signal = prim_load_audio(str(wav_path))
    if isinstance(signal, tuple):
        signal = signal[0]

    user_action = "太悶了" if scenario == "market_too_muffled" else "none"
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario}")
    print(f"  user_action: {user_action}")
    print(f"{'='*60}")

    result = harness(
        raw_signal=signal,
        user_action=user_action,
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
        audiogram_json='{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
    )

    # Strategy-level output (semantic)
    strat = result.strategy
    print(f"\n  === Strategy (L6 semantic output) ===")
    print(f"  target_azimuth_deg:  {strat.target_azimuth_deg}")
    print(f"  beam_width_deg:      {strat.beam_width_deg}")
    print(f"  nr_aggressiveness:   {strat.nr_aggressiveness}")
    print(f"  preserve_bands_json: {strat.preserve_bands_json}")
    print(f"  gain_per_frequency:  {strat.gain_per_frequency}")
    print(f"  compression_ratio:   {strat.compression_ratio}")
    print(f"  confidence:          {strat.confidence}")

    # DSP-level output (physical)
    dsp = result.dsp_params
    print(f"\n  === DSP Parameters (physical output) ===")
    print(f"  beam_weights:      {dsp.beam_weights}")
    print(f"  compression_ratio: {dsp.compression_ratio}")
    print(f"  attack_ms:         {dsp.attack_ms}")
    print(f"  release_ms:        {dsp.release_ms}")
    print(f"  noise_mask mean:   {np.mean(dsp.noise_mask):.3f}")
    print(f"  noise_mask range:  [{min(dsp.noise_mask):.3f}, {max(dsp.noise_mask):.3f}]")
    print(f"  filter_coeffs len: {len(dsp.filter_coeffs)}")

    # L5 scene
    print(f"\n  === Scene (L5) ===")
    print(f"  situation: {result.scene.situation[:200]}")

    # L4 percept
    print(f"\n  === Percept (L4) ===")
    print(f"  noise_desc:  {str(result.percept.noise_description)[:200]}")
    print(f"  speech_desc: {str(result.percept.speech_description)[:200]}")

    # Preferences after run
    print(f"\n  === Preferences ===")
    print(f"  {json.dumps(result.current_preferences, ensure_ascii=False, indent=2)}")

    # Reset harness for next scenario
    harness.current_preferences = {
        "noise_tolerance": "medium",
        "processing_preference": "natural",
        "environment_awareness": "moderate",
        "known_situations": ["菜市場: 增強正前方, 保留環境感"]
    }
    harness.feedback_history = []
