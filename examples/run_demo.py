"""
ASIR demo — run the full 7-layer pipeline.

Usage:
    python -m examples.run_demo           # full L1-L7 pipeline (needs API key)
    python -m examples.run_demo --gepa    # full pipeline + GEPA optimization
    python -m examples.run_demo --l1-l3   # deterministic layers only (no API key)
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path for direct execution.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asir.architecture import ARCHITECTURE_MAP
from asir.gepa.compiler import compile_with_gepa
from asir.harness import AcousticSemanticHarness
from asir.primitives import (
    comp_extract_full_features,
    comp_spectral_subtract,
    prim_beamform,
    prim_estimate_noise_psd,
    prim_fft,
    prim_generate_gain_params,
    prim_sample_audio,
)
from asir.primitives.signal import prim_load_audio

SCENARIO_DIR = Path(__file__).parent.parent / "asir" / "eval" / "audio" / "scenarios"
USER_PROFILE = "72-year-old man with bilateral moderate sensorineural hearing loss who prefers natural sound"
AUDIOGRAM = '{"250":30,"500":35,"1000":40,"2000":50,"4000":60}'


def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_load_env()


def demo_deterministic_layers():
    """
    Run the first three deterministic layers end to end without an LLM API.

    These outputs become the inputs to the L4 semantic modules.
    """
    print("=" * 70)
    print("DEMO: deterministic execution of layers 1-3")
    print("Scenario: Mr. Li, 12:15 PM, wet market fish stall")
    print("=" * 70)

    # Use a real WAV file when available; otherwise synthesize a signal.
    wav_path = SCENARIO_DIR / "wet_market_vendor.wav"
    if wav_path.exists():
        print(f"\n[Layer 1] Physical Sensing — prim_load_audio({wav_path.name})")
        signal = prim_load_audio(str(wav_path))
        if isinstance(signal, tuple):
            signal = signal[0]
    else:
        print("\n[Layer 1] Physical Sensing — prim_sample_audio() (synthetic)")
        signal = prim_sample_audio(duration_ms=32.0, n_channels=2)

    print(f"  Channels: {signal.n_channels}")
    print(f"  Sample rate: {signal.sample_rate} Hz")
    print(f"  Duration: {signal.duration_ms} ms")
    print(f"  Samples per channel: {len(signal.samples[0])}")

    # Layer 2
    print("\n[Layer 2] Signal Processing")
    spectrum = prim_fft(signal)
    print(f"  [PRIM] FFT: {spectrum['freq_bins']} frequency bins")

    noise_psd = prim_estimate_noise_psd(signal)
    print(f"  [PRIM] Noise PSD: estimated {sum(1 for x in noise_psd if x > 0)} active bins")

    beamformed = prim_beamform(signal, target_azimuth_deg=0.0)
    print(f"  [PRIM] Beamform(0°): {len(beamformed)} samples")

    cleaned = comp_spectral_subtract(signal, noise_psd, alpha=1.0)
    print(f"  [COMP] Spectral subtract: {len(cleaned)} samples")

    # Layer 3
    print("\n[Layer 3] Acoustic Features")
    features = comp_extract_full_features(signal)
    print("  [COMP] Full features extracted:")
    print(f"    SNR: {features.snr_db} dB")
    print(f"    RT60: {features.rt60_s} s")
    print(f"    Active sources: {features.n_active_sources}")
    print(f"    Spectral centroid: {features.spectral_centroid_hz} Hz")
    print(f"    Energy: {features.energy_db} dB SPL")
    print(f"    Temporal pattern: {features.temporal_pattern}")
    print(f"    MFCC summary: {features.mfcc_summary}")

    # Deterministic layer-6 primitive
    print("\n[Layer 6] Deterministic Primitive — prim_generate_gain_params()")
    gain = prim_generate_gain_params(AUDIOGRAM, "market scene")
    print(f"  Gain per frequency: {gain['gain_per_frequency']}")
    print(f"  Compression ratio: {gain['compression_ratio']}")
    print(f"  Deterministic: {gain['deterministic']}")

    print("\n" + "=" * 70)
    print("Everything above runs without an LLM.")
    print("Layers 4-7 require an LLM API after `dspy.configure(lm=...)`.")
    print("GEPA optimization is triggered through `compile_with_gepa()`.")
    print("=" * 70)

    return features


def _print_result(result):
    """Print the full L3→L7 reasoning chain."""
    print("\n[Layer 3] Features:")
    print(f"  SNR: {result.features.snr_db} dB")
    print(f"  Energy: {result.features.energy_db} dB SPL")
    print(f"  Active sources: {result.features.n_active_sources}")

    print(f"\n[Layer 4] Perceptual Description (confidence={result.percept.confidence}):")
    print(f"  Noise: {result.percept.noise_description[:200]}")
    print(f"  Speech: {result.percept.speech_description[:200]}")
    print(f"  Environment: {result.percept.environment_description[:200]}")

    print(f"\n[Layer 5] Scene Understanding (confidence={result.scene.confidence}):")
    print(f"  Situation: {result.scene.situation[:300]}")
    print(f"  Challenges: {result.scene.challenges_json[:300]}")

    print("\n[Layer 6] Strategy:")
    print(f"  Beam: azimuth={result.strategy.target_azimuth_deg}, width={result.strategy.beam_width_deg}")
    print(f"  NR: method={result.strategy.nr_method}, aggressiveness={result.strategy.nr_aggressiveness}")
    print(f"  Gain: {result.strategy.gain_per_frequency}")
    print(f"  Compression: {result.strategy.compression_ratio}")

    print("\n[DSP Params]:")
    print(f"  Beam weights: {result.dsp_params.beam_weights}")
    print(f"  Compression: {result.dsp_params.compression_ratio}")
    print(f"  Filter coeffs (first 5): {result.dsp_params.filter_coeffs[:5]}")


def _result_to_trace(result, label):
    """Serialize a harness result into an MLflow-friendly dictionary."""
    return {
        "scenario": label,
        "L3_features": {
            "snr_db": result.features.snr_db,
            "energy_db": result.features.energy_db,
            "n_active_sources": result.features.n_active_sources,
        },
        "L4_percept": {
            "noise": str(result.percept.noise_description)[:300],
            "speech": str(result.percept.speech_description)[:300],
            "environment": str(result.percept.environment_description)[:300],
            "confidence": str(result.percept.confidence),
        },
        "L5_scene": {
            "situation": str(result.scene.situation)[:500],
            "confidence": str(result.scene.confidence),
        },
        "L6_strategy": {
            "beam_azimuth": str(result.strategy.target_azimuth_deg),
            "beam_width": str(result.strategy.beam_width_deg),
            "nr_aggressiveness": str(result.strategy.nr_aggressiveness),
            "compression": str(result.strategy.compression_ratio),
        },
        "dsp_params": {
            "beam_weights": result.dsp_params.beam_weights[:5],
            "compression": result.dsp_params.compression_ratio,
        },
        "preferences": getattr(result, "current_preferences", None),
    }


def demo_full_pipeline():
    """
    Run the full seven-layer pipeline (requires an OpenAI API key).

    Scenario: wet market vendor conversation -> user says "too muffled" ->
    preferences update -> strategy adapts. Results are logged to MLflow.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment. Skipping full pipeline.")
        return

    import dspy
    import mlflow

    print("\n" + "=" * 70)
    print("DEMO: full seven-layer pipeline including LLM layers")
    print("Scenario: Mr. Li, 12:15 PM, wet market fish stall")
    print("=" * 70)

    # Configure the LLMs.
    task_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)
    strong_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)
    dspy.configure(lm=task_lm)

    # Load a real WAV file if available.
    wav_path = SCENARIO_DIR / "wet_market_vendor.wav"
    signal = None
    if wav_path.exists():
        signal = prim_load_audio(str(wav_path))
        if isinstance(signal, tuple):
            signal = signal[0]
        print(f"  Audio: {wav_path.name} ({signal.duration_ms:.0f}ms, {signal.n_channels}ch)")

    harness = AcousticSemanticHarness(
        fast_lm=task_lm,
        strong_lm=strong_lm,
        enable_multimodal=True,
    )

    mlflow.set_experiment("asir-demo")
    with mlflow.start_run(run_name="demo_market_scenario"):
        # First pass: no explicit user action.
        print("\n>>> Scenario 1: wet market conversation with a vendor (automatic processing)")
        result = harness(
            raw_signal=signal,
            user_action="none",
            user_profile=USER_PROFILE,
            audiogram_json=AUDIOGRAM,
        )
        _print_result(result)

        # Second pass: user says "too muffled".
        print("\n" + "-" * 70)
        print('>>> Scenario 2: same scene, user says "too muffled" -> update preferences')
        result2 = harness(
            raw_signal=signal,
            user_action="too muffled",
            user_profile=USER_PROFILE,
            audiogram_json=AUDIOGRAM,
        )
        _print_result(result2)

        print("\n[Layer 7] Preferences after feedback:")
        print(f"  {json.dumps(result2.current_preferences, ensure_ascii=False, indent=2)}")

        print("\n[Scene History]:")
        for i, s in enumerate(result2.scene_history):
            print(f"  {i + 1}. {s[:100]}")

        mlflow.log_dict({
            "wet_market_vendor": _result_to_trace(result, "wet_market_vendor"),
            "market_too_muffled": _result_to_trace(result2, "market_too_muffled"),
        }, "demo_trace.json")
        mlflow.log_metric("scene1_nr", float(result.strategy.nr_aggressiveness))
        mlflow.log_metric("scene2_nr", float(result2.strategy.nr_aggressiveness))

    print("\n" + "=" * 70)
    print("Full seven-layer pipeline completed.")
    print("Reasoning traces were logged to MLflow (experiment: asir-demo).")
    print("Inspect them via: mlflow ui -> asir-demo -> demo_market_scenario -> Artifacts -> demo_trace.json")
    print("=" * 70)


if __name__ == "__main__":
    demo_deterministic_layers()
    print("\n" + ARCHITECTURE_MAP)

    if "--l1-l3" not in sys.argv:
        # By default, run the full pipeline with semantic reasoning and feedback.
        demo_full_pipeline()

    if "--gepa" in sys.argv:
        print("\n\n" + "=" * 70)
        print("Starting GEPA optimization...")
        print("=" * 70)
        try:
            compile_with_gepa()
        except Exception as e:
            print(f"\nGEPA failed: {e}")
            import traceback

            traceback.print_exc()
