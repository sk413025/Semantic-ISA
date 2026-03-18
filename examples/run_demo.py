"""
ASIR Demo — Run the full 7-layer pipeline.

Usage:
    python -m examples.run_demo           # deterministic layers only
    python -m examples.run_demo --full    # full pipeline (needs API key)
    python -m examples.run_demo --gepa    # full + GEPA optimization
"""
import os
import sys

# Add project root to path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asir.primitives import (
    prim_sample_audio, prim_fft, prim_estimate_noise_psd,
    prim_beamform, comp_spectral_subtract, comp_extract_full_features,
    prim_generate_gain_params,
)
from asir.harness import AcousticSemanticHarness
from asir.architecture import ARCHITECTURE_MAP
from asir.gepa.compiler import compile_with_gepa


def demo_deterministic_layers():
    """
    展示前三層（確定性）的完整執行——不需要 LLM API
    這些層的輸出就是第四層 LLM Primitive 的輸入
    """
    print("=" * 70)
    print("DEMO: 確定性層（第一～三層）完整執行")
    print("場景：李伯伯，12:15，菜市場魚攤")
    print("=" * 70)

    # 第一層
    print("\n[Layer 1] Physical Sensing — prim_sample_audio()")
    signal = prim_sample_audio(duration_ms=32.0, n_channels=2)
    print(f"  Channels: {signal.n_channels}")
    print(f"  Sample rate: {signal.sample_rate} Hz")
    print(f"  Duration: {signal.duration_ms} ms")
    print(f"  Samples per channel: {len(signal.samples[0])}")

    # 第二層
    print("\n[Layer 2] Signal Processing")
    spectrum = prim_fft(signal)
    print(f"  [PRIM] FFT: {spectrum['freq_bins']} frequency bins")

    noise_psd = prim_estimate_noise_psd(signal)
    print(f"  [PRIM] Noise PSD: estimated {sum(1 for x in noise_psd if x > 0)} active bins")

    beamformed = prim_beamform(signal, target_azimuth_deg=0.0)
    print(f"  [PRIM] Beamform(0°): {len(beamformed)} samples")

    cleaned = comp_spectral_subtract(signal, noise_psd, alpha=1.0)
    print(f"  [COMP] Spectral subtract: {len(cleaned)} samples")

    # 第三層
    print("\n[Layer 3] Acoustic Features")
    features = comp_extract_full_features(signal)
    print(f"  [COMP] Full features extracted:")
    print(f"    SNR: {features.snr_db} dB")
    print(f"    RT60: {features.rt60_s} s")
    print(f"    Active sources: {features.n_active_sources}")
    print(f"    Spectral centroid: {features.spectral_centroid_hz} Hz")
    print(f"    Energy: {features.energy_db} dB SPL")
    print(f"    Temporal pattern: {features.temporal_pattern}")
    print(f"    MFCC summary: {features.mfcc_summary}")

    # 第六層確定性 Primitive
    print("\n[Layer 6] Deterministic Primitive — prim_generate_gain_params()")
    audiogram = '{"250":30,"500":35,"1000":40,"2000":50,"4000":60}'
    gain = prim_generate_gain_params(audiogram, "market scene")
    print(f"  Gain per frequency: {gain['gain_per_frequency']}")
    print(f"  Compression ratio: {gain['compression_ratio']}")
    print(f"  Deterministic: {gain['deterministic']}")

    print("\n" + "=" * 70)
    print("以上是不需要 LLM 的部分。")
    print("第四～七層需要 LLM API (dspy.configure(lm=...) 後執行)")
    print("GEPA 優化需要呼叫 compile_with_gepa()")
    print("=" * 70)

    return features


def demo_full_pipeline():
    """
    展示完整七層管線（需要 OpenAI API key）
    """
    import json

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment. Skipping full pipeline.")
        return

    print("\n" + "=" * 70)
    print("DEMO: 完整七層管線（含 LLM 層）")
    print("場景：李伯伯，12:15，菜市場魚攤")
    print("=" * 70)

    # 配置 LLM
    task_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)
    strong_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)  # 用同一個模型降低成本
    dspy.configure(lm=task_lm)

    # 創建 Harness
    harness = AcousticSemanticHarness(
        fast_lm=task_lm,
        strong_lm=strong_lm
    )

    # 執行完整管線
    print("\n>>> 執行第一次：無使用者動作（自動處理）")
    result = harness(
        user_action="none",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲"
    )

    print(f"\n[Layer 3] Features:")
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

    print(f"\n[Layer 6] Strategy:")
    print(f"  Beam: azimuth={result.strategy.target_azimuth_deg}°, width={result.strategy.beam_width_deg}°")
    print(f"  NR: method={result.strategy.nr_method}, aggressiveness={result.strategy.nr_aggressiveness}")
    print(f"  Gain: {result.strategy.gain_per_frequency}")
    print(f"  Compression: {result.strategy.compression_ratio}")

    print(f"\n[Layer 2] DSP Params (translated from strategy):")
    print(f"  Beam weights: {result.dsp_params.beam_weights}")
    print(f"  Compression: {result.dsp_params.compression_ratio}")
    print(f"  Filter coeffs (first 5): {result.dsp_params.filter_coeffs[:5]}")

    # 第二次：使用者按「不滿意」
    print("\n" + "-" * 70)
    print(">>> 執行第二次：使用者按了「不滿意」按鈕")
    result2 = harness(
        user_action="button_press:dissatisfied",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲"
    )

    print(f"\n[Layer 7] Intent parsed from button press:")
    print(f"  Updated preferences: {json.dumps(result2.current_preferences, ensure_ascii=False, indent=2)}")

    print(f"\n[Layer 6] Updated Strategy:")
    print(f"  Beam: azimuth={result2.strategy.target_azimuth_deg}°, width={result2.strategy.beam_width_deg}°")
    print(f"  NR: method={result2.strategy.nr_method}, aggressiveness={result2.strategy.nr_aggressiveness}")

    print(f"\n[Scene History]:")
    for i, s in enumerate(result2.scene_history):
        print(f"  {i+1}. {s[:100]}")

    print("\n" + "=" * 70)
    print("完整七層管線執行完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo_deterministic_layers()
    print("\n" + ARCHITECTURE_MAP)

    if "--full" in sys.argv or "--gepa" in sys.argv:
        demo_full_pipeline()

    if "--gepa" in sys.argv:
        print("\n\n" + "=" * 70)
        print("接下來執行 GEPA 優化...")
        print("=" * 70)
        try:
            optimized = compile_with_gepa()
        except Exception as e:
            print(f"\nGEPA 執行失敗: {e}")
            import traceback
            traceback.print_exc()
