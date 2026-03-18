from typing import Optional
import dspy
from asir.types import AcousticFeatures
from asir.primitives.perception import DescribeNoiseSig, DescribeSpeechSig, DescribeEnvironmentSig
from asir.routing.perceptual import PerceptAggregateRoutingSig


class FullPerceptualDescription(dspy.Module):
    """
    [COMP] 第四層：完整感知描述
    = describe_noise + describe_speech + describe_environment
    + ★ aggregate_router（Method A：可學習的聚合決策）

    改造前：confidence = min(三者)（hardcoded 保守策略）
    改造後：confidence 由 aggregate_router 根據場景動態決定
    """
    def __init__(self):
        super().__init__()
        # 三個 Primitive，各自獨立可優化（prompt 不動）
        self.describe_noise = dspy.ChainOfThought(DescribeNoiseSig)
        self.describe_speech = dspy.ChainOfThought(DescribeSpeechSig)
        self.describe_env = dspy.ChainOfThought(DescribeEnvironmentSig)
        # ★ 新增：Routing Predictor — GEPA 可優化
        self.aggregate_router = dspy.ChainOfThought(PerceptAggregateRoutingSig)

    def forward(self, acoustic_features: AcousticFeatures,
                user_context: str,
                audio_clip: Optional[dspy.Audio] = None,
                spectrogram: Optional[dspy.Image] = None,
                ) -> dspy.Prediction:
        features_str = (
            f"SNR: {acoustic_features.snr_db} dB, "
            f"RT60: {acoustic_features.rt60_s} s, "
            f"Active sources: {acoustic_features.n_active_sources}, "
            f"Spectral centroid: {acoustic_features.spectral_centroid_hz} Hz, "
            f"Energy: {acoustic_features.energy_db} dB SPL, "
            f"Temporal pattern: {acoustic_features.temporal_pattern}, "
            f"MFCC: {acoustic_features.mfcc_summary}"
        )

        # === Phase 1: 三個 PRIM 照跑 ===
        # ★ Phase 3: 根據可用的多模態資料動態構建 kwargs
        noise_kwargs = {
            "acoustic_features": features_str,
            "user_context": user_context,
        }
        speech_kwargs = {
            "acoustic_features": features_str,
        }
        env_kwargs = {
            "acoustic_features": features_str,
        }
        # ★ 多模態注入：只在有資料時傳入，否則 LLM 只看文字
        if audio_clip is not None:
            noise_kwargs["audio_clip"] = audio_clip
            speech_kwargs["audio_clip"] = audio_clip
            env_kwargs["audio_clip"] = audio_clip
        if spectrogram is not None:
            noise_kwargs["spectrogram"] = spectrogram
            speech_kwargs["spectrogram"] = spectrogram
            env_kwargs["spectrogram"] = spectrogram

        noise_result = self.describe_noise(**noise_kwargs)
        speech_result = self.describe_speech(**speech_kwargs)
        env_result = self.describe_env(**env_kwargs)

        # === Phase 2: Routing Predictor 決定怎麼聚合 ===
        routing = self.aggregate_router(
            noise_summary=(
                f"sources={str(noise_result.noise_sources_json)[:200]}, "
                f"confidence={noise_result.confidence}"
            ),
            speech_summary=(
                f"speakers={speech_result.n_speakers}, "
                f"intelligibility={speech_result.intelligibility}, "
                f"confidence={speech_result.confidence}"
            ),
            env_summary=(
                f"type={env_result.environment_type}, "
                f"character={env_result.acoustic_character}, "
                f"confidence={env_result.confidence}"
            ),
            snr_db=acoustic_features.snr_db,
            n_sources=acoustic_features.n_active_sources
        )

        # === Phase 3: 用 router 的判斷組合輸出 ===
        overall_confidence = float(routing.overall_confidence)

        return dspy.Prediction(
            noise_description=noise_result.noise_sources_json,
            speech_description=(
                f"Speakers: {speech_result.n_speakers}, "
                f"Target: {speech_result.target_direction} at {speech_result.target_distance}, "
                f"Intelligibility: {speech_result.intelligibility}"
            ),
            environment_description=(
                f"Type: {env_result.environment_type}, "
                f"Character: {env_result.acoustic_character}"
            ),
            confidence=overall_confidence,
            # ★ 暴露權重，讓下游層和 metric 能看到
            percept_weights={
                "noise": float(routing.noise_weight),
                "speech": float(routing.speech_weight),
                "env": float(routing.env_weight)
            },
            routing_reasoning=routing.routing_reasoning
        )
