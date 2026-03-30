from typing import Optional

import dspy

from asir.primitives.perception import (
    DescribeEnvironmentSig,
    DescribeNoiseSig,
    DescribeSpeechSig,
)
from asir.routing.perceptual import PerceptAggregateRoutingSig
from asir.types import AcousticFeatures


class FullPerceptualDescription(dspy.Module):
    """
    [COMP] Layer 4: full perceptual description.
    = describe_noise + describe_speech + describe_environment
    + aggregate_router (Method A: learnable aggregation policy)

    Before the refactor, confidence was the hardcoded conservative minimum
    across the three predictors. Now the aggregate router decides confidence
    dynamically based on the scene.
    """

    def __init__(self):
        super().__init__()
        # Three independent PRIMs. Each remains separately optimizable.
        self.describe_noise = dspy.ChainOfThought(DescribeNoiseSig)
        self.describe_speech = dspy.ChainOfThought(DescribeSpeechSig)
        self.describe_env = dspy.ChainOfThought(DescribeEnvironmentSig)
        # Learnable routing predictor for GEPA.
        self.aggregate_router = dspy.ChainOfThought(PerceptAggregateRoutingSig)

    def forward(
        self,
        acoustic_features: AcousticFeatures,
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

        # Phase 1: run all three PRIMs.
        # Phase 3: build multimodal kwargs dynamically based on available inputs.
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

        # Inject multimodal inputs only when they are available.
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

        # Phase 2: let the routing predictor decide how to aggregate the outputs.
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
            n_sources=acoustic_features.n_active_sources,
        )

        # Phase 3: assemble the output using the router's judgment.
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
            # Expose weights so downstream modules and metrics can inspect them.
            percept_weights={
                "noise": float(routing.noise_weight),
                "speech": float(routing.speech_weight),
                "env": float(routing.env_weight),
            },
            routing_reasoning=routing.routing_reasoning,
        )
