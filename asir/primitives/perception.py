from typing import Optional

import dspy


# --- [PRIM] describe_noise: translate numeric features into perceptual semantics ---
class DescribeNoiseSig(dspy.Signature):
    """
    [PRIM] Layer 4: noise perceptual description (multimodal v0.3).
    BACKEND: LLM (multimodal-aware)
    RELIABILITY: source_identification_accuracy >= 0.80

    Translate acoustic features into human-readable noise descriptions. This is
    an irreducible semantic operation because noise type, direction, and
    temporal structure are tightly coupled.

    Multimodal inputs:
    - audio_clip lets audio-capable models directly listen to the waveform.
    - spectrogram lets vision-capable models inspect time-frequency structure.
    - both are optional and fall back to text-only mode when unavailable.
    """

    acoustic_features: str = dspy.InputField(
        desc="Text description of acoustic features, including SNR, spectral properties, and energy"
    )
    user_context: str = dspy.InputField(
        desc="User context such as age, hearing profile, and current activity"
    )
    audio_clip: Optional[dspy.Audio] = dspy.InputField(
        desc="Raw audio clip. Audio-capable models can listen directly; otherwise this is None.",
        default=None,
    )
    spectrogram: Optional[dspy.Image] = dspy.InputField(
        desc="Mel spectrogram image so vision models can inspect audio structure; otherwise None.",
        default=None,
    )

    noise_sources_json: str = dspy.OutputField(
        desc="JSON list of noise sources, each with type / direction / temporal / severity. "
        "Severity must reflect SNR: <5 dB -> high or severe; 5-15 dB -> moderate; >15 dB -> low or minimal."
    )
    confidence: float = dspy.OutputField(desc="Confidence of the noise description [0,1]")


# --- [PRIM] describe_speech: speech perceptual description ---
class DescribeSpeechSig(dspy.Signature):
    """
    [PRIM] Layer 4: speech perceptual description (multimodal v0.3).
    BACKEND: LLM (multimodal-aware)
    RELIABILITY: n_speakers_accuracy >= 0.85 (for 1-4 speakers)
    FAILURE_MODE: >4 speakers -> n_speakers = -1 (meaning "many")

    Speech analysis benefits strongly from multimodal input because direct audio
    can improve speaker-count, direction, and intelligibility judgments.
    """

    acoustic_features: str = dspy.InputField(
        desc="Acoustic features with emphasis on pitch, modulation patterns, and energy envelope"
    )
    audio_clip: Optional[dspy.Audio] = dspy.InputField(
        desc="Raw audio clip. Listening directly can improve speaker-count and intelligibility estimates.",
        default=None,
    )
    spectrogram: Optional[dspy.Image] = dspy.InputField(
        desc="Spectrogram image that reveals formant structure and temporal speech patterns.",
        default=None,
    )

    n_speakers: int = dspy.OutputField(desc="Estimated number of speakers; use -1 for more than four")
    target_direction: str = dspy.OutputField(desc="Description of the target speaker direction")
    target_distance: str = dspy.OutputField(desc="Estimated distance: near / medium / far")
    intelligibility: str = dspy.OutputField(
        desc="Intelligibility: clear / slightly_masked / heavily_masked / inaudible"
    )
    confidence: float = dspy.OutputField(desc="Confidence [0,1]")


# --- [PRIM] describe_environment: environment perceptual description ---
class DescribeEnvironmentSig(dspy.Signature):
    """
    [PRIM] Layer 4: environment perceptual description (multimodal v0.3).
    BACKEND: LLM (multimodal-aware)
    RELIABILITY: environment_type_accuracy >= 0.75
    FAILURE_MODE: novel_environment -> confidence < 0.5

    Environment recognition especially benefits from spectrograms because
    reverberation and background-noise texture are visually salient.
    """

    acoustic_features: str = dspy.InputField(
        desc="Acoustic features, especially reverberation, spectral distribution, and temporal pattern"
    )
    audio_clip: Optional[dspy.Audio] = dspy.InputField(
        desc="Raw audio clip for direct listening to the environment soundscape",
        default=None,
    )
    spectrogram: Optional[dspy.Image] = dspy.InputField(
        desc="Spectrogram image, useful for reverberation cues and background texture",
        default=None,
    )

    environment_type: str = dspy.OutputField(desc="Environment-type description")
    acoustic_character: str = dspy.OutputField(desc="Acoustic-character description")
    confidence: float = dspy.OutputField(desc="Environment-classification confidence [0,1]")
