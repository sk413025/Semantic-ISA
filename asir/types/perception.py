from pydantic import BaseModel, Field


class NoiseSource(BaseModel):
    type: str = Field(desc="Noise type, e.g. 'multiple speakers' or 'metallic impacts'")
    direction: str = Field(desc="Direction description, e.g. 'surrounding' or 'front-right'")
    temporal: str = Field(desc="Temporal pattern: continuous / intermittent / occasional")
    severity: str = Field(desc="Perceived severity: mild / moderate / severe")


class SpeechInfo(BaseModel):
    n_speakers: int = Field(desc="Estimated number of speakers")
    target_speaker_direction: str = Field(desc="Direction of the target speaker")
    target_speaker_distance: str = Field(desc="Estimated distance: near / mid / far")
    intelligibility: str = Field(desc="Intelligibility: clear / slightly_masked / heavily_masked / inaudible")


class PerceptualDescription(BaseModel):
    """Perceptual description above the semantic boundary, with confidence."""
    noise_sources: list[NoiseSource] = Field(desc="List of identified noise sources")
    speech: SpeechInfo = Field(desc="Speech-related information")
    environment_type: str = Field(desc="Environment type, e.g. 'indoor wet market'")
    acoustic_character: str = Field(desc="Acoustic character description")
    confidence: float = Field(desc="Overall confidence [0,1]")
