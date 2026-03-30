from pydantic import BaseModel, Field


class AcousticChallenge(BaseModel):
    challenge: str = Field(desc="Challenge description")
    severity: str = Field(desc="Severity: mild / moderate / severe")
    physical_cause: str = Field(desc="Physical cause")


class SceneUnderstanding(BaseModel):
    """Scene understanding that requires cross-dimensional reasoning."""
    situation: str = Field(desc="Scene description")
    acoustic_challenges: list[AcousticChallenge] = Field(desc="List of acoustic challenges")
    preservation_notes: list[str] = Field(desc="Environmental cues that should be preserved")
    confidence: float = Field(desc="Scene-understanding confidence [0,1]")
