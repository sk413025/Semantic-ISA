from pydantic import BaseModel, Field


class AcousticChallenge(BaseModel):
    challenge: str = Field(desc="挑戰描述")
    severity: str = Field(desc="嚴重程度: mild/moderate/severe")
    physical_cause: str = Field(desc="物理原因")


class SceneUnderstanding(BaseModel):
    """場景理解 — 需要跨維度推理"""
    situation: str = Field(desc="場景敘述")
    acoustic_challenges: list[AcousticChallenge] = Field(desc="聲學挑戰列表")
    preservation_notes: list[str] = Field(desc="需要保留的環境聲線索")
    confidence: float = Field(desc="場景理解信心度 [0,1]")
