from pydantic import BaseModel, Field


class UserIntent(BaseModel):
    primary_goal: str = Field(desc="主要目標")
    secondary_goals: list[str] = Field(desc="次要目標")
    constraints: list[str] = Field(desc="使用者約束")


class UserPreferences(BaseModel):
    noise_tolerance: str = Field(desc="噪音容忍度: low/medium/high")
    processing_preference: str = Field(desc="處理偏好: natural/balanced/maximal_clarity")
    environment_awareness: str = Field(desc="環境感知需求: minimal/moderate/full")
    known_situations: list[str] = Field(desc="已知場景的偏好策略")
