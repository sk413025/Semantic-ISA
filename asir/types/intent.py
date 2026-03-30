from pydantic import BaseModel, Field


class UserIntent(BaseModel):
    primary_goal: str = Field(desc="Primary goal")
    secondary_goals: list[str] = Field(desc="Secondary goals")
    constraints: list[str] = Field(desc="User constraints")


class UserPreferences(BaseModel):
    noise_tolerance: str = Field(desc="Noise tolerance: low / medium / high")
    processing_preference: str = Field(desc="Processing preference: natural / balanced / maximal_clarity")
    environment_awareness: str = Field(desc="Need for environmental awareness: minimal / moderate / full")
    known_situations: list[str] = Field(desc="Preference strategies for known situations")
