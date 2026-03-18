from pydantic import BaseModel, Field


class RawSignal(BaseModel):
    """多聲道原始音訊信號 — 確定性型別，無不確定性"""
    samples: list[list[float]] = Field(desc="[channels][samples] PCM 數值")
    sample_rate: int = Field(default=16000, desc="取樣率 Hz")
    n_channels: int = Field(default=2, desc="麥克風數量")
    duration_ms: float = Field(desc="信號持續時間（毫秒）")
