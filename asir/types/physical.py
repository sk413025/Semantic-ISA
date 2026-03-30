from pydantic import BaseModel, Field


class RawSignal(BaseModel):
    """Raw multi-channel audio signal as a deterministic type."""
    samples: list[list[float]] = Field(desc="[channels][samples] PCM values")
    sample_rate: int = Field(default=16000, desc="Sampling rate in Hz")
    n_channels: int = Field(default=2, desc="Number of microphones")
    duration_ms: float = Field(desc="Signal duration in milliseconds")
