from typing import Optional
from pydantic import BaseModel, Field


class AcousticFeatures(BaseModel):
    """Acoustic feature vector extracted by deterministic algorithms."""
    mfcc_summary: str = Field(desc="Text summary of MFCC-like features for the LLM")
    snr_db: float = Field(desc="Estimated signal-to-noise ratio (dB)")
    rt60_s: float = Field(desc="Estimated reverberation time (seconds)")
    pitch_hz: Optional[float] = Field(desc="Estimated fundamental frequency (Hz); None when speech is absent")
    n_active_sources: int = Field(desc="Estimated number of active sources")
    spectral_centroid_hz: float = Field(desc="Spectral centroid (Hz)")
    energy_db: float = Field(desc="Signal energy (dB SPL)")
    temporal_pattern: str = Field(desc="Temporal pattern descriptor: stationary / impulsive / modulated")
