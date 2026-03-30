from pydantic import BaseModel, Field


class DSPParameterSet(BaseModel):
    """Deterministic DSP parameters translated from the strategy layer."""
    filter_coeffs: list[float] = Field(desc="FIR filter coefficients")
    beam_weights: list[float] = Field(desc="Beamforming weights")
    noise_mask: list[float] = Field(desc="Frequency-domain noise mask")
    compression_ratio: float = Field(desc="Dynamic compression ratio")
    attack_ms: float = Field(desc="Compressor attack time (ms)")
    release_ms: float = Field(desc="Compressor release time (ms)")
