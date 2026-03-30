from pydantic import BaseModel, Field


class BeamformingParams(BaseModel):
    target_azimuth_deg: float = Field(desc="Target azimuth in degrees")
    beam_width_deg: float = Field(desc="Beam width in degrees")
    null_directions: list[float] = Field(desc="List of null directions in degrees")


class NoiseReductionParams(BaseModel):
    method: str = Field(desc="Method: spectral_subtraction / wiener / dnn_masking")
    aggressiveness: float = Field(desc="Aggressiveness [0,1]")
    preserve_bands: list[str] = Field(desc="Descriptions of frequency bands that should be preserved")


class ProcessingStrategy(BaseModel):
    """Processing strategy translating semantic space into physical DSP actions."""
    beamforming: BeamformingParams
    noise_reduction: NoiseReductionParams
    gain_adjustment_db: float = Field(desc="Overall gain adjustment (dB)")
    compression_ratio: float = Field(desc="Dynamic compression ratio")
    direct_to_processed_ratio: float = Field(desc="Raw-to-processed mix ratio [0,1]")
    reasoning: str = Field(desc="Reasoning behind the strategy decision")
    confidence: float = Field(desc="Strategy confidence [0,1]")
