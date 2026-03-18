from pydantic import BaseModel, Field


class BeamformingParams(BaseModel):
    target_azimuth_deg: float = Field(desc="目標方位角（度）")
    beam_width_deg: float = Field(desc="波束寬度（度）")
    null_directions: list[float] = Field(desc="零點方向列表（度）")


class NoiseReductionParams(BaseModel):
    method: str = Field(desc="方法：spectral_subtraction/wiener/dnn_masking")
    aggressiveness: float = Field(desc="攻擊性 [0,1]")
    preserve_bands: list[str] = Field(desc="需保留的頻段描述")


class ProcessingStrategy(BaseModel):
    """處理策略 — 語義空間到物理空間的翻譯源"""
    beamforming: BeamformingParams
    noise_reduction: NoiseReductionParams
    gain_adjustment_db: float = Field(desc="整體增益調整 (dB)")
    compression_ratio: float = Field(desc="動態壓縮比")
    direct_to_processed_ratio: float = Field(desc="原始/處理後混合比 [0,1]")
    reasoning: str = Field(desc="策略決策推理過程")
    confidence: float = Field(desc="策略信心度 [0,1]")
