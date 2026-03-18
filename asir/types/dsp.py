from pydantic import BaseModel, Field


class DSPParameterSet(BaseModel):
    """DSP 處理參數 — 由策略層翻譯而來的確定性參數"""
    filter_coeffs: list[float] = Field(desc="FIR 濾波器係數")
    beam_weights: list[float] = Field(desc="波束成形權重")
    noise_mask: list[float] = Field(desc="頻域噪音遮罩")
    compression_ratio: float = Field(desc="動態壓縮比")
    attack_ms: float = Field(desc="壓縮器 attack time (ms)")
    release_ms: float = Field(desc="壓縮器 release time (ms)")
