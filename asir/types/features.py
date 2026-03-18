from typing import Optional
from pydantic import BaseModel, Field


class AcousticFeatures(BaseModel):
    """聲學特徵向量 — 由確定性演算法提取"""
    mfcc_summary: str = Field(desc="MFCC 特徵的文字摘要（供 LLM 理解）")
    snr_db: float = Field(desc="估計信噪比 (dB)")
    rt60_s: float = Field(desc="估計混響時間 (秒)")
    pitch_hz: Optional[float] = Field(desc="估計基頻 (Hz)，無語音時為 None")
    n_active_sources: int = Field(desc="估計活躍聲源數量")
    spectral_centroid_hz: float = Field(desc="頻譜重心 (Hz)")
    energy_db: float = Field(desc="信號能量 (dB SPL)")
    temporal_pattern: str = Field(desc="時域模式描述：stationary/impulsive/modulated")
