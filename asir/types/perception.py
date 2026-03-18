from pydantic import BaseModel, Field


class NoiseSource(BaseModel):
    type: str = Field(desc="噪音類型，如 '多人交談聲' '金屬碰撞'")
    direction: str = Field(desc="方向描述，如 '四周環繞' '右前方'")
    temporal: str = Field(desc="時間模式：持續/間歇/偶發")
    severity: str = Field(desc="感知嚴重程度：mild/moderate/severe")


class SpeechInfo(BaseModel):
    n_speakers: int = Field(desc="估計說話者數量")
    target_speaker_direction: str = Field(desc="目標說話者方向")
    target_speaker_distance: str = Field(desc="估計距離：近/中/遠")
    intelligibility: str = Field(desc="可懂度：clear/slightly_masked/heavily_masked/inaudible")


class PerceptualDescription(BaseModel):
    """感知描述 — 語義邊界以上的第一個型別，帶 confidence"""
    noise_sources: list[NoiseSource] = Field(desc="辨識到的噪音源列表")
    speech: SpeechInfo = Field(desc="語音相關資訊")
    environment_type: str = Field(desc="環境類型，如 '室內菜市場'")
    acoustic_character: str = Field(desc="聲學特性描述")
    confidence: float = Field(desc="整體信心度 [0,1]")
