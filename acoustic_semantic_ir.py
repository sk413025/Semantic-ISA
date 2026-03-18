"""
═══════════════════════════════════════════════════════════════════════════════
Acoustic Semantic IR (ASIR) — Full 7-Layer Implementation
Using DSPy + GEPA for Hearing Aid Scenario

場景：72 歲的李伯伯戴著智慧助聽器，12:15 在菜市場買菜
核心時刻：攤販在說價錢，四周多人交談，偶有金屬碰撞聲

每一層都標註：
  [PRIM] = Primitive（不可再分解的原子操作）
  [COMP] = Composite（由 Primitive 組合而成）
  BACKEND: deterministic | LLM | LLM+physics
═══════════════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
from dotenv import load_dotenv
# Try multiple .env locations
for _p in [
    Path(__file__).parent / '.env',
    Path(__file__).parent / 'Semantic-ISA' / '.env',
    Path(os.environ.get('USERPROFILE', '')) / 'OneDrive' / 'Semantic-ISA' / '.env',
]:
    if _p.exists():
        load_dotenv(_p)
        break
else:
    load_dotenv()  # fallback: cwd

import dspy
import numpy as np
from enum import Enum
from typing import Optional
from contextlib import nullcontext
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# 型別系統 (Semantic Type System)
# 每個語義型別都帶有 confidence 和 alternatives —— 這是 LLVM 型別系統沒有的
# ═══════════════════════════════════════════════════════════════════════════

# --- 第一層：物理感測型別 (完全確定性) ---
class RawSignal(BaseModel):
    """多聲道原始音訊信號 — 確定性型別，無不確定性"""
    samples: list[list[float]] = Field(desc="[channels][samples] PCM 數值")
    sample_rate: int = Field(default=16000, desc="取樣率 Hz")
    n_channels: int = Field(default=2, desc="麥克風數量")
    duration_ms: float = Field(desc="信號持續時間（毫秒）")


# --- 第二層：信號處理型別 (完全確定性) ---
class DSPParameterSet(BaseModel):
    """DSP 處理參數 — 由策略層翻譯而來的確定性參數"""
    filter_coeffs: list[float] = Field(desc="FIR 濾波器係數")
    beam_weights: list[float] = Field(desc="波束成形權重")
    noise_mask: list[float] = Field(desc="頻域噪音遮罩")
    compression_ratio: float = Field(desc="動態壓縮比")
    attack_ms: float = Field(desc="壓縮器 attack time (ms)")
    release_ms: float = Field(desc="壓縮器 release time (ms)")


# --- 第三層：聲學特徵型別 (確定性) ---
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


# ═══════════ SEMANTIC BOUNDARY（Shannon 切割修復點）═══════════

# --- 第四層：感知描述型別 (統計性 — 帶 confidence) ---
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


# --- 第五層：場景理解型別 ---
class AcousticChallenge(BaseModel):
    challenge: str = Field(desc="挑戰描述")
    severity: str = Field(desc="嚴重程度: mild/moderate/severe")
    physical_cause: str = Field(desc="物理原因")

class SceneUnderstanding(BaseModel):
    """場景理解 — 需要跨維度推理"""
    situation: str = Field(desc="場景敘述")
    acoustic_challenges: list[AcousticChallenge] = Field(desc="聲學挑戰列表")
    preservation_notes: list[str] = Field(desc="需要保留的環境聲線索")
    confidence: float = Field(desc="場景理解信心度 [0,1]")


# --- 第六層：策略型別 ---
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


# --- 第七層：意圖與偏好型別 ---
class UserIntent(BaseModel):
    primary_goal: str = Field(desc="主要目標")
    secondary_goals: list[str] = Field(desc="次要目標")
    constraints: list[str] = Field(desc="使用者約束")

class UserPreferences(BaseModel):
    noise_tolerance: str = Field(desc="噪音容忍度: low/medium/high")
    processing_preference: str = Field(desc="處理偏好: natural/balanced/maximal_clarity")
    environment_awareness: str = Field(desc="環境感知需求: minimal/moderate/full")
    known_situations: list[str] = Field(desc="已知場景的偏好策略")


# ═══════════════════════════════════════════════════════════════════════════
# 第一～三層：確定性 Primitive（純 Python 函式，不用 LLM）
# 這些是傳統 DSP ISA 的實現——GEPA 不優化它們
# ═══════════════════════════════════════════════════════════════════════════

def prim_sample_audio(duration_ms: float = 32.0, n_channels: int = 2,
                      sample_rate: int = 16000) -> RawSignal:
    """
    [PRIM] 第一層：物理感測
    BACKEND: deterministic (硬體)
    可靠性: 100%
    類比: CPU 的 LOAD 指令
    """
    n_samples = int(sample_rate * duration_ms / 1000)
    # 模擬菜市場音訊：多聲源混合
    t = np.linspace(0, duration_ms / 1000, n_samples)
    # 目標語音（攤販，正前方）
    target_speech = 0.3 * np.sin(2 * np.pi * 200 * t) * np.random.normal(1, 0.1, n_samples)
    # 背景多人交談（環繞）
    babble = 0.4 * np.random.randn(n_samples)
    # 偶發金屬碰撞（右側）
    impulse = np.zeros(n_samples)
    impulse[n_samples // 3] = 2.0  # 單一脈衝
    
    ch0 = (target_speech + babble + impulse * 0.3).tolist()  # 左耳
    ch1 = (target_speech * 0.8 + babble + impulse * 0.7).tolist()  # 右耳
    
    return RawSignal(
        samples=[ch0, ch1],
        sample_rate=sample_rate,
        n_channels=n_channels,
        duration_ms=duration_ms
    )


def prim_fft(signal: RawSignal) -> dict:
    """
    [PRIM] 第二層：FFT
    BACKEND: deterministic
    可靠性: 100%（數值精度內）
    """
    spectrum = np.fft.rfft(signal.samples[0])
    return {
        "magnitude": np.abs(spectrum).tolist(),
        "phase": np.angle(spectrum).tolist(),
        "freq_bins": len(spectrum)
    }


def prim_estimate_noise_psd(signal: RawSignal) -> list[float]:
    """
    [PRIM] 第二層：噪音功率譜密度估計
    BACKEND: deterministic (minimum statistics 方法)
    可靠性: 100%
    """
    spectrum = np.fft.rfft(signal.samples[0])
    # 簡化版 minimum statistics
    psd = np.abs(spectrum) ** 2
    noise_floor = np.percentile(psd, 25)  # 取 25th percentile 作為噪音底線
    return (psd * (psd < noise_floor * 4)).tolist()


def prim_beamform(signal: RawSignal, target_azimuth_deg: float = 0.0) -> list[float]:
    """
    [PRIM] 第二層：固定方向波束成形
    BACKEND: deterministic
    可靠性: 100%
    """
    ch0 = np.array(signal.samples[0])
    ch1 = np.array(signal.samples[1])
    # 簡化版 delay-and-sum beamformer
    delay_samples = int(0.01 * np.sin(np.radians(target_azimuth_deg)) * signal.sample_rate)
    if delay_samples > 0:
        ch1_delayed = np.roll(ch1, delay_samples)
    else:
        ch1_delayed = np.roll(ch1, delay_samples)
    beamformed = (ch0 + ch1_delayed) / 2.0
    return beamformed.tolist()


def comp_spectral_subtract(signal: RawSignal, noise_psd: list[float],
                           alpha: float = 1.0) -> list[float]:
    """
    [COMP] 第二層：頻譜減法
    = fft(signal) |> subtract_magnitude(noise, alpha) |> ifft()
    由三個 primitive 組合，仍完全確定性
    """
    spectrum = np.fft.rfft(signal.samples[0])
    noise = np.array(noise_psd[:len(spectrum)])
    magnitude = np.maximum(np.abs(spectrum) - alpha * np.sqrt(noise), 0)
    cleaned = magnitude * np.exp(1j * np.angle(spectrum))
    return np.fft.irfft(cleaned).tolist()


def prim_extract_mfcc(signal: RawSignal, n_coeffs: int = 13) -> str:
    """
    [PRIM] 第三層：MFCC 特徵提取
    BACKEND: deterministic
    返回文字摘要供 LLM 理解（因為 LLM 不直接理解張量）
    """
    # 簡化計算 — 真實系統用 librosa
    spectrum = np.abs(np.fft.rfft(signal.samples[0]))
    energy = np.sum(spectrum ** 2)
    spectral_centroid = np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + 1e-10)
    spectral_rolloff = np.searchsorted(np.cumsum(spectrum), 0.85 * np.sum(spectrum))
    
    return (f"Energy: {energy:.1f}, Spectral centroid: {spectral_centroid:.0f} bins, "
            f"Rolloff: {spectral_rolloff} bins, Distribution: "
            f"{'broadband' if spectral_centroid > len(spectrum) * 0.3 else 'low-frequency dominant'}")


def prim_estimate_snr(signal: RawSignal) -> float:
    """[PRIM] 第三層：信噪比估計 — deterministic"""
    s = np.array(signal.samples[0])
    signal_power = np.mean(s ** 2)
    # 簡化：用最低 10% 能量的幀作為噪音估計
    frame_size = len(s) // 10
    frames = [s[i:i+frame_size] for i in range(0, len(s) - frame_size, frame_size)]
    noise_power = min(np.mean(f ** 2) for f in frames) if frames else 1e-10
    return float(10 * np.log10(signal_power / (noise_power + 1e-10)))


def prim_estimate_rt60(signal: RawSignal) -> float:
    """[PRIM] 第三層：混響時間估計 — deterministic"""
    # 簡化版：從能量衰減曲線估計
    s = np.array(signal.samples[0])
    energy_curve = np.cumsum(s[::-1] ** 2)[::-1]
    energy_curve = energy_curve / (energy_curve[0] + 1e-10)
    # 找到能量降至 -60dB 的時間
    db_curve = 10 * np.log10(energy_curve + 1e-10)
    idx_60 = np.searchsorted(-db_curve, 60)
    return float(idx_60 / signal.sample_rate) if idx_60 < len(s) else 0.5


def comp_extract_full_features(signal: RawSignal) -> AcousticFeatures:
    """
    [COMP] 第三層：完整特徵提取
    由多個第三層 Primitive 組合
    仍然完全確定性
    """
    mfcc_summary = prim_extract_mfcc(signal)
    snr = prim_estimate_snr(signal)
    rt60 = prim_estimate_rt60(signal)
    
    s = np.array(signal.samples[0])
    energy = float(10 * np.log10(np.mean(s ** 2) + 1e-10) + 94)  # dB SPL
    spectrum = np.abs(np.fft.rfft(s))
    centroid = float(np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + 1e-10))
    centroid_hz = centroid * signal.sample_rate / (2 * len(spectrum))
    
    # 估計活躍聲源數（用頻譜峰值數量近似）
    peaks = np.diff(np.sign(np.diff(spectrum)))
    n_sources = int(np.sum(peaks < 0) / 5)  # 簡化估計
    
    # 時域模式判斷
    frame_energies = [np.mean(s[i:i+160]**2) for i in range(0, len(s)-160, 160)]
    energy_var = np.var(frame_energies) / (np.mean(frame_energies)**2 + 1e-10)
    if energy_var > 1.0:
        temporal = "impulsive"
    elif energy_var > 0.1:
        temporal = "modulated"
    else:
        temporal = "stationary"
    
    return AcousticFeatures(
        mfcc_summary=mfcc_summary,
        snr_db=round(snr, 1),
        rt60_s=round(rt60, 3),
        pitch_hz=round(centroid_hz, 1) if centroid_hz > 80 else None,
        n_active_sources=max(1, min(n_sources, 10)),
        spectral_centroid_hz=round(centroid_hz, 1),
        energy_db=round(energy, 1),
        temporal_pattern=temporal
    )


# ═══════════════════════════════════════════════════════════════════════════
# 第四層：感知描述層 — 進入語義空間
# ★ Shannon 切割的修復點 ★
# 這裡開始用 DSPy Signature + LLM
# ═══════════════════════════════════════════════════════════════════════════

# --- [PRIM] describe_noise: 從數值翻譯成感知語義 ---
class DescribeNoiseSig(dspy.Signature):
    """
    [PRIM] 第四層：噪音感知描述
    BACKEND: LLM
    RELIABILITY: source_identification_accuracy >= 0.80
    
    將聲學特徵翻譯成人類可理解的噪音描述。
    這是一個不可分解的語義原子操作——因為噪音的
    類型、方向、時間模式之間有強耦合。
    
    ★ 這個翻譯在傳統演算法中不可能 ——
      「多人交談聲，四周環繞」不是從 SNR 數字能精確計算出來的。
    """
    acoustic_features: str = dspy.InputField(
        desc="聲學特徵的文字描述，包含 SNR、頻譜特性、能量等"
    )
    user_context: str = dspy.InputField(
        desc="使用者上下文：年齡、聽力狀況、當前活動"
    )
    
    noise_sources_json: str = dspy.OutputField(
        desc="JSON 格式的噪音源列表，每個包含 type/direction/temporal/severity"
    )
    confidence: float = dspy.OutputField(
        desc="噪音描述的信心度 [0,1]"
    )


# --- [PRIM] describe_speech: 語音感知描述 ---
class DescribeSpeechSig(dspy.Signature):
    """
    [PRIM] 第四層：語音感知描述
    BACKEND: LLM
    RELIABILITY: n_speakers_accuracy >= 0.85 (for 1-4 speakers)
    FAILURE_MODE: >4 speakers -> n_speakers = -1 (meaning "many")
    """
    acoustic_features: str = dspy.InputField(
        desc="聲學特徵，特別關注基頻、調變模式、能量包絡"
    )
    
    n_speakers: int = dspy.OutputField(desc="估計說話者數量，>4 時返回 -1")
    target_direction: str = dspy.OutputField(desc="目標說話者方向描述")
    target_distance: str = dspy.OutputField(desc="估計距離: near/medium/far")
    intelligibility: str = dspy.OutputField(
        desc="可懂度: clear/slightly_masked/heavily_masked/inaudible"
    )
    confidence: float = dspy.OutputField(desc="信心度 [0,1]")


# --- [PRIM] describe_environment: 環境感知描述 ---
class DescribeEnvironmentSig(dspy.Signature):
    """
    [PRIM] 第四層：環境感知描述
    BACKEND: LLM
    RELIABILITY: environment_type_accuracy >= 0.75
    FAILURE_MODE: novel_environment -> confidence < 0.5
    """
    acoustic_features: str = dspy.InputField(
        desc="聲學特徵，關注混響、頻譜分布、時域模式"
    )
    
    environment_type: str = dspy.OutputField(desc="環境類型描述")
    acoustic_character: str = dspy.OutputField(desc="聲學特性描述")
    confidence: float = dspy.OutputField(desc="環境判斷信心度 [0,1]")


# ═══════════════════════════════════════════════════════════════════════════
# 第五層：場景理解層
# ═══════════════════════════════════════════════════════════════════════════

class ReasonAboutSceneSig(dspy.Signature):
    """
    [PRIM] 第五層：場景推理
    BACKEND: LLM (需要強推理能力 — 建議用大模型)
    RELIABILITY: situation_relevance >= 0.80
    
    ★ 這是一條 Primitive，因為「理解場景」不能被分解為更小的
      語義操作——它需要同時考慮所有感知維度並做跨維度推理。
    
    例如：「金屬碰撞聲可幫助李伯伯定位攤位」這個判斷
    需要同時理解噪音類型、使用者情境、和空間導航需求。
    """
    noise_description: str = dspy.InputField(desc="第四層噪音描述")
    speech_description: str = dspy.InputField(desc="第四層語音描述")
    environment_description: str = dspy.InputField(desc="第四層環境描述")
    user_profile: str = dspy.InputField(
        desc="使用者資料：年齡、聽損程度、偏好、當前活動"
    )
    recent_scene_history: str = dspy.InputField(
        desc="最近 N 個場景理解的摘要，用於連續性判斷"
    )
    
    situation: str = dspy.OutputField(desc="完整場景敘述")
    challenges_json: str = dspy.OutputField(
        desc="JSON: 聲學挑戰列表，每個含 challenge/severity/physical_cause"
    )
    preservation_notes_json: str = dspy.OutputField(
        desc="JSON: 需要保留的環境聲線索列表（附保留理由）"
    )
    confidence: float = dspy.OutputField(desc="場景理解信心度 [0,1]")


# ═══════════════════════════════════════════════════════════════════════════
# 第六層：策略生成層 — ★ 目前研究最大空缺 ★
# ═══════════════════════════════════════════════════════════════════════════

class GenerateBeamformingParamsSig(dspy.Signature):
    """
    [PRIM] 第六層：波束成形參數生成
    BACKEND: LLM + physics_constraints
    RELIABILITY: target_direction_error <= 15° in 90% of cases
    CONSTRAINT: beam_width >= 20°（物理約束：麥克風陣列最小波束寬度）
    
    ★ Primitive，因為波束參數需要同時考慮
      語義（目標說話者方向）和物理（陣列幾何約束）。
    """
    scene_understanding: str = dspy.InputField(desc="第五層場景理解")
    mic_geometry: str = dspy.InputField(
        desc="麥克風陣列幾何：間距、數量、排列方式"
    )
    
    target_azimuth_deg: float = dspy.OutputField(desc="目標方位角（度）[-180, 180]")
    beam_width_deg: float = dspy.OutputField(desc="波束寬度（度）[20, 360]")
    null_directions_json: str = dspy.OutputField(desc="JSON: 零點方向列表")
    reasoning: str = dspy.OutputField(desc="決策推理")


class GenerateNoiseReductionParamsSig(dspy.Signature):
    """
    [PRIM] 第六層：降噪參數生成
    BACKEND: LLM
    RELIABILITY: 策略應用後 PESQ 提升 >= 0.3 in 80% of cases
    
    ★ Primitive，因為降噪參數需要權衡使用者偏好和場景需求。
    """
    scene_understanding: str = dspy.InputField(desc="第五層場景理解")
    user_preferences: str = dspy.InputField(desc="使用者偏好：噪音容忍度、處理偏好等")
    
    method: str = dspy.OutputField(
        desc="降噪方法: spectral_subtraction/wiener/dnn_masking"
    )
    aggressiveness: float = dspy.OutputField(desc="攻擊性 [0, 1]")
    preserve_bands_json: str = dspy.OutputField(desc="JSON: 需保留頻段列表")
    reasoning: str = dspy.OutputField(desc="決策推理")


# --- 第六層唯一的確定性 Primitive ---
def prim_generate_gain_params(audiogram_json: str,
                               scene_understanding: str) -> dict:
    """
    [PRIM] 第六層：增益參數生成
    BACKEND: deterministic（NAL-NL2 聽力學公式）
    可靠性: 100%
    
    ★ 注意：這是語義邊界以上的確定性 Primitive。
      不是所有語義層操作都需要 LLM。
      增益計算有成熟的聽力學公式。
    """
    # NAL-NL2 簡化版
    # 真實實現會讀取 audiogram 的每個頻率點
    import json
    try:
        audiogram = json.loads(audiogram_json)
    except:
        audiogram = {"250": 30, "500": 35, "1000": 40, "2000": 50, "4000": 60}
    
    gain_db = {}
    for freq, hearing_loss in audiogram.items():
        # NAL-NL2 簡化公式：gain ≈ hearing_loss * 0.46 + adjustment
        gain_db[freq] = round(float(hearing_loss) * 0.46 + 5, 1)
    
    # 壓縮比根據聽損程度調整
    avg_loss = np.mean(list(audiogram.values()))
    compression = 1.0 + (avg_loss / 100) * 2  # 1:1 到 3:1
    
    return {
        "gain_per_frequency": gain_db,
        "compression_ratio": round(float(compression), 2),
        "attack_ms": 5.0,
        "release_ms": 50.0,
        "deterministic": True  # 標記這是確定性計算
    }


# ═══════════════════════════════════════════════════════════════════════════
# 第七層：意圖與偏好層
# ═══════════════════════════════════════════════════════════════════════════

class ParseIntentSig(dspy.Signature):
    """
    [PRIM] 第七層：意圖解析
    BACKEND: LLM
    RELIABILITY: primary_goal_extraction >= 0.90
    
    使用者可能不會每次都說話——可能只是按了一個按鈕。
    這時 intent 從歷史偏好和當前場景推斷。
    """
    user_action: str = dspy.InputField(
        desc="使用者動作：自然語言指令 或 'button_press:dissatisfied' 或 'none'"
    )
    current_scene: str = dspy.InputField(desc="當前場景理解摘要")
    user_history: str = dspy.InputField(desc="使用者歷史偏好和行為模式")
    
    primary_goal: str = dspy.OutputField(desc="推斷的主要目標")
    secondary_goals_json: str = dspy.OutputField(desc="JSON: 次要目標列表")
    constraints_json: str = dspy.OutputField(desc="JSON: 使用者約束列表")
    confidence: float = dspy.OutputField(desc="意圖解析信心度 [0,1]")


class UpdatePreferencesSig(dspy.Signature):
    """
    [PRIM] 第七層：偏好更新
    BACKEND: LLM
    RELIABILITY: preference_consistency >= 0.85
    
    ★ 從單次回饋推斷長期偏好變化是不可分解的語義推理。
      一個二元信號（滿意/不滿意）加上場景 context，
      要推斷偏好向量的哪個維度需要調整。
    """
    current_preferences: str = dspy.InputField(desc="當前偏好設定")
    user_feedback: str = dspy.InputField(
        desc="使用者回饋：button_press/verbal/implicit_behavior"
    )
    current_scene: str = dspy.InputField(desc="回饋發生時的場景")
    feedback_history: str = dspy.InputField(desc="最近 N 次回饋的摘要")
    
    updated_preferences_json: str = dspy.OutputField(desc="JSON: 更新後的偏好")
    change_reasoning: str = dspy.OutputField(desc="偏好變更的推理過程")
    drift_detected: bool = dspy.OutputField(
        desc="是否偵測到偏好漂移（如聽力退化導致的偏好變化）"
    )


# ═══════════════════════════════════════════════════════════════════════════
# ROUTING SIGNATURES — Composite 層的可學習路由決策（Method A）
# ★ 每個 Routing Predictor 都是 dspy.ChainOfThought → GEPA 可優化
# ★ Primitive 的 prompt 不變，只新增「怎麼組合」的決策層
# ═══════════════════════════════════════════════════════════════════════════

class PerceptAggregateRoutingSig(dspy.Signature):
    """
    [ROUTING] L4 Composite：觀察三個感知 PRIM 的輸出後，
    決定如何整合為統一的感知描述。

    你不做感知描述——三個 PRIM 已經做完了。
    你的職責：
    1. 判斷三個描述各自的品質和在此場景中的重要性
    2. 給出合理的權重分配
    3. 給出不受單一低信心 PRIM 過度拖累的整體信心度

    物理常識提示：
    - SNR 高（>20dB）→ 噪音不嚴重，noise 描述的權重可以降低
    - 聲源數多 → speech 描述的可靠性可能下降（多人重疊）
    - 三個 PRIM 的 confidence 應交叉驗證，不只取 min
    """
    noise_summary: str = dspy.InputField(
        desc="噪音 PRIM 輸出摘要：類型/方向/嚴重度/confidence"
    )
    speech_summary: str = dspy.InputField(
        desc="語音 PRIM 輸出摘要：人數/方向/可懂度/confidence"
    )
    env_summary: str = dspy.InputField(
        desc="環境 PRIM 輸出摘要：類型/特性/confidence"
    )
    snr_db: float = dspy.InputField(desc="原始 SNR(dB)，用於交叉驗證")
    n_sources: int = dspy.InputField(desc="偵測到的聲源數量")

    noise_weight: float = dspy.OutputField(
        desc="噪音描述權重 [0,1]，反映此場景中噪音描述的重要性"
    )
    speech_weight: float = dspy.OutputField(
        desc="語音描述權重 [0,1]"
    )
    env_weight: float = dspy.OutputField(
        desc="環境描述權重 [0,1]"
    )
    overall_confidence: float = dspy.OutputField(
        desc="整合後整體信心度 [0,1]，不一定是三者的 min"
    )
    routing_reasoning: str = dspy.OutputField(
        desc="為什麼這樣分配權重？哪個 PRIM 在此場景最關鍵？"
    )


class SceneRoutingSig(dspy.Signature):
    """
    [ROUTING] L5 Composite：場景推理完成後，決定是否需要矛盾解決，
    以及場景信心度是否需要因歷史一致性而調整。

    你看到的是：
    1. reason_scene 剛產出的場景判斷
    2. 最近的場景歷史
    你要決定：
    - 新場景跟歷史有沒有明顯矛盾？
    - 如果有，是真的場景轉換還是 reason_scene 判斷錯誤？
    - 啟動矛盾解決值不值得？（它會多花一次 LLM 呼叫）
    """
    current_scene_situation: str = dspy.InputField(
        desc="reason_scene 剛產出的場景描述"
    )
    current_scene_confidence: float = dspy.InputField(
        desc="reason_scene 的信心度 [0,1]"
    )
    recent_history: str = dspy.InputField(
        desc="最近 N 個場景的摘要，用 | 分隔；首次執行為 'No history'"
    )
    history_length: int = dspy.InputField(
        desc="歷史紀錄筆數（0=首次執行）"
    )

    should_resolve: bool = dspy.OutputField(
        desc="是否需要啟動矛盾解決？"
        "歷史為空或場景與歷史一致時不需要。"
    )
    history_consistency: str = dspy.OutputField(
        desc="'consistent'(場景穩定) | 'gradual_shift'(漸變) | "
        "'abrupt_change'(突變，可能是真的也可能是誤判)"
    )
    adjusted_confidence: float = dspy.OutputField(
        desc="調整後信心度 [0,1]。"
        "歷史一致→可提高；突變且原始信心低→應降低"
    )
    routing_reasoning: str = dspy.OutputField(desc="決策理由")


class StrategyPlanSig(dspy.Signature):
    """
    [ROUTING] L6 Composite 前置規劃：在 beam/NR/gain 三個 PRIM 執行之前，
    根據場景理解先規劃它們的協作方式。

    你要回答：
    - 此場景的核心聲學挑戰是什麼？
    - beam 和 NR 應該怎麼配合？
    - 總處理預算（保守/中等/積極）是多少？

    物理約束提示：
    - BTE 助聽器只有 2 支麥克風，間距 10mm → beam 最窄 ~20°
    - 方向性噪音 → beam 主導（把 null 對準噪音）
    - 擴散噪音 → NR 主導（beam 幫不了）
    - 使用者偏好「自然」→ 保守預算，避免過度處理
    """
    scene_situation: str = dspy.InputField(desc="L5 場景描述")
    scene_challenges: str = dspy.InputField(desc="L5 識別的挑戰列表 JSON")
    user_preferences: str = dspy.InputField(desc="使用者偏好 JSON")
    mic_geometry: str = dspy.InputField(desc="麥克風陣列幾何")

    primary_challenge: str = dspy.OutputField(
        desc="核心聲學挑戰: "
        "'directional_noise' | 'diffuse_noise' | "
        "'reverberation' | 'quiet'"
    )
    beam_nr_coordination: str = dspy.OutputField(
        desc="beam 和 NR 的協作指令（會被注入各 PRIM 的 context）。"
        "例：'Beam 瞄準前方 0°，NR 應保留 beam 主軸方向的語音頻段'"
    )
    aggressiveness_budget: str = dspy.OutputField(
        desc="'conservative' | 'moderate' | 'aggressive'"
    )
    planning_reasoning: str = dspy.OutputField(desc="規劃理由")


class StrategyIntegrateSig(dspy.Signature):
    """
    [ROUTING] L6 Composite 後置整合：三個 PRIM 都跑完了，
    檢查結果有沒有衝突，給出最終策略信心度。

    常見衝突：
    - beam 瞄 30° 但 NR preserve_bands 沒有保護該方向
    - NR aggressiveness=0.8 但使用者偏好自然（通常 0.3-0.5）
    - beam_width 太窄（<20°）違反物理約束
    - gain 壓縮比太高會讓聲音不自然

    你可以微調 NR aggressiveness 來解決衝突，
    但不要大幅改動（±0.2 以內）。
    """
    beam_summary: str = dspy.InputField(
        desc="beam 結果：azimuth, width, nulls, reasoning"
    )
    nr_summary: str = dspy.InputField(
        desc="NR 結果：method, aggressiveness, preserve_bands, reasoning"
    )
    gain_summary: str = dspy.InputField(
        desc="gain 結果：per-frequency gains, compression ratio"
    )
    coordination_plan: str = dspy.InputField(
        desc="Phase 1 router 規劃的協作指令"
    )
    user_preferences: str = dspy.InputField(desc="使用者偏好 JSON")

    has_conflict: bool = dspy.OutputField(
        desc="三個子策略之間有沒有衝突？"
    )
    conflict_description: str = dspy.OutputField(
        desc="如有衝突，描述是什麼；如無，寫 'none'"
    )
    adjusted_nr_aggressiveness: float = dspy.OutputField(
        desc="整合後 NR 攻擊性 [0,1]，可能微調以配合 beam 和偏好"
    )
    overall_confidence: float = dspy.OutputField(
        desc="策略整體信心度 [0,1]"
    )
    integration_reasoning: str = dspy.OutputField(desc="整合推理")


class PipelineRoutingSig(dspy.Signature):
    """
    [ROUTING] 頂層 Composite：每一幀執行前，決定要跑哪些層。

    助聽器有嚴格的延遲預算：
    - fast (<10ms): L1→L2 only，用 cached DSP params
    - medium (<500ms): L1→L5，更新場景但用 cached 策略
    - full (>500ms): L1→L7，完整更新

    決策依據：
    - 信號變化小 + 上次信心高 → fast（省電、低延遲）
    - 信號變化中等 → medium（更新場景就好）
    - 信號劇烈變化 / 使用者有動作 / 太久沒完整更新 → full
    """
    signal_change_magnitude: float = dspy.InputField(
        desc="當前信號相比上一幀的變化量 [0,1]"
    )
    last_scene_confidence: float = dspy.InputField(
        desc="上一次場景理解的信心度 [0,1]"
    )
    last_strategy_confidence: float = dspy.InputField(
        desc="上一次策略的信心度 [0,1]"
    )
    user_action: str = dspy.InputField(
        desc="使用者動作：'none' | 'button_press:dissatisfied' | ..."
    )
    frames_since_full_update: int = dspy.InputField(
        desc="距離上一次完整七層更新的幀數"
    )

    execution_depth: str = dspy.OutputField(
        desc="'fast' | 'medium' | 'full'"
    )
    force_strategy_update: bool = dspy.OutputField(
        desc="是否強制更新策略？即使 depth 是 medium"
    )
    routing_reasoning: str = dspy.OutputField(desc="選擇此深度的理由")


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE MODULES — 由 Primitive 組合而成的 DSPy Module
# ★ 每個 Composite 現在包含 Routing Predictor（Method A）
# ═══════════════════════════════════════════════════════════════════════════

class FullPerceptualDescription(dspy.Module):
    """
    [COMP] 第四層：完整感知描述
    = describe_noise + describe_speech + describe_environment
    + ★ aggregate_router（Method A：可學習的聚合決策）

    改造前：confidence = min(三者)（hardcoded 保守策略）
    改造後：confidence 由 aggregate_router 根據場景動態決定
    """
    def __init__(self):
        super().__init__()
        # 三個 Primitive，各自獨立可優化（prompt 不動）
        self.describe_noise = dspy.ChainOfThought(DescribeNoiseSig)
        self.describe_speech = dspy.ChainOfThought(DescribeSpeechSig)
        self.describe_env = dspy.ChainOfThought(DescribeEnvironmentSig)
        # ★ 新增：Routing Predictor — GEPA 可優化
        self.aggregate_router = dspy.ChainOfThought(PerceptAggregateRoutingSig)

    def forward(self, acoustic_features: AcousticFeatures,
                user_context: str) -> dspy.Prediction:
        features_str = (
            f"SNR: {acoustic_features.snr_db} dB, "
            f"RT60: {acoustic_features.rt60_s} s, "
            f"Active sources: {acoustic_features.n_active_sources}, "
            f"Spectral centroid: {acoustic_features.spectral_centroid_hz} Hz, "
            f"Energy: {acoustic_features.energy_db} dB SPL, "
            f"Temporal pattern: {acoustic_features.temporal_pattern}, "
            f"MFCC: {acoustic_features.mfcc_summary}"
        )

        # === Phase 1: 三個 PRIM 照跑（frozen prompt 不動）===
        noise_result = self.describe_noise(
            acoustic_features=features_str,
            user_context=user_context
        )
        speech_result = self.describe_speech(
            acoustic_features=features_str
        )
        env_result = self.describe_env(
            acoustic_features=features_str
        )

        # === Phase 2: Routing Predictor 決定怎麼聚合 ===
        routing = self.aggregate_router(
            noise_summary=(
                f"sources={str(noise_result.noise_sources_json)[:200]}, "
                f"confidence={noise_result.confidence}"
            ),
            speech_summary=(
                f"speakers={speech_result.n_speakers}, "
                f"intelligibility={speech_result.intelligibility}, "
                f"confidence={speech_result.confidence}"
            ),
            env_summary=(
                f"type={env_result.environment_type}, "
                f"character={env_result.acoustic_character}, "
                f"confidence={env_result.confidence}"
            ),
            snr_db=acoustic_features.snr_db,
            n_sources=acoustic_features.n_active_sources
        )

        # === Phase 3: 用 router 的判斷組合輸出 ===
        overall_confidence = float(routing.overall_confidence)

        return dspy.Prediction(
            noise_description=noise_result.noise_sources_json,
            speech_description=(
                f"Speakers: {speech_result.n_speakers}, "
                f"Target: {speech_result.target_direction} at {speech_result.target_distance}, "
                f"Intelligibility: {speech_result.intelligibility}"
            ),
            environment_description=(
                f"Type: {env_result.environment_type}, "
                f"Character: {env_result.acoustic_character}"
            ),
            confidence=overall_confidence,
            # ★ 暴露權重，讓下游層和 metric 能看到
            percept_weights={
                "noise": float(routing.noise_weight),
                "speech": float(routing.speech_weight),
                "env": float(routing.env_weight)
            },
            routing_reasoning=routing.routing_reasoning
        )


class SceneWithHistory(dspy.Module):
    """
    [COMP] 第五層：帶歷史的場景理解
    = reason_about_scene |> ★ scene_router |> (conditional) resolve_contradictions

    改造前：if recent_scenes → 一定跑 resolve_contradictions
    改造後：scene_router 根據場景一致性動態決定是否需要矛盾解決
    """
    def __init__(self):
        super().__init__()
        self.reason_scene = dspy.ChainOfThought(ReasonAboutSceneSig)
        # 矛盾解決也是一個 LLM Primitive
        self.resolve_contradictions = dspy.ChainOfThought(
            "current_scene, recent_history -> resolved_scene: str, "
            "is_scene_change: bool, resolution_reasoning: str"
        )
        # ★ 新增：Routing Predictor — 決定要不要跑矛盾解決
        self.scene_router = dspy.ChainOfThought(SceneRoutingSig)

    def forward(self, percept: dspy.Prediction, user_profile: str,
                recent_scenes: list[str]) -> dspy.Prediction:
        history_str = " | ".join(recent_scenes[-5:]) if recent_scenes else "No history"

        # === Phase 1: reason_scene 必跑 ===
        scene_result = self.reason_scene(
            noise_description=percept.noise_description,
            speech_description=percept.speech_description,
            environment_description=percept.environment_description,
            user_profile=user_profile,
            recent_scene_history=history_str
        )

        # === Phase 2: Router 決定要不要跑矛盾解決 ===
        routing = self.scene_router(
            current_scene_situation=scene_result.situation,
            current_scene_confidence=float(scene_result.confidence),
            recent_history=history_str,
            history_length=len(recent_scenes)
        )

        # === Phase 3: 根據 routing 決定執行路徑 ===
        if routing.should_resolve and recent_scenes:
            resolved = self.resolve_contradictions(
                current_scene=scene_result.situation,
                recent_history=history_str
            )
            final_situation = resolved.resolved_scene
        else:
            # ★ Router 說不需要 → 跳過矛盾解決，省一次 LLM call
            final_situation = scene_result.situation

        return dspy.Prediction(
            situation=final_situation,
            challenges_json=scene_result.challenges_json,
            preservation_notes_json=scene_result.preservation_notes_json,
            # ★ 信心度由 router 調整，不再只用 reason_scene 的原始值
            confidence=float(routing.adjusted_confidence),
            history_consistency=routing.history_consistency,
            routing_reasoning=routing.routing_reasoning
        )


class GenerateFullStrategy(dspy.Module):
    """
    [COMP] 第六層：完整策略生成
    = ★ strategy_planner → gen_beam + gen_nr + gain → ★ strategy_integrator

    改造前：beam 和 NR 互不知道對方、confidence 公式離譜
    改造後：
      Phase 1 — planner 規劃協作方式，注入 enriched context
      Phase 2 — beam/NR/gain 執行（beam 和 NR 共享協作指令）
      Phase 3 — integrator 檢查衝突、微調、計算信心度
    """
    def __init__(self):
        super().__init__()
        # 原有 PRIM（prompt 不動）
        self.gen_beam = dspy.ChainOfThought(GenerateBeamformingParamsSig)
        self.gen_nr = dspy.ChainOfThought(GenerateNoiseReductionParamsSig)
        # ★ 新增：前置規劃 + 後置整合
        self.strategy_planner = dspy.ChainOfThought(StrategyPlanSig)
        self.strategy_integrator = dspy.ChainOfThought(StrategyIntegrateSig)

    def forward(self, scene: dspy.Prediction, user_prefs_str: str,
                audiogram_json: str) -> dspy.Prediction:
        scene_str = (
            f"Situation: {scene.situation}\n"
            f"Challenges: {scene.challenges_json}\n"
            f"Preservation notes: {scene.preservation_notes_json}"
        )

        # === Phase 1: 前置規劃 — planner 永遠執行 ===
        plan = self.strategy_planner(
            scene_situation=scene.situation,
            scene_challenges=scene.challenges_json,
            user_preferences=user_prefs_str,
            mic_geometry="BTE hearing aid, 2 mics, 10mm spacing, linear array"
        )

        # ★ 把規劃結果注入 PRIM 的 context — 讓 beam 和 NR 知道彼此的大方向
        enriched_scene_str = (
            f"{scene_str}\n"
            f"[Coordination Plan] {plan.beam_nr_coordination}\n"
            f"[Aggressiveness Budget] {plan.aggressiveness_budget}"
        )

        # === Phase 2: 三個 PRIM 執行（各自獨立 try/except）===
        # ★ 修復：gen_beam 和 gen_nr 失敗時用 PRIM 級 fallback，
        #   不再讓整個 Composite fallback — 確保 planner/integrator 有 trace

        beam_used_fallback = False
        try:
            beam_result = self.gen_beam(
                scene_understanding=enriched_scene_str,
                mic_geometry="BTE hearing aid, 2 mics, 10mm spacing, linear array"
            )
        except Exception as e:
            beam_used_fallback = True
            beam_result = dspy.Prediction(
                target_azimuth_deg=0.0,
                beam_width_deg=60.0,
                null_directions_json='[]',
                reasoning=f"[FALLBACK] gen_beam failed: {str(e)[:100]}. "
                          "Using safe defaults: front-facing, wide beam."
            )

        nr_used_fallback = False
        try:
            nr_result = self.gen_nr(
                scene_understanding=enriched_scene_str,
                user_preferences=user_prefs_str
            )
        except Exception as e:
            nr_used_fallback = True
            nr_result = dspy.Prediction(
                method="wiener",
                aggressiveness=0.5,
                preserve_bands_json='["low-frequency environmental"]',
                reasoning=f"[FALLBACK] gen_nr failed: {str(e)[:100]}. "
                          "Using safe defaults: wiener, moderate aggressiveness."
            )

        # Deterministic Primitive: 增益（永遠成功）
        gain_result = prim_generate_gain_params(audiogram_json, scene_str)

        # === Phase 3: 後置整合 — integrator 永遠執行 ===
        # ★ 即使 beam/NR 用了 fallback，integrator 仍然執行
        #   它能看到 [FALLBACK] 標記，做出合理的信心度判斷
        integration = self.strategy_integrator(
            beam_summary=(
                f"azimuth={beam_result.target_azimuth_deg}°, "
                f"width={beam_result.beam_width_deg}°, "
                f"nulls={beam_result.null_directions_json}, "
                f"reasoning={str(beam_result.reasoning)[:200]}"
                f"{' [USED_FALLBACK]' if beam_used_fallback else ''}"
            ),
            nr_summary=(
                f"method={nr_result.method}, "
                f"aggressiveness={nr_result.aggressiveness}, "
                f"preserve={nr_result.preserve_bands_json}, "
                f"reasoning={str(nr_result.reasoning)[:200]}"
                f"{' [USED_FALLBACK]' if nr_used_fallback else ''}"
            ),
            gain_summary=(
                f"gains={gain_result['gain_per_frequency']}, "
                f"compression={gain_result['compression_ratio']}"
            ),
            coordination_plan=plan.beam_nr_coordination,
            user_preferences=user_prefs_str
        )

        # ★ NR aggressiveness 可能被 integrator 微調
        try:
            final_nr_agg = float(integration.adjusted_nr_aggressiveness)
            final_nr_agg = max(0.0, min(1.0, final_nr_agg))  # clamp to [0,1]
        except (ValueError, TypeError):
            final_nr_agg = float(nr_result.aggressiveness)

        combined_reasoning = (
            f"[Plan] {plan.planning_reasoning}\n"
            f"[Beam] {beam_result.reasoning}"
            f"{' ⚠️ FALLBACK' if beam_used_fallback else ''}\n"
            f"[NR] {nr_result.reasoning}"
            f"{' ⚠️ FALLBACK' if nr_used_fallback else ''}\n"
            f"[Gain] deterministic NAL-NL2, compression={gain_result['compression_ratio']}\n"
            f"[Integration] {integration.integration_reasoning}"
        )

        # ★ 如果用了 fallback，整體信心度打折
        try:
            base_confidence = float(integration.overall_confidence)
        except (ValueError, TypeError):
            base_confidence = 0.5
        fallback_penalty = 0.15 * (beam_used_fallback + nr_used_fallback)
        final_confidence = max(0.1, base_confidence - fallback_penalty)

        return dspy.Prediction(
            target_azimuth_deg=float(beam_result.target_azimuth_deg),
            beam_width_deg=float(beam_result.beam_width_deg),
            null_directions_json=beam_result.null_directions_json,
            nr_method=nr_result.method,
            nr_aggressiveness=final_nr_agg,
            preserve_bands_json=nr_result.preserve_bands_json,
            gain_per_frequency=gain_result["gain_per_frequency"],
            compression_ratio=gain_result["compression_ratio"],
            combined_reasoning=combined_reasoning,
            confidence=final_confidence,
            has_conflict=integration.has_conflict,
            conflict_description=integration.conflict_description,
            primary_challenge=plan.primary_challenge,
            aggressiveness_budget=plan.aggressiveness_budget,
            beam_used_fallback=beam_used_fallback,
            nr_used_fallback=nr_used_fallback
        )


# ═══════════════════════════════════════════════════════════════════════════
# 語意-物理翻譯器 (Semantic → Physical Translator)
# ★ 整個 IR 中最獨特的 Composite：輸入在語義空間，輸出在物理空間
# BACKEND: deterministic — 一旦策略確定，DSP 參數就確定
# ═══════════════════════════════════════════════════════════════════════════

def comp_strategy_to_dsp_params(strategy: dspy.Prediction,
                                 mic_spacing_m: float = 0.01,
                                 sample_rate: int = 16000) -> DSPParameterSet:
    """
    [COMP] 第六→二層：語意到物理的下行翻譯
    BACKEND: deterministic
    
    ★ 輸入是語義空間的（ProcessingStrategy with 自然語言推理）
    ★ 輸出是物理空間的（精確的濾波器係數和波束權重）
    ★ 翻譯本身是確定性的——LLM 的作用在於生成策略，
      翻譯過程是純數學
    """
    import json
    
    # 1. 波束成形權重計算（確定性）
    azimuth_rad = np.radians(strategy.target_azimuth_deg)
    d = mic_spacing_m
    c = 343.0  # 聲速 m/s
    freq_center = 2000  # 2kHz 中心頻率
    phase_diff = 2 * np.pi * freq_center * d * np.sin(azimuth_rad) / c
    beam_weights = [1.0, float(np.cos(phase_diff))]
    
    # 2. 降噪遮罩計算（確定性）
    n_bins = 129  # 256-point FFT
    try:
        preserve = json.loads(strategy.preserve_bands_json)
    except:
        preserve = []
    
    mask = np.ones(n_bins) * strategy.nr_aggressiveness
    # 保留頻段的遮罩設為較低值
    for band in preserve:
        if isinstance(band, str) and "low" in band.lower():
            mask[:n_bins // 4] *= 0.3
        elif isinstance(band, str) and "mid" in band.lower():
            mask[n_bins // 4: n_bins * 3 // 4] *= 0.3
    
    # 3. 增益濾波器係數（確定性，從 gain_per_frequency 計算 FIR）
    gain_dict = strategy.gain_per_frequency
    if isinstance(gain_dict, dict):
        gains = list(gain_dict.values())
    else:
        gains = [20.0] * 6
    
    # 簡化版：從頻域增益生成 FIR 係數
    freq_response = np.interp(
        np.linspace(0, 8000, n_bins),
        [250, 500, 1000, 2000, 4000, 8000][:len(gains)],
        [10 ** (g / 20) for g in gains[:6]]
    )
    filter_coeffs = np.fft.irfft(freq_response, n=32).tolist()
    
    return DSPParameterSet(
        filter_coeffs=filter_coeffs,
        beam_weights=beam_weights,
        noise_mask=mask.tolist(),
        compression_ratio=strategy.compression_ratio,
        attack_ms=5.0,
        release_ms=50.0
    )


# ═══════════════════════════════════════════════════════════════════════════
# HARNESS — 橫跨 Runtime + Linker + Scheduler + Persistent Store
# 這是整個系統的「操作系統」
# ═══════════════════════════════════════════════════════════════════════════

class AcousticSemanticHarness(dspy.Module):
    """
    ★★★ 最頂層 Composite：完整的七層管線 ★★★

    [COMP] intent_aware_strategy
    = ★ pipeline_router → 決定 execution_depth
      → 第一層(sample) → 第二層(DSP) → 第三層(features)
      → 第四層(percept) → 第五層(scene) → 第六層(strategy)
      → 翻譯回第二層(DSP params)

    Harness 的四個子系統：

    1. Semantic Linker — 各層輸出→輸入的格式適配
    2. Semantic Runtime — Context window 管理、Error recovery
    3. Semantic Scheduler — 模型選擇、延遲約束
       ★ 現在由 pipeline_router 動態決定 execution depth
    4. Persistent Store — 偏好持久化、場景歷史、cached 結果
    """

    def __init__(self,
                 fast_lm=None,    # 第四層用的小模型（低延遲）
                 strong_lm=None,  # 第五、六層用的大模型（強推理）
                 ):
        super().__init__()

        # === Semantic Scheduler: 不同層用不同模型 ===
        self.fast_lm = fast_lm
        self.strong_lm = strong_lm

        # === Semantic Modules（全部是 GEPA 可優化的） ===
        # 第四層 Composite（含 aggregate_router）
        self.perceptual_desc = FullPerceptualDescription()
        # 第五層 Composite（含 scene_router）
        self.scene_understanding = SceneWithHistory()
        # 第六層 Composite（含 strategy_planner + strategy_integrator）
        self.strategy_gen = GenerateFullStrategy()
        # 第七層 Primitives
        self.parse_intent = dspy.ChainOfThought(ParseIntentSig)
        self.update_prefs = dspy.ChainOfThought(UpdatePreferencesSig)
        # ★ 新增：Pipeline Router — 決定每幀跑多深
        self.pipeline_router = dspy.ChainOfThought(PipelineRoutingSig)

        # === Persistent Store ===
        self.scene_history: list[str] = []
        self.feedback_history: list[str] = []
        self.current_preferences = {
            "noise_tolerance": "medium",
            "processing_preference": "natural",
            "environment_awareness": "moderate",
            "known_situations": ["菜市場: 增強正前方, 保留環境感"]
        }
        self.current_dsp_params: Optional[DSPParameterSet] = None
        # ★ 新增：cached 中間結果（供 fast/medium 通道使用）
        self._cached_percept: Optional[dspy.Prediction] = None
        self._cached_scene: Optional[dspy.Prediction] = None
        self._cached_strategy: Optional[dspy.Prediction] = None
        self._last_scene_conf: float = 0.5
        self._last_strategy_conf: float = 0.5
        self._frames_since_full: int = 0
        self._last_signal_energy: float = 0.0

    def _estimate_signal_change(self, raw_signal: RawSignal) -> float:
        """估算信號相比上一幀的變化量 [0,1]"""
        current_energy = np.mean(np.abs(raw_signal.samples[0]))
        if self._last_signal_energy == 0:
            self._last_signal_energy = current_energy
            return 1.0  # 第一幀，視為最大變化 → 強制 full
        delta = abs(current_energy - self._last_signal_energy)
        normalized = min(1.0, delta / (self._last_signal_energy + 1e-8))
        self._last_signal_energy = current_energy
        return float(normalized)

    def forward(self,
                # --- 輸入 ---
                raw_signal: Optional[RawSignal] = None,
                user_action: str = "none",
                audiogram_json: str = '{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
                user_profile: str = "72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
                ) -> dspy.Prediction:
        """
        ★ 控制迴路的一次完整執行 ★

        多速率執行模式（現在由 pipeline_router 動態決定）：
        - fast  (<10ms):  L1→L2 only，用 cached params
        - medium (<500ms): L1→L5，更新場景但用 cached 策略
        - full  (>500ms):  L1→L7，完整更新
        """

        import json

        # ═══ 第一層：物理感測（確定性，永遠跑）═══
        if raw_signal is None:
            raw_signal = prim_sample_audio(duration_ms=32.0)

        # ═══ 第二層：快速通道 DSP（確定性，用 cached 參數）═══
        if self.current_dsp_params is not None:
            processed = prim_beamform(raw_signal,
                                      self.current_dsp_params.beam_weights[0] * 30)
        else:
            processed = raw_signal.samples[0]

        # ═══ Pipeline Router：決定這一幀要跑多深 ═══
        signal_change = self._estimate_signal_change(raw_signal)

        routing = self.pipeline_router(
            signal_change_magnitude=signal_change,
            last_scene_confidence=self._last_scene_conf,
            last_strategy_confidence=self._last_strategy_conf,
            user_action=user_action,
            frames_since_full_update=self._frames_since_full
        )

        depth = str(routing.execution_depth).strip().lower()

        # ═══ Fast 通道：直接回傳 cached 結果 ═══
        if (depth == "fast"
                and self.current_dsp_params is not None
                and self._cached_percept is not None):
            self._frames_since_full += 1
            return dspy.Prediction(
                features=None,
                percept=self._cached_percept,
                scene=self._cached_scene,
                strategy=self._cached_strategy,
                dsp_params=self.current_dsp_params,
                scene_history=self.scene_history[-5:],
                current_preferences=self.current_preferences,
                execution_depth="fast",
                routing_reasoning=routing.routing_reasoning
            )

        # ═══ 第三層：特徵提取（確定性）═══
        features = comp_extract_full_features(raw_signal)

        # ═══ 第四層：感知描述（LLM — 用小模型）═══
        fast_ctx = dspy.context(lm=self.fast_lm) if self.fast_lm else nullcontext()
        try:
            with fast_ctx:
                percept = self.perceptual_desc(
                    acoustic_features=features,
                    user_context=user_profile
                )
        except Exception as e:
            percept = dspy.Prediction(
                noise_description='[{"type":"unknown","direction":"unknown","temporal":"unknown","severity":"moderate"}]',
                speech_description="Speakers: 1, Target: front, Intelligibility: unknown",
                environment_description="Type: unknown, Character: noisy",
                confidence=0.3
            )

        # ═══ 第五層：場景理解（LLM — 用大模型）═══
        strong_ctx = dspy.context(lm=self.strong_lm) if self.strong_lm else nullcontext()
        try:
            with strong_ctx:
                scene = self.scene_understanding(
                    percept=percept,
                    user_profile=user_profile,
                    recent_scenes=self.scene_history
                )
        except Exception as e:
            scene = dspy.Prediction(
                situation="Unable to determine scene",
                challenges_json='[]',
                preservation_notes_json='[]',
                confidence=0.2
            )

        # Persistent Store: 記錄場景歷史 + cache
        self.scene_history.append(scene.situation)
        if len(self.scene_history) > 20:
            self.scene_history = self.scene_history[-20:]
        self._last_scene_conf = float(scene.confidence)
        self._cached_percept = percept
        self._cached_scene = scene

        # ═══ Medium 通道：更新場景但不更新策略 ═══
        if (depth == "medium"
                and not routing.force_strategy_update
                and self._cached_strategy is not None
                and self.current_dsp_params is not None):
            self._frames_since_full += 1
            return dspy.Prediction(
                features=features,
                percept=percept,
                scene=scene,
                strategy=self._cached_strategy,
                dsp_params=self.current_dsp_params,
                scene_history=self.scene_history[-5:],
                current_preferences=self.current_preferences,
                execution_depth="medium",
                routing_reasoning=routing.routing_reasoning
            )

        # ═══ Full 通道：以下執行完整的 L6 + L7 ═══

        # ═══ 第七層：意圖解析（如果有使用者動作）═══
        if user_action != "none":
            intent = self.parse_intent(
                user_action=user_action,
                current_scene=scene.situation,
                user_history=json.dumps(self.current_preferences, ensure_ascii=False)
            )

            if "dissatisfied" in user_action or "satisfied" in user_action:
                pref_update = self.update_prefs(
                    current_preferences=json.dumps(self.current_preferences,
                                                    ensure_ascii=False),
                    user_feedback=user_action,
                    current_scene=scene.situation,
                    feedback_history=" | ".join(self.feedback_history[-5:])
                )
                self.feedback_history.append(
                    f"{user_action} in {scene.situation[:50]}"
                )
                try:
                    updated = json.loads(pref_update.updated_preferences_json)
                    self.current_preferences.update(updated)
                except:
                    pass

        # ═══ 第六層：策略生成（LLM + deterministic 混合）═══
        # ★ 不再用外層 try/except — GenerateFullStrategy 內部已有
        #   per-PRIM 的 fallback，planner/integrator 永遠會執行
        prefs_str = json.dumps(self.current_preferences, ensure_ascii=False)

        # ★ 建立新的 context manager（不能重用 strong_ctx，它是一次性的）
        l6_ctx = dspy.context(lm=self.strong_lm) if self.strong_lm else nullcontext()
        with l6_ctx:
            strategy = self.strategy_gen(
                scene=scene,
                user_prefs_str=prefs_str,
                audiogram_json=audiogram_json
            )

        # ═══ 語意→物理翻譯（確定性）═══
        dsp_params = comp_strategy_to_dsp_params(strategy)

        # Persistent Store: cache 所有結果
        self.current_dsp_params = dsp_params
        self._cached_strategy = strategy
        self._last_strategy_conf = float(strategy.confidence)
        self._frames_since_full = 0

        return dspy.Prediction(
            features=features,
            percept=percept,
            scene=scene,
            strategy=strategy,
            dsp_params=dsp_params,
            scene_history=self.scene_history[-5:],
            current_preferences=self.current_preferences,
            execution_depth="full",
            routing_reasoning=routing.routing_reasoning
        )


# ═══════════════════════════════════════════════════════════════════════════
# GEPA 優化 — 用 GEPA 編譯整個管線
# ★ GEPA 的 feedback function 是整個系統的「可靠性規格」
# ═══════════════════════════════════════════════════════════════════════════

def create_acoustic_feedback_metric(gold, pred, trace=None,
                                     pred_name=None, pred_trace=None):
    """
    GEPA 的 feedback metric — 這是整個系統的「可靠性契約」

    回傳邏輯：
    - 不管有沒有 pred_name，score 都以 module-level（全管線）為準
    - pred_name 存在時，回傳針對該 predictor 的文字 feedback
      （GEPA 用 feedback 做 reflection，不用 predictor-level score）
    """
    import json

    # ===== 先算 module-level score（全管線）=====
    module_score = 0.0
    module_feedback = []

    has_dsp = hasattr(pred, 'dsp_params') and pred.dsp_params is not None
    if has_dsp:
        module_score += 0.25
        module_feedback.append("OK: DSP params generated.")
    else:
        module_feedback.append("FAIL: No DSP params produced.")

    has_scene_conf = (hasattr(pred, 'scene')
                      and hasattr(pred.scene, 'confidence')
                      and float(pred.scene.confidence) > 0.3)
    if has_scene_conf:
        module_score += 0.25
        module_feedback.append(f"OK: Scene confidence={pred.scene.confidence}.")
    else:
        conf = getattr(getattr(pred, 'scene', None), 'confidence', 'N/A')
        module_feedback.append(
            f"WEAK: Scene confidence={conf} <= 0.3. "
            "Improve perceptual description clarity so scene reasoning has stronger inputs."
        )

    has_reasoning = (hasattr(pred, 'strategy')
                     and hasattr(pred.strategy, 'combined_reasoning')
                     and len(str(pred.strategy.combined_reasoning)) > 30)
    if has_reasoning:
        module_score += 0.25
        module_feedback.append("OK: Strategy reasoning present.")
    else:
        module_feedback.append("WEAK: Strategy reasoning too short or missing.")

    has_percept = (hasattr(pred, 'percept')
                   and hasattr(pred.percept, 'confidence')
                   and float(pred.percept.confidence) > 0.3)
    if has_percept:
        module_score += 0.25
    else:
        module_feedback.append("WEAK: Perceptual description confidence low.")

    module_score = min(module_score, 1.0)

    # ===== 如果 GEPA 要求 predictor-level feedback =====
    if pred_name:
        pred_feedback = []

        if "describe_noise" in pred_name:
            try:
                sources = json.loads(pred.noise_sources_json)
                if len(sources) > 0:
                    pred_feedback.append(f"Good: identified {len(sources)} noise source(s).")
                    if hasattr(gold, 'snr_db') and gold.snr_db > 20:
                        if any(s.get('severity') == 'severe' for s in sources):
                            pred_feedback.append(
                                "PHYSICS VIOLATION: SNR > 20dB but noise rated severe. "
                                "High SNR means noise is low relative to signal."
                            )
                else:
                    pred_feedback.append("No noise sources identified — identify at least one.")
            except Exception:
                pred_feedback.append("Failed to parse noise_sources_json as valid JSON.")

        elif "describe_speech" in pred_name:
            if hasattr(pred, 'n_speakers'):
                pred_feedback.append(f"Detected {pred.n_speakers} speaker(s).")
            if hasattr(pred, 'intelligibility'):
                pred_feedback.append(f"Intelligibility: {pred.intelligibility}.")
            if not pred_feedback:
                pred_feedback.append("Ensure n_speakers and intelligibility are filled.")

        elif "describe_env" in pred_name:
            if hasattr(pred, 'environment_type') and len(str(pred.environment_type)) > 5:
                pred_feedback.append(f"Environment: {pred.environment_type}.")
            else:
                pred_feedback.append("Environment type too vague — be specific (e.g. 'indoor market').")

        elif "reason_scene" in pred_name:
            if hasattr(pred, 'challenges_json'):
                try:
                    challenges = json.loads(pred.challenges_json)
                    if len(challenges) > 0:
                        for c in challenges:
                            if 'physical_cause' not in c or len(str(c.get('physical_cause', ''))) < 10:
                                pred_feedback.append(
                                    f"Challenge '{c.get('challenge','')}' lacks physical cause. "
                                    "Every challenge must trace to a physical mechanism."
                                )
                        if not pred_feedback:
                            pred_feedback.append("Good: all challenges have physical causes.")
                    else:
                        pred_feedback.append("No challenges identified — a noisy scene always has challenges.")
                except Exception:
                    pred_feedback.append("challenges_json is not valid JSON.")
            if hasattr(pred, 'preservation_notes_json'):
                try:
                    notes = json.loads(pred.preservation_notes_json)
                    if len(notes) == 0:
                        pred_feedback.append(
                            "No preservation notes. Which sounds help the user navigate?"
                        )
                except Exception:
                    pass

        elif "gen_beam" in pred_name:
            if hasattr(pred, 'beam_width_deg'):
                bw = float(pred.beam_width_deg)
                if bw < 20:
                    pred_feedback.append(
                        f"PHYSICS VIOLATION: beam_width {bw}° < 20°. "
                        "A 2-mic BTE with 10mm spacing cannot form beams < 20°."
                    )
                else:
                    pred_feedback.append(f"Beam width {bw}° respects physical minimum.")
            if hasattr(pred, 'reasoning') and len(str(pred.reasoning)) < 30:
                pred_feedback.append("Provide detailed reasoning for direction choice.")

        elif "gen_nr" in pred_name:
            if hasattr(pred, 'aggressiveness'):
                agg = float(pred.aggressiveness)
                if hasattr(gold, 'user_prefs_natural') and gold.user_prefs_natural and agg > 0.8:
                    pred_feedback.append(
                        f"User prefers 'natural' but aggressiveness={agg} > 0.8. "
                        "Natural preference typically maps to 0.3-0.5."
                    )

        # ★ 新增：Routing Predictor feedback ★

        elif "aggregate_router" in pred_name:
            # L4 聚合路由的 feedback
            if hasattr(pred, 'noise_weight'):
                try:
                    nw = float(pred.noise_weight)
                    sw = float(pred.speech_weight)
                    ew = float(pred.env_weight)
                    total = nw + sw + ew
                    if total < 0.5 or total > 2.0:
                        pred_feedback.append(
                            f"Weights sum={total:.2f} — should be close to 1.0. "
                            "Normalize so downstream can interpret as relative importance."
                        )
                    if hasattr(gold, 'snr_db') and gold.snr_db > 20 and nw > 0.5:
                        pred_feedback.append(
                            f"SNR={gold.snr_db}dB (quiet) but noise_weight={nw:.2f} > 0.5. "
                            "In quiet scenes, speech and environment should dominate."
                        )
                    if hasattr(gold, 'snr_db') and gold.snr_db < 5 and nw < 0.2:
                        pred_feedback.append(
                            f"SNR={gold.snr_db}dB (very noisy) but noise_weight={nw:.2f} < 0.2. "
                            "In noisy scenes, noise description is critical for downstream."
                        )
                except (ValueError, TypeError):
                    pred_feedback.append("Could not parse weights as floats.")
            if hasattr(pred, 'overall_confidence'):
                try:
                    conf = float(pred.overall_confidence)
                    if conf > 0.95:
                        pred_feedback.append(
                            "Confidence > 0.95 is overconfident for LLM-based perception."
                        )
                    if conf < 0.1:
                        pred_feedback.append(
                            "Confidence < 0.1 is too pessimistic — some information was extracted."
                        )
                except (ValueError, TypeError):
                    pass

        elif "scene_router" in pred_name:
            # L5 場景路由的 feedback
            if hasattr(pred, 'should_resolve'):
                if hasattr(pred, 'history_length'):
                    try:
                        hl = int(pred.history_length) if not isinstance(pred.history_length, int) else pred.history_length
                    except (ValueError, TypeError):
                        hl = -1
                    if hl == 0 and pred.should_resolve:
                        pred_feedback.append(
                            "No history exists (history_length=0) but should_resolve=True. "
                            "Cannot resolve contradictions without history."
                        )
            if hasattr(pred, 'adjusted_confidence'):
                try:
                    ac = float(pred.adjusted_confidence)
                    if hasattr(pred, 'current_scene_confidence'):
                        orig = float(pred.current_scene_confidence)
                        if abs(ac - orig) > 0.4:
                            pred_feedback.append(
                                f"Adjusted confidence ({ac:.2f}) deviates > 0.4 from "
                                f"original ({orig:.2f}). Large adjustments should have "
                                "strong justification."
                            )
                except (ValueError, TypeError):
                    pass
            if not pred_feedback:
                pred_feedback.append("Scene routing decision looks reasonable.")

        elif "strategy_planner" in pred_name:
            # L6 前置規劃的 feedback
            if hasattr(pred, 'primary_challenge'):
                challenge = str(pred.primary_challenge).strip()
                valid_challenges = {'directional_noise', 'diffuse_noise',
                                    'reverberation', 'quiet'}
                if challenge not in valid_challenges:
                    pred_feedback.append(
                        f"primary_challenge='{challenge}' is not one of "
                        f"{valid_challenges}. Use an exact match."
                    )
                if hasattr(gold, 'snr_db'):
                    if gold.snr_db > 20 and challenge in ('directional_noise', 'diffuse_noise'):
                        pred_feedback.append(
                            f"SNR={gold.snr_db}dB is high but challenge='{challenge}'. "
                            "High SNR means noise is not the primary issue."
                        )
                    if gold.snr_db < 5 and challenge == 'quiet':
                        pred_feedback.append(
                            f"SNR={gold.snr_db}dB is very low but challenge='quiet'. "
                            "This is a noisy scene."
                        )
            if hasattr(pred, 'aggressiveness_budget'):
                budget = str(pred.aggressiveness_budget).strip()
                if hasattr(gold, 'user_prefs_natural') and gold.user_prefs_natural:
                    if budget == 'aggressive':
                        # 除非使用者不滿意，否則偏好自然的人不該用 aggressive
                        if hasattr(gold, 'user_action') and 'dissatisfied' not in str(gold.user_action):
                            pred_feedback.append(
                                "User prefers natural sound but budget='aggressive'. "
                                "Only use aggressive if user explicitly dissatisfied."
                            )
            if hasattr(pred, 'beam_nr_coordination'):
                coord = str(pred.beam_nr_coordination)
                if len(coord) < 20:
                    pred_feedback.append(
                        "Coordination plan is too short. Provide specific guidance "
                        "for how beam and NR should work together."
                    )
            if not pred_feedback:
                pred_feedback.append("Strategy planning looks reasonable.")

        elif "strategy_integrator" in pred_name:
            # L6 後置整合的 feedback
            if hasattr(pred, 'adjusted_nr_aggressiveness'):
                try:
                    agg = float(pred.adjusted_nr_aggressiveness)
                    if agg < 0 or agg > 1:
                        pred_feedback.append(
                            f"adjusted_nr_aggressiveness={agg} out of [0,1] range."
                        )
                    if hasattr(gold, 'user_prefs_natural') and gold.user_prefs_natural:
                        if agg > 0.8:
                            pred_feedback.append(
                                f"User prefers natural but final aggressiveness={agg:.2f}. "
                                "Should be 0.3-0.6 for natural preference."
                            )
                except (ValueError, TypeError):
                    pred_feedback.append(
                        "Could not parse adjusted_nr_aggressiveness as float."
                    )
            if hasattr(pred, 'overall_confidence'):
                try:
                    conf = float(pred.overall_confidence)
                    if conf > 0.95:
                        pred_feedback.append(
                            "Strategy confidence > 0.95 is overconfident."
                        )
                except (ValueError, TypeError):
                    pass
            if not pred_feedback:
                pred_feedback.append("Strategy integration looks reasonable.")

        elif "pipeline_router" in pred_name:
            # 頂層管線路由的 feedback
            if hasattr(pred, 'execution_depth'):
                depth = str(pred.execution_depth).strip().lower()
                valid_depths = {'fast', 'medium', 'full'}
                if depth not in valid_depths:
                    pred_feedback.append(
                        f"execution_depth='{depth}' is not one of {valid_depths}."
                    )
                if hasattr(gold, 'user_action'):
                    if str(gold.user_action) != 'none' and depth != 'full':
                        pred_feedback.append(
                            f"User took action '{gold.user_action}' but depth='{depth}'. "
                            "User actions should trigger full pipeline execution."
                        )
                if hasattr(pred, 'frames_since_full_update'):
                    try:
                        fsfu = int(pred.frames_since_full_update) if not isinstance(
                            pred.frames_since_full_update, int
                        ) else pred.frames_since_full_update
                        if fsfu > 50 and depth == 'fast':
                            pred_feedback.append(
                                f"frames_since_full={fsfu} > 50 but depth='fast'. "
                                "Run at least medium to prevent stale scene data."
                            )
                    except (ValueError, TypeError):
                        pass
            if not pred_feedback:
                pred_feedback.append("Pipeline routing decision looks reasonable.")

        if pred_feedback:
            return dspy.Prediction(
                score=module_score,
                feedback="\n".join(pred_feedback)
            )

    return dspy.Prediction(
        score=module_score,
        feedback="\n".join(module_feedback)
    )


def create_training_examples():
    """
    創建訓練集——李伯伯一天中的不同場景
    每個 Example 代表一個（輸入, 期望行為）對
    """
    examples = []
    
    # 場景 1: 菜市場魚攤（核心場景）
    examples.append(dspy.Example(
        scenario="market_fish_stall",
        snr_db=5.0,
        rt60_s=0.8,
        n_active_sources=6,
        energy_db=78.0,
        temporal_pattern="modulated",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
        user_action="none",
        # 期望行為
        expected_env="indoor market",
        expected_target_dir="front",
        expected_nr_agg_range=(0.3, 0.6),  # natural preference
        expected_beam_width_range=(30, 60),
        user_prefs_natural=True,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action"
    ))
    
    # 場景 2: 安靜的家中跟老伴說話
    examples.append(dspy.Example(
        scenario="home_quiet",
        snr_db=25.0,
        rt60_s=0.4,
        n_active_sources=1,
        energy_db=55.0,
        temporal_pattern="modulated",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
        user_action="none",
        expected_env="home/living room",
        expected_target_dir="front",
        expected_nr_agg_range=(0.0, 0.2),
        expected_beam_width_range=(60, 120),
        user_prefs_natural=True,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action"
    ))
    
    # 場景 3: 看電視 + 老伴說話
    examples.append(dspy.Example(
        scenario="tv_plus_conversation",
        snr_db=10.0,
        rt60_s=0.5,
        n_active_sources=3,
        energy_db=65.0,
        temporal_pattern="modulated",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
        user_action="none",
        expected_env="home/living room",
        expected_target_dir="variable",
        expected_nr_agg_range=(0.2, 0.5),
        expected_beam_width_range=(40, 80),
        user_prefs_natural=True,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action"
    ))
    
    # 場景 4: 公園散步（風聲 + 兒童笑聲）
    examples.append(dspy.Example(
        scenario="park_walking",
        snr_db=15.0,
        rt60_s=0.1,  # 戶外幾乎無混響
        n_active_sources=4,
        energy_db=60.0,
        temporal_pattern="stationary",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
        user_action="none",
        expected_env="outdoor park",
        expected_target_dir="variable",
        expected_nr_agg_range=(0.1, 0.4),
        expected_beam_width_range=(60, 180),
        user_prefs_natural=True,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action"
    ))
    
    # 場景 5: 菜市場 + 使用者按「不滿意」按鈕
    examples.append(dspy.Example(
        scenario="market_dissatisfied",
        snr_db=3.0,
        rt60_s=0.9,
        n_active_sources=8,
        energy_db=82.0,
        temporal_pattern="modulated",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
        user_action="button_press:dissatisfied",
        expected_env="indoor market",
        expected_target_dir="front",
        expected_nr_agg_range=(0.5, 0.8),  # 按了不滿意，應該增強降噪
        expected_beam_width_range=(20, 40),  # 按了不滿意，應該收窄波束
        user_prefs_natural=True,
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action"
    ))
    
    return examples


class GEPATrainableHarness(dspy.Module):
    """
    ★ GEPA 訓練用 Wrapper ★

    橋接 training examples 的 input fields（scenario, snr_db, ...）
    和 AcousticSemanticHarness.forward() 的參數。

    GEPA 呼叫 forward(example_inputs) → 這個 wrapper 做轉換 → 呼叫真正的 Harness。
    """
    def __init__(self, fast_lm=None, strong_lm=None):
        super().__init__()
        self.harness = AcousticSemanticHarness(
            fast_lm=fast_lm,
            strong_lm=strong_lm
        )

    def forward(self, scenario: str = "", snr_db: float = 10.0,
                rt60_s: float = 0.5, n_active_sources: int = 3,
                energy_db: float = 65.0, temporal_pattern: str = "modulated",
                user_profile: str = "72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
                user_action: str = "none") -> dspy.Prediction:
        # 用 prim_sample_audio 生成模擬信號（確定性層不需要優化）
        raw_signal = prim_sample_audio(duration_ms=32.0, n_channels=2)

        # 呼叫真正的 Harness
        result = self.harness(
            raw_signal=raw_signal,
            user_action=user_action,
            user_profile=user_profile
        )

        return result


def compile_with_gepa():
    """
    ★★★ 用 GEPA 編譯整個管線 ★★★

    GEPA 做的事情類比 LLVM 的 optimization passes：
    1. 它看到每個 Primitive 的失敗案例
    2. 用 reflection_lm 分析失敗原因
    3. 提出改進的 prompt（= 改進的指令）
    4. 在 Pareto frontier 上保留最佳候選

    GEPA 的 feedback function 就是 IR 的可靠性規格。
    物理約束以文字回饋的形式注入 GEPA 的反思過程。
    """
    import json

    # 配置 — 全部用 gpt-4o-mini 降低成本，可替換為 gpt-4o
    task_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)
    reflection_lm = dspy.LM("openai/gpt-4o-mini", temperature=1.0, max_tokens=8000)
    dspy.configure(lm=task_lm)

    # 創建可訓練的 Harness（用 wrapper 橋接 example fields）
    trainable = GEPATrainableHarness(
        fast_lm=task_lm,
        strong_lm=task_lm
    )

    # 訓練集 — 包含 dissatisfied example 以觸發 L7 predictors
    examples = create_training_examples()
    # examples: [market, home, tv, park, market_dissatisfied]
    # trainset 要包含 dissatisfied 讓 parse_intent/update_prefs 有 trace
    train_set = [examples[0], examples[2], examples[4]]  # market, tv, dissatisfied
    val_set = [examples[1], examples[3]]                  # home, park

    print("\n" + "=" * 70)
    print("★★★ GEPA 編譯開始 ★★★")
    print(f"  訓練集: {len(train_set)} examples")
    for ex in train_set:
        print(f"    - {ex.scenario} (user_action={ex.user_action})")
    print(f"  驗證集: {len(val_set)} examples")
    for ex in val_set:
        print(f"    - {ex.scenario}")
    print(f"  Task LM: openai/gpt-4o-mini")
    print(f"  Reflection LM: openai/gpt-4o-mini")
    print("  Predictors to optimize:")
    for name, _ in trainable.named_predictors():
        print(f"    - {name}")
    print("=" * 70)

    # ★ GEPA 編譯 — 限制預算在 ~5-10 分鐘
    optimizer = dspy.GEPA(
        metric=create_acoustic_feedback_metric,
        max_metric_calls=100,
        num_threads=2,
        track_stats=True,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        warn_on_score_mismatch=False,
    )

    optimized = optimizer.compile(
        trainable,
        trainset=train_set,
        valset=val_set,
    )

    print("\n" + "=" * 70)
    print("★★★ GEPA 編譯完成 ★★★")
    print("=" * 70)

    # 展示優化後的 predictor instructions
    print("\n[GEPA 優化後的 Predictor Instructions]")
    for name, pred in optimized.named_predictors():
        sig = pred.signature
        inst = sig.instructions if hasattr(sig, 'instructions') else str(sig)
        print(f"\n--- {name} ---")
        print(f"  {str(inst)[:300]}")

    # 用優化後的模型跑一次驗證
    print("\n" + "-" * 70)
    print(">>> 用優化後的管線執行驗證場景")
    dspy.configure(lm=task_lm)
    try:
        val_example = val_set[0]
        val_result = optimized(
            scenario=val_example.scenario,
            snr_db=val_example.snr_db,
            rt60_s=val_example.rt60_s,
            n_active_sources=val_example.n_active_sources,
            energy_db=val_example.energy_db,
            temporal_pattern=val_example.temporal_pattern,
            user_profile=val_example.user_profile,
            user_action=val_example.user_action,
        )
        print(f"\n  場景: {val_example.scenario}")
        print(f"  [Layer 4] Percept confidence: {val_result.percept.confidence}")
        print(f"  [Layer 5] Scene: {val_result.scene.situation[:200]}")
        print(f"  [Layer 6] Strategy: beam={val_result.strategy.target_azimuth_deg}°, "
              f"NR={val_result.strategy.nr_aggressiveness}")
        print(f"  [Layer 2] DSP params generated: {val_result.dsp_params is not None}")
    except Exception as e:
        print(f"  驗證執行時發生錯誤: {e}")

    return optimized


# ═══════════════════════════════════════════════════════════════════════════
# 完整架構圖的文字版
# ═══════════════════════════════════════════════════════════════════════════

ARCHITECTURE_MAP = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                    ACOUSTIC SEMANTIC IR (ASIR) v0.2                     ║
║      DSPy + GEPA + Method A (Learnable Routing) Architecture           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  ★ Pipeline Router (PipelineRoutingSig)                                ║
║  ┌─────────────────────────────────────────────────────────────────┐   ║
║  │ [ROUTING] pipeline_router → 'fast' | 'medium' | 'full'         │   ║
║  │   決定每幀跑多深：fast=L1-2, medium=L1-5, full=L1-7            │   ║
║  └──────────────────────────────┬──────────────────────────────────┘   ║
║                                 │                                       ║
║  Layer 7: Intent & Preference   │                                       ║
║  ┌──────────────────────────────┴──────────────────────────────────┐   ║
║  │ [PRIM] ParseIntentSig         → dspy.ChainOfThought  → LLM     │   ║
║  │ [PRIM] UpdatePreferencesSig   → dspy.ChainOfThought  → LLM     │   ║
║  └──────────────────────────────┬──────────────────────────────────┘   ║
║                                 │                                       ║
║  Layer 6: Strategy Generation   │ ★ 前置規劃 + 後置整合               ║
║  ┌──────────────────────────────┴──────────────────────────────────┐   ║
║  │ [ROUTING] strategy_planner   → 前置：規劃 beam/NR 協作方式      │   ║
║  │    ↓ enriched context (coordination plan + budget)               │   ║
║  │ [PRIM] GenerateBeamformingSig → LLM+physics (收到協作指令)      │   ║
║  │ [PRIM] GenerateNRSig         → LLM         (收到協作指令)      │   ║
║  │ [PRIM] prim_generate_gain()  → deterministic (NAL-NL2)         │   ║
║  │    ↓ 三個結果                                                    │   ║
║  │ [ROUTING] strategy_integrator → 後置：衝突檢查、微調NR、信心度  │   ║
║  │ [COMP] GenerateFullStrategy  → dspy.Module (orchestrates above) │   ║
║  │                                                                  │   ║
║  │ ★ comp_strategy_to_dsp_params(): 語意→物理翻譯 (deterministic)  │   ║
║  └──────────┬────────────────────────────────────────────┬──────────┘   ║
║             │                                            │              ║
║  ═══════════╪══ SEMANTIC BOUNDARY (Shannon Cut) ═════════╪══════════   ║
║             │                                            │              ║
║  Layer 5: Scene Understanding   ↑ 回饋路徑               │              ║
║  ┌──────────┴───────────────────┐                        │              ║
║  │ [PRIM] ReasonAboutSceneSig  │                        │              ║
║  │   → strong_lm (大模型)      │                        │              ║
║  │ [ROUTING] scene_router      │ → should_resolve?      │              ║
║  │   → 決定要不要跑矛盾解決   │   + adjusted_confidence │              ║
║  │ [PRIM] resolve_contradictions (conditional)          │              ║
║  │ [COMP] SceneWithHistory     │                        │              ║
║  └──────────┬──────────────────┘                        │              ║
║             │                                            │              ║
║  Layer 4: Perceptual Description                        │              ║
║  ┌──────────┴──────────────────────────────────────┐    │              ║
║  │ [PRIM] DescribeNoiseSig    → fast_lm (小模型)  │    │              ║
║  │ [PRIM] DescribeSpeechSig   → fast_lm           │    │              ║
║  │ [PRIM] DescribeEnvSig      → fast_lm           │    │              ║
║  │     ↓ 三個結果                                  │    │              ║
║  │ [ROUTING] aggregate_router → 動態權重 + 信心度  │    │              ║
║  │ [COMP] FullPerceptualDescription                │    │              ║
║  └──────────┬──────────────────────────────────────┘    │              ║
║             │                                            │              ║
║  ═══════════╪══ SEMANTIC BOUNDARY ═══════════════════════╪══════════   ║
║             │                                            │              ║
║  Layer 3: Acoustic Features (deterministic)             │              ║
║  ┌──────────┴──────────────────────────────────────┐    │              ║
║  │ [PRIM] prim_extract_mfcc()                      │    │              ║
║  │ [PRIM] prim_estimate_snr()                      │    │              ║
║  │ [PRIM] prim_estimate_rt60()                     │    │              ║
║  │ [COMP] comp_extract_full_features()             │    │              ║
║  └──────────┬──────────────────────────────────────┘    │              ║
║             │                                            │              ║
║  Layer 2: Signal Processing (deterministic)             │              ║
║  ┌──────────┴──────────────────────────────────────┐    │              ║
║  │ [PRIM] prim_fft()                               │    │              ║
║  │ [PRIM] prim_estimate_noise_psd()                │    │              ║
║  │ [PRIM] prim_beamform()  ←── DSPParameterSet ────┘    │              ║
║  │ [COMP] comp_spectral_subtract()                 │ ◄──┘              ║
║  └──────────┬──────────────────────────────────────┘                   ║
║             │                                                           ║
║  Layer 1: Physical Sensing (deterministic, hardware)                   ║
║  ┌──────────┴──────────────────────────────────────┐                   ║
║  │ [PRIM] prim_sample_audio()                      │                   ║
║  └─────────────────────────────────────────────────┘                   ║
║                                                                         ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  HARNESS = AcousticSemanticHarness(dspy.Module)                        ║
║  ├─ Semantic Linker:    forward() 中的字串序列化和格式適配              ║
║  ├─ Semantic Runtime:   try/except fallback + dspy.context             ║
║  ├─ Semantic Scheduler: pipeline_router + fast_lm/strong_lm           ║
║  └─ Persistent Store:   scene_history, cached results, preferences     ║
║                                                                         ║
║  GEPA Optimizer (v0.2):                                                ║
║  ├─ Optimizable targets:                                               ║
║  │   ├─ 9 original PRIMs  (can be frozen in Phase 1)                  ║
║  │   └─ 5 new ROUTERs     (★ Method A learnable routing)             ║
║  ├─ Feedback metric:   create_acoustic_feedback_metric()               ║
║  │   └─ Encodes:  物理約束 + 聽力學原則 + routing 合理性              ║
║  ├─ Per-predictor feedback: PRIM + ROUTER 各有獨立回饋                ║
║  └─ Reflection:  用 strong_lm 反思失敗案例，自動改進 prompt           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════
# 提供 nullcontext for Python < 3.10 compatibility
# ═══════════════════════════════════════════════════════════════════════════
from contextlib import nullcontext


# ═══════════════════════════════════════════════════════════════════════════
# 示範執行（不需要 API key — 只展示確定性層）
# ═══════════════════════════════════════════════════════════════════════════

def demo_deterministic_layers():
    """
    展示前三層（確定性）的完整執行——不需要 LLM API
    這些層的輸出就是第四層 LLM Primitive 的輸入
    """
    print("=" * 70)
    print("DEMO: 確定性層（第一～三層）完整執行")
    print("場景：李伯伯，12:15，菜市場魚攤")
    print("=" * 70)
    
    # 第一層
    print("\n[Layer 1] Physical Sensing — prim_sample_audio()")
    signal = prim_sample_audio(duration_ms=32.0, n_channels=2)
    print(f"  Channels: {signal.n_channels}")
    print(f"  Sample rate: {signal.sample_rate} Hz")
    print(f"  Duration: {signal.duration_ms} ms")
    print(f"  Samples per channel: {len(signal.samples[0])}")
    
    # 第二層
    print("\n[Layer 2] Signal Processing")
    spectrum = prim_fft(signal)
    print(f"  [PRIM] FFT: {spectrum['freq_bins']} frequency bins")
    
    noise_psd = prim_estimate_noise_psd(signal)
    print(f"  [PRIM] Noise PSD: estimated {sum(1 for x in noise_psd if x > 0)} active bins")
    
    beamformed = prim_beamform(signal, target_azimuth_deg=0.0)
    print(f"  [PRIM] Beamform(0°): {len(beamformed)} samples")
    
    cleaned = comp_spectral_subtract(signal, noise_psd, alpha=1.0)
    print(f"  [COMP] Spectral subtract: {len(cleaned)} samples")
    
    # 第三層
    print("\n[Layer 3] Acoustic Features")
    features = comp_extract_full_features(signal)
    print(f"  [COMP] Full features extracted:")
    print(f"    SNR: {features.snr_db} dB")
    print(f"    RT60: {features.rt60_s} s")
    print(f"    Active sources: {features.n_active_sources}")
    print(f"    Spectral centroid: {features.spectral_centroid_hz} Hz")
    print(f"    Energy: {features.energy_db} dB SPL")
    print(f"    Temporal pattern: {features.temporal_pattern}")
    print(f"    MFCC summary: {features.mfcc_summary}")
    
    # 第六層確定性 Primitive
    print("\n[Layer 6] Deterministic Primitive — prim_generate_gain_params()")
    audiogram = '{"250":30,"500":35,"1000":40,"2000":50,"4000":60}'
    gain = prim_generate_gain_params(audiogram, "market scene")
    print(f"  Gain per frequency: {gain['gain_per_frequency']}")
    print(f"  Compression ratio: {gain['compression_ratio']}")
    print(f"  Deterministic: {gain['deterministic']}")
    
    print("\n" + "=" * 70)
    print("以上是不需要 LLM 的部分。")
    print("第四～七層需要 LLM API (dspy.configure(lm=...) 後執行)")
    print("GEPA 優化需要呼叫 compile_with_gepa()")
    print("=" * 70)
    
    return features


def demo_full_pipeline():
    """
    展示完整七層管線（需要 OpenAI API key）
    """
    import json

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment. Skipping full pipeline.")
        return

    print("\n" + "=" * 70)
    print("DEMO: 完整七層管線（含 LLM 層）")
    print("場景：李伯伯，12:15，菜市場魚攤")
    print("=" * 70)

    # 配置 LLM
    task_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)
    strong_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)  # 用同一個模型降低成本
    dspy.configure(lm=task_lm)

    # 創建 Harness
    harness = AcousticSemanticHarness(
        fast_lm=task_lm,
        strong_lm=strong_lm
    )

    # 執行完整管線
    print("\n>>> 執行第一次：無使用者動作（自動處理）")
    result = harness(
        user_action="none",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲"
    )

    print(f"\n[Layer 3] Features:")
    print(f"  SNR: {result.features.snr_db} dB")
    print(f"  Energy: {result.features.energy_db} dB SPL")
    print(f"  Active sources: {result.features.n_active_sources}")

    print(f"\n[Layer 4] Perceptual Description (confidence={result.percept.confidence}):")
    print(f"  Noise: {result.percept.noise_description[:200]}")
    print(f"  Speech: {result.percept.speech_description[:200]}")
    print(f"  Environment: {result.percept.environment_description[:200]}")

    print(f"\n[Layer 5] Scene Understanding (confidence={result.scene.confidence}):")
    print(f"  Situation: {result.scene.situation[:300]}")
    print(f"  Challenges: {result.scene.challenges_json[:300]}")

    print(f"\n[Layer 6] Strategy:")
    print(f"  Beam: azimuth={result.strategy.target_azimuth_deg}°, width={result.strategy.beam_width_deg}°")
    print(f"  NR: method={result.strategy.nr_method}, aggressiveness={result.strategy.nr_aggressiveness}")
    print(f"  Gain: {result.strategy.gain_per_frequency}")
    print(f"  Compression: {result.strategy.compression_ratio}")

    print(f"\n[Layer 2] DSP Params (translated from strategy):")
    print(f"  Beam weights: {result.dsp_params.beam_weights}")
    print(f"  Compression: {result.dsp_params.compression_ratio}")
    print(f"  Filter coeffs (first 5): {result.dsp_params.filter_coeffs[:5]}")

    # 第二次：使用者按「不滿意」
    print("\n" + "-" * 70)
    print(">>> 執行第二次：使用者按了「不滿意」按鈕")
    result2 = harness(
        user_action="button_press:dissatisfied",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲"
    )

    print(f"\n[Layer 7] Intent parsed from button press:")
    print(f"  Updated preferences: {json.dumps(result2.current_preferences, ensure_ascii=False, indent=2)}")

    print(f"\n[Layer 6] Updated Strategy:")
    print(f"  Beam: azimuth={result2.strategy.target_azimuth_deg}°, width={result2.strategy.beam_width_deg}°")
    print(f"  NR: method={result2.strategy.nr_method}, aggressiveness={result2.strategy.nr_aggressiveness}")

    print(f"\n[Scene History]:")
    for i, s in enumerate(result2.scene_history):
        print(f"  {i+1}. {s[:100]}")

    print("\n" + "=" * 70)
    print("完整七層管線執行完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo_deterministic_layers()
    print("\n" + ARCHITECTURE_MAP)
    demo_full_pipeline()

    # ★ GEPA 優化
    print("\n\n" + "=" * 70)
    print("接下來執行 GEPA 優化...")
    print("=" * 70)
    try:
        optimized = compile_with_gepa()
    except Exception as e:
        print(f"\nGEPA 執行失敗: {e}")
        import traceback
        traceback.print_exc()
