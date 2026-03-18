import json
import numpy as np
import dspy


class GenerateBeamformingParamsSig(dspy.Signature):
    """
    [PRIM] 第六層：波束成形參數生成
    BACKEND: LLM + physics_constraints
    RELIABILITY: target_direction_error <= 15° in 90% of cases
    CONSTRAINT: beam_width >= 20°（物理約束：麥克風陣列最小波束寬度）

    ★ Primitive，因為波束參數需要同時考慮
      語義（目標說話者方向）和物理（陣列幾何約束）。

    波束寬度選擇指南：
    - 安靜/環境聆聽（SNR>15dB, 1-2聲源）→ 寬波束 90-360° (開放聆聽)
    - 中等噪音/多聲源（SNR 5-15dB）→ 中等波束 45-90°
    - 高噪音/需聚焦（SNR<5dB 或使用者要求聚焦）→ 窄波束 20-45°
    """
    scene_understanding: str = dspy.InputField(desc="第五層場景理解")
    mic_geometry: str = dspy.InputField(
        desc="麥克風陣列幾何：間距、數量、排列方式"
    )

    target_azimuth_deg: float = dspy.OutputField(desc="目標方位角（度）[-180, 180]")
    beam_width_deg: float = dspy.OutputField(
        desc="波束寬度（度）[20, 360]。安靜場景用寬波束(90-360°)，"
             "噪音場景用窄波束(20-45°)，中等場景用中等波束(45-90°)"
    )
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
