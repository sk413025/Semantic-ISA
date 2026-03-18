import dspy
from asir.primitives.signal import prim_sample_audio
from asir.harness import AcousticSemanticHarness


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
    ★ GEPA 訓練用 Wrapper（v0.3 多模態版）★

    橋接 training examples 的 input fields（scenario, snr_db, ...）
    和 AcousticSemanticHarness.forward() 的參數。

    GEPA 呼叫 forward(example_inputs) → 這個 wrapper 做轉換 → 呼叫真正的 Harness。

    ★ v0.3: enable_multimodal 控制是否在 GEPA 訓練時使用多模態。
      注意：GEPA + dspy.Image 有已知 memory leak（Issue #8848），
      若記憶體不足可設為 False。
    """
    def __init__(self, fast_lm=None, strong_lm=None,
                 enable_multimodal: bool = True):
        super().__init__()
        self.harness = AcousticSemanticHarness(
            fast_lm=fast_lm,
            strong_lm=strong_lm,
            enable_multimodal=enable_multimodal,
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
