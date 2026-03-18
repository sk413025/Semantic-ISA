"""
Evaluation Examples — 跟 gepa/training.py 的 5 個場景完全分離

這 8 個場景測試系統的泛化能力，涵蓋：
- 不同噪音等級 (安靜 → 極吵)
- 不同空間 (室內/室外/大廳)
- 不同使用者動作 (none / 口語指令 / 按鈕)
- 不同聽力程度 (輕度 → 重度)

=== Example 欄位說明 ===

物理輸入（由 run.py:build_features() 轉成 AcousticFeatures 注入 L4）:
  scenario          str    場景名稱（辨識用，不傳給 LLM）
  snr_db            float  信噪比 (dB)，越低越吵
  rt60_s            float  迴響時間 (秒)，越大越迴響
  n_active_sources  int    活躍聲源數量
  energy_db         float  信號能量 (dB SPL)
  temporal_pattern  str    時域模式: stationary / modulated / impulsive
  user_profile      str    使用者描述（傳給 L4/L5）
  user_action       str    使用者動作（傳給 L7 pipeline router）
  audiogram_json    str    聽力圖 JSON（傳給 L6 NAL-NL2 增益計算）

約束欄位（由 metrics.py 的 check 函數讀取，見 metrics.py 頂部的 mapping 表）:
  expect_noisy           bool       → check_l5: noise_level_consistent（場景噪音描述一致）
  expect_reverberant     bool       → check_l5: reverb_consistent（高迴響場景應提到迴響）
  expect_strong_nr       bool       → check_l6: NR aggressiveness 應 > 0.4
  expect_beam_focus      bool       → check_dsp: beam_width 應 < 90°
  expect_high_gain       bool       → check_dsp: high_gain_for_severe_loss（重度聽損→高增益）
  expect_full_depth      bool       → check_l7: execution_depth 應為 "full"
"""
import dspy


def create_eval_examples():
    """建立 8 個 evaluation examples（跟 training set 完全不同的場景）。"""
    examples = []

    # --- E1: 餐廳多人對話 ---
    examples.append(dspy.Example(
        scenario="restaurant_dinner",
        snr_db=3.0,
        rt60_s=0.7,
        n_active_sources=5,
        energy_db=72.0,
        temporal_pattern="modulated",
        user_profile="65歲女性，雙耳輕度至中度感音神經性聽損",
        user_action="none",
        audiogram_json='{"250":20,"500":25,"1000":30,"2000":40,"4000":50}',
        # 約束
        expect_noisy=True,           # SNR=3 一定吵
        expect_reverberant=False,    # RT60=0.7 中等，不算高迴響
        expect_strong_nr=True,       # 吵 → 應啟用降噪
        expect_beam_focus=True,      # 多人場景應集中波束
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E2: 教堂/禮堂（高迴響） ---
    examples.append(dspy.Example(
        scenario="church_ceremony",
        snr_db=12.0,
        rt60_s=2.5,
        n_active_sources=2,
        energy_db=60.0,
        temporal_pattern="modulated",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
        user_action="none",
        audiogram_json='{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
        expect_noisy=False,          # SNR=12 算可以
        expect_reverberant=True,     # RT60=2.5 非常迴響
        expect_strong_nr=False,      # 不太吵，不需強降噪
        expect_beam_focus=True,      # 講台方向
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E3: 安靜圖書館 ---
    examples.append(dspy.Example(
        scenario="quiet_library",
        snr_db=30.0,
        rt60_s=0.6,
        n_active_sources=1,
        energy_db=40.0,
        temporal_pattern="stationary",
        user_profile="55歲女性，右耳輕度高頻聽損",
        user_action="none",
        audiogram_json='{"250":10,"500":15,"1000":15,"2000":25,"4000":35}',
        expect_noisy=False,          # SNR=30 很安靜
        expect_reverberant=False,
        expect_strong_nr=False,      # 安靜場景不該強降噪
        expect_beam_focus=False,     # 安靜場景不需要集中波束
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E4: 馬路邊打電話 ---
    examples.append(dspy.Example(
        scenario="street_phone_call",
        snr_db=-2.0,
        rt60_s=0.1,
        n_active_sources=7,
        energy_db=80.0,
        temporal_pattern="stationary",
        user_profile="45歲男性，左耳中度感音神經性聽損",
        user_action="focus_front",
        audiogram_json='{"250":25,"500":30,"1000":35,"2000":45,"4000":55}',
        expect_noisy=True,           # SNR=-2 極吵
        expect_reverberant=False,    # 戶外
        expect_strong_nr=True,       # 極吵 → 強降噪
        expect_beam_focus=True,      # focus_front 指令
        expect_preference_updated=False,  # 指令，不是偏好回饋
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E5: 超市購物 ---
    examples.append(dspy.Example(
        scenario="supermarket_shopping",
        snr_db=8.0,
        rt60_s=1.0,
        n_active_sources=4,
        energy_db=68.0,
        temporal_pattern="modulated",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
        user_action="none",
        audiogram_json='{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
        expect_noisy=True,           # SNR=8 中等偏吵
        expect_reverberant=True,     # RT60=1.0 有迴響
        expect_strong_nr=False,      # 中等，不需最強
        expect_beam_focus=False,     # 購物不特別集中
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E6: 車內對話 ---
    examples.append(dspy.Example(
        scenario="car_conversation",
        snr_db=6.0,
        rt60_s=0.15,
        n_active_sources=3,
        energy_db=70.0,
        temporal_pattern="stationary",
        user_profile="68歲男性，雙耳中度感音神經性聽損",
        user_action="none",
        audiogram_json='{"250":25,"500":30,"1000":35,"2000":45,"4000":55}',
        expect_noisy=True,           # SNR=6 引擎+風噪
        expect_reverberant=False,    # 車內空間小
        expect_strong_nr=False,      # 中等降噪
        expect_beam_focus=True,      # 副駕對話方向
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E7: 使用者抱怨「太吵了」---
    examples.append(dspy.Example(
        scenario="noisy_cafe_complaint",
        snr_db=2.0,
        rt60_s=0.6,
        n_active_sources=6,
        energy_db=75.0,
        temporal_pattern="modulated",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
        user_action="太吵了",
        audiogram_json='{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
        expect_noisy=True,
        expect_reverberant=False,
        expect_strong_nr=True,       # 使用者明確抱怨 → 加強降噪
        expect_beam_focus=True,
        expect_full_depth=True,      # 有 user_action → 應跑 full pipeline
        expect_preference_updated=True,  # 抱怨噪音 → 更新偏好
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E8: 重度聽損 + 安靜環境 ---
    examples.append(dspy.Example(
        scenario="severe_loss_quiet_home",
        snr_db=25.0,
        rt60_s=0.4,
        n_active_sources=1,
        energy_db=50.0,
        temporal_pattern="modulated",
        user_profile="80歲女性，雙耳重度感音神經性聽損",
        user_action="none",
        audiogram_json='{"250":50,"500":55,"1000":60,"2000":70,"4000":80}',
        expect_noisy=False,
        expect_reverberant=False,
        expect_strong_nr=False,      # 安靜場景
        expect_beam_focus=False,
        expect_high_gain=True,       # 重度聽損 → 高增益
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E9: 菜市場跟攤販對話（README 旗艦場景）---
    examples.append(dspy.Example(
        scenario="wet_market_vendor",
        snr_db=0.0,
        rt60_s=0.6,
        n_active_sources=8,
        energy_db=78.0,
        temporal_pattern="modulated",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
        user_action="none",
        audiogram_json='{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
        # 約束：README 描述的核心場景
        expect_noisy=True,           # SNR=0 極吵
        expect_reverberant=False,    # 半開放空間，RT60=0.6 中等
        expect_strong_nr=True,       # 嘈雜菜市場 → 強降噪
        expect_beam_focus=True,      # 跟攤販對話 → 集中波束
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    # --- E10: 菜市場 + 使用者抱怨「太悶了」（README L7 回饋場景）---
    examples.append(dspy.Example(
        scenario="market_too_muffled",
        snr_db=0.0,
        rt60_s=0.6,
        n_active_sources=8,
        energy_db=78.0,
        temporal_pattern="modulated",
        user_profile="72歲男性，雙耳中度感音神經性聽損，偏好自然聲",
        user_action="太悶了",
        audiogram_json='{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
        # 約束：使用者抱怨降噪太強 → 系統應降低降噪
        expect_noisy=True,           # 場景仍然吵
        expect_reverberant=False,
        expect_strong_nr=False,      # ★ 使用者說「太悶了」→ NR 不該太強
        expect_beam_focus=True,      # 仍在對話
        expect_full_depth=True,      # 有 user_action → full pipeline
        expect_preference_updated=True,  # 抱怨太悶 → 更新偏好
    ).with_inputs(
        "scenario", "snr_db", "rt60_s", "n_active_sources",
        "energy_db", "temporal_pattern", "user_profile", "user_action",
        "audiogram_json",
    ))

    return examples
