"""
Per-Layer Constraint Metrics for ASIR Evaluation

不是 exact match — 是 constraint satisfaction (約束滿足檢查)。
每個 check 函數回傳 dict: {check_name: (passed: bool, detail: str)}

=== Constraint → Check Mapping ===

example fields           → check function          → what it tests
─────────────────────────────────────────────────────────────────────
snr_db                   → check_l4_perceptual      → noise_consistent: 低 SNR 描述應提到噪音大
(always)                 → check_l4_perceptual      → speech_present: 語音描述不為空
(always)                 → check_l4_perceptual      → env_reasonable: 環境描述不為空
(always)                 → check_l4_perceptual      → confidence_calibrated: [0.1, 0.95]

expect_noisy             → check_l5_scene           → noise_level_consistent: 場景噪音描述一致
expect_reverberant       → check_l5_scene           → reverb_consistent: 高迴響場景應提到迴響
n_active_sources         → check_l5_scene           → multi_source_aware: 多聲源→應提到多人/多聲源
(always)                 → check_l5_scene           → scene_confidence: [0.1, 0.95]

expect_strong_nr         → check_l6_strategy        → nr_matches_scene: NR 強度 vs 噪音程度
(always)                 → check_l6_strategy        → strategy_has_reasoning: 策略有推理過程

audiogram_json           → check_dsp_output         → gain_matches_loss: 高頻聽損→高頻增益大
expect_high_gain         → check_dsp_output         → high_gain_for_severe_loss: 重度聽損→高增益
snr_db                   → check_dsp_output         → nr_matches_noise: SNR 低→NR 強
expect_beam_focus        → check_dsp_output         → beam_appropriate: 預期聚焦→窄波束
(always)                 → check_dsp_output         → compression_reasonable: [1.0, 4.0]
(always)                 → check_dsp_output         → dsp_structure_complete: 三個欄位都有值

user_action              → check_l7_routing         → action_triggers_full: 有動作→depth=full
(always)                 → check_l7_routing         → depth_valid: fast/medium/full

=== Field Name Coupling ===

check 函數存取的 prediction 欄位名稱來自 composites 的 output：
- pred.percept.noise_description      ← FullPerceptualDescription
- pred.percept.speech_description     ← FullPerceptualDescription
- pred.percept.environment_description ← FullPerceptualDescription
- pred.percept.confidence             ← FullPerceptualDescription
- pred.scene.situation                ← SceneWithHistory
- pred.scene.confidence               ← SceneWithHistory
- pred.strategy.nr_aggressiveness     ← GenerateFullStrategy
- pred.strategy.beam_width_deg        ← GenerateFullStrategy
- pred.strategy.gain_per_frequency    ← GenerateFullStrategy
- pred.strategy.compression_ratio     ← GenerateFullStrategy
- pred.strategy.combined_reasoning    ← GenerateFullStrategy
- pred.dsp_params.beam_weights        ← comp_strategy_to_dsp_params
- pred.dsp_params.noise_mask          ← comp_strategy_to_dsp_params
- pred.dsp_params.filter_coeffs       ← comp_strategy_to_dsp_params
- pred.execution_depth                ← PipelineRoutingSig
"""
import json


def _safe_str(obj, default=""):
    try:
        return str(obj).lower()
    except Exception:
        return default


def _has_any_keyword(text, keywords):
    text_lower = _safe_str(text)
    return any(k.lower() in text_lower for k in keywords)


# ===== L4: Perceptual Description =====

def check_l4_perceptual(example, pred):
    """
    L4 約束：描述是否跟注入的 AcousticFeatures 一致？

    因為 features 是直接從 example 建構的，LLM 看到的 SNR 就是 example.snr_db，
    所以 noise description 應該跟 SNR 值一致。
    """
    results = {}
    percept = getattr(pred, 'percept', None)

    if percept is None:
        return {"l4_available": (False, "No perceptual description produced")}

    # 噪音描述 vs SNR（LLM 直接看到 SNR=3.0dB，應該描述為吵）
    noise_desc = _safe_str(getattr(percept, 'noise_description', ''))
    snr = float(example.snr_db)
    if snr < 5:
        noisy_words = ["loud", "noisy", "high", "significant", "severe",
                       "multiple", "strong", "intense", "heavy",
                       "吵", "嘈雜", "噪音大", "嚴重"]
        results["noise_consistent"] = (
            _has_any_keyword(noise_desc, noisy_words),
            f"SNR={snr}dB → noise desc should mention loudness. Got: {noise_desc[:100]}"
        )
    elif snr > 20:
        quiet_words = ["low", "quiet", "minimal", "mild", "slight", "soft",
                       "安靜", "輕微", "低"]
        results["noise_consistent"] = (
            _has_any_keyword(noise_desc, quiet_words),
            f"SNR={snr}dB → noise should be mild. Got: {noise_desc[:100]}"
        )

    # 語音描述
    speech_desc = _safe_str(getattr(percept, 'speech_description', ''))
    results["speech_present"] = (
        len(speech_desc) > 10,
        f"Speech description length={len(speech_desc)}"
    )

    # 環境描述
    env_desc = _safe_str(getattr(percept, 'environment_description', ''))
    results["env_reasonable"] = (
        len(env_desc) > 10,
        f"Env description length={len(env_desc)}"
    )

    # 信心度
    conf = getattr(percept, 'confidence', None)
    if conf is not None:
        try:
            c = float(conf)
            results["confidence_calibrated"] = (
                0.1 <= c <= 0.95,
                f"L4 confidence={c:.2f}"
            )
        except (ValueError, TypeError):
            results["confidence_calibrated"] = (False, f"Cannot parse confidence: {conf}")

    return results


# ===== L5: Scene Understanding =====

def check_l5_scene(example, pred):
    """
    L5 約束：場景理解是否跟聲學特性一致？

    不測場景名稱猜對沒有（LLM 無法從數字推出「餐廳」vs「酒吧」），
    而是測場景描述是否反映了聲學特性（吵/安靜、迴響大/小、多聲源/單人）。
    這些都是 LLM 直接看到的數字，可以合理推論的。
    """
    results = {}
    scene = getattr(pred, 'scene', None)

    if scene is None:
        return {"l5_available": (False, "No scene understanding produced")}

    scene_text = _safe_str(getattr(scene, 'situation', ''))

    # 噪音程度一致性：LLM 看到 SNR=3dB + 5 聲源 → 描述應反映吵雜
    expect_noisy = getattr(example, 'expect_noisy', None)
    if expect_noisy is not None:
        if expect_noisy:
            noisy_words = ["noisy", "loud", "noise", "crowded", "busy", "chaotic",
                           "multiple speaker", "multi-talker", "challenging",
                           "吵", "嘈雜", "噪音", "多人", "擁擠", "嘈"]
            results["noise_level_consistent"] = (
                _has_any_keyword(scene_text, noisy_words),
                f"SNR={example.snr_db}dB, {example.n_active_sources} sources "
                f"→ scene should describe noisy conditions. Got: {scene_text[:100]}"
            )
        else:
            # 安靜場景：不該被描述為極度嘈雜
            extreme_noise = ["extremely noisy", "deafening", "unbearable",
                             "extremely loud", "overwhelming noise"]
            results["noise_level_consistent"] = (
                not _has_any_keyword(scene_text, extreme_noise),
                f"SNR={example.snr_db}dB → shouldn't describe as extremely noisy. "
                f"Got: {scene_text[:100]}"
            )

    # 迴響一致性：LLM 看到 RT60=2.5s → 描述應提到迴響
    expect_reverberant = getattr(example, 'expect_reverberant', None)
    if expect_reverberant is not None and expect_reverberant:
        reverb_words = ["reverb", "echo", "hall", "resonan", "large space",
                        "spacious", "cathedral", "迴響", "回音", "殘響", "混響", "空間大"]
        results["reverb_consistent"] = (
            _has_any_keyword(scene_text, reverb_words),
            f"RT60={example.rt60_s}s → scene should mention reverb. "
            f"Got: {scene_text[:100]}"
        )

    # 多聲源感知：LLM 看到 n_active_sources=5 → 應提到多個聲源
    n_sources = int(example.n_active_sources)
    if n_sources >= 4:
        multi_words = ["multiple", "several", "many", "group", "crowd",
                       "conversation", "speaker", "talker", "voices", "busy",
                       "多人", "多個", "群", "對話", "繁忙", "熱鬧"]
        results["multi_source_aware"] = (
            _has_any_keyword(scene_text, multi_words),
            f"n_sources={n_sources} → scene should mention multiple sources. "
            f"Got: {scene_text[:100]}"
        )

    # 信心度
    conf = getattr(scene, 'confidence', None)
    if conf is not None:
        try:
            c = float(conf)
            results["scene_confidence"] = (
                0.1 <= c <= 0.95,
                f"L5 confidence={c:.2f}"
            )
        except (ValueError, TypeError):
            pass

    return results


# ===== L6: Strategy Reasoning =====

def check_l6_strategy(example, pred):
    """
    L6 約束：策略推理是否合理？

    只檢查 LLM 的推理品質和 NR aggressiveness 決策。
    DSP 參數的物理正確性由 check_dsp_output 檢查。
    """
    results = {}
    strategy = getattr(pred, 'strategy', None)

    if strategy is None:
        return {"l6_available": (False, "No strategy produced")}

    # NR aggressiveness vs 場景噪音
    expect_strong_nr = getattr(example, 'expect_strong_nr', None)
    nr_agg = getattr(strategy, 'nr_aggressiveness', None)
    if nr_agg is None:
        nr_agg = getattr(strategy, 'adjusted_nr_aggressiveness', None)

    if expect_strong_nr is not None and nr_agg is not None:
        try:
            agg = float(nr_agg)
            if expect_strong_nr:
                results["nr_matches_scene"] = (
                    agg > 0.4,
                    f"Noisy scene → NR agg={agg:.2f}, expected > 0.4"
                )
            else:
                results["nr_matches_scene"] = (
                    agg < 0.7,
                    f"Quiet scene → NR agg={agg:.2f}, expected < 0.7"
                )
        except (ValueError, TypeError):
            pass

    # 策略推理長度（應有實質推理，不只是 template）
    reasoning = _safe_str(getattr(strategy, 'combined_reasoning', ''))
    results["strategy_has_reasoning"] = (
        len(reasoning) > 50,
        f"Reasoning length={len(reasoning)}"
    )

    return results


# ===== DSP Output: Physical Constraints (核心測試) =====

def check_dsp_output(example, pred):
    """
    DSP 輸出物理約束 — 這是最直接反映專案目標的測試。

    專案目標：給定聲學場景 + 聽損程度 → 產生合理的 DSP 參數。
    這個 check 直接驗證輸出 DSP 參數是否對這個場景/聽損有意義。
    """
    results = {}
    strategy = getattr(pred, 'strategy', None)
    dsp = getattr(pred, 'dsp_params', None)

    if strategy is None:
        return {"dsp_available": (False, "No strategy output")}

    # 1. Gain 跟聽損程度對應
    #    聽損越嚴重的頻率 → 增益應越大（NAL-NL2 的核心邏輯）
    gain_gpf = getattr(strategy, 'gain_per_frequency', None)
    audiogram_str = getattr(example, 'audiogram_json', None)
    if gain_gpf and audiogram_str:
        try:
            audiogram = json.loads(str(audiogram_str))
            max_loss_freq = max(audiogram, key=lambda k: audiogram[k])
            min_loss_freq = min(audiogram, key=lambda k: audiogram[k])
            if audiogram[max_loss_freq] != audiogram[min_loss_freq]:
                results["gain_matches_loss"] = (
                    gain_gpf[max_loss_freq] > gain_gpf[min_loss_freq],
                    f"gain@{max_loss_freq}Hz(loss={audiogram[max_loss_freq]}dB)="
                    f"{gain_gpf[max_loss_freq]:.1f} should > "
                    f"gain@{min_loss_freq}Hz(loss={audiogram[min_loss_freq]}dB)="
                    f"{gain_gpf[min_loss_freq]:.1f}"
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # 2. 重度聽損 → 高增益（expect_high_gain）
    expect_high_gain = getattr(example, 'expect_high_gain', None)
    if expect_high_gain and gain_gpf:
        try:
            max_gain = max(float(v) for v in gain_gpf.values())
            results["high_gain_for_severe_loss"] = (
                max_gain >= 20,
                f"Severe hearing loss → max gain should be >= 20dB. Got: {max_gain:.1f}dB"
            )
        except (ValueError, TypeError):
            pass

    # 3. NR aggressiveness vs 噪音程度
    nr_agg = getattr(strategy, 'nr_aggressiveness', None)
    if nr_agg is not None:
        snr = float(example.snr_db)
        try:
            agg = float(nr_agg)
            if snr < 5:
                results["nr_matches_noise"] = (
                    agg >= 0.3,
                    f"SNR={snr}dB → NR agg={agg:.2f}, expected >= 0.3"
                )
            elif snr > 20:
                results["nr_matches_noise"] = (
                    agg <= 0.6,
                    f"SNR={snr}dB → NR agg={agg:.2f}, expected <= 0.6"
                )
        except (ValueError, TypeError):
            pass

    # 4. Beam focus 是否恰當
    beam_width = getattr(strategy, 'beam_width_deg', None)
    expect_focus = getattr(example, 'expect_beam_focus', None)
    if beam_width is not None and expect_focus is not None:
        try:
            bw = float(beam_width)
            if expect_focus:
                results["beam_appropriate"] = (
                    bw < 90,
                    f"Expected focused beam (< 90°) but width={bw:.0f}°"
                )
            else:
                results["beam_appropriate"] = (
                    bw >= 45,
                    f"Expected wide beam (>= 45°) but width={bw:.0f}°"
                )
        except (ValueError, TypeError):
            pass

    # 5. Compression ratio 在合理範圍
    cr = getattr(strategy, 'compression_ratio', None)
    if cr is not None:
        try:
            results["compression_reasonable"] = (
                1.0 <= float(cr) <= 4.0,
                f"compression_ratio={float(cr):.2f}"
            )
        except (ValueError, TypeError):
            pass

    # 6. DSP 參數結構完整（beam_weights + noise_mask + filter_coeffs）
    if dsp is not None:
        has_beam = hasattr(dsp, 'beam_weights') and dsp.beam_weights is not None
        has_mask = hasattr(dsp, 'noise_mask') and dsp.noise_mask is not None
        has_filter = hasattr(dsp, 'filter_coeffs') and dsp.filter_coeffs is not None
        results["dsp_structure_complete"] = (
            has_beam and has_mask and has_filter,
            f"beam={has_beam}, mask={has_mask}, filter={has_filter}"
        )

    return results


# ===== L7: Pipeline Routing =====

def check_l7_routing(example, pred):
    """
    L7 約束：Pipeline Router 的決策是否合理？

    使用者有動作時 → 應觸發 full pipeline。
    execution_depth 應是 fast/medium/full 之一。
    """
    results = {}
    depth = _safe_str(getattr(pred, 'execution_depth', 'unknown'))
    user_action = str(getattr(example, 'user_action', 'none'))

    if user_action != 'none':
        results["action_triggers_full"] = (
            depth == 'full',
            f"User action='{user_action}' but depth='{depth}', expected 'full'"
        )

    if depth in ('fast', 'medium', 'full'):
        results["depth_valid"] = (True, f"depth='{depth}'")
    else:
        results["depth_valid"] = (False, f"depth='{depth}' not in (fast/medium/full)")

    return results


# ===== Scoring =====

def compute_score(check_results):
    """從 check_results dict 算出 0-1 分數。"""
    if not check_results:
        return 0.0
    passed = sum(1 for v, _ in check_results.values() if v)
    return passed / len(check_results)
