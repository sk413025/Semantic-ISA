"""
Per-Layer Constraint Metrics for ASIR Evaluation

不是 exact match — 是 constraint satisfaction (約束滿足檢查)。
每個 metric 回傳 dict: {check_name: (passed: bool, detail: str)}
"""
import json


def _safe_str(obj, default=""):
    """安全地把任何東西轉成小寫字串。"""
    try:
        return str(obj).lower()
    except Exception:
        return default


def _has_any_keyword(text, keywords):
    """檢查 text 是否包含 keywords 中的任一個。"""
    text_lower = _safe_str(text)
    return any(k.lower() in text_lower for k in keywords)


# ===== L4: Perceptual Description =====

def check_l4_perceptual(example, pred):
    """
    L4 約束檢查：描述是否跟物理特性一致？

    檢查項目:
    - noise_consistent: SNR 低 → 應描述噪音大
    - speech_present: 有聲源 → 應偵測到語音
    - env_reasonable: 環境描述不應為空
    - confidence_calibrated: 信心度在合理範圍 [0.1, 0.95]
    """
    results = {}
    percept = getattr(pred, 'percept', None)

    if percept is None:
        return {"l4_available": (False, "No perceptual description produced")}

    # 噪音描述 vs SNR
    noise_desc = _safe_str(getattr(percept, 'noise_description', ''))
    snr = float(example.snr_db)
    if snr < 5:
        noisy_words = ["loud", "noisy", "high", "significant", "severe",
                       "吵", "嘈雜", "噪音大", "嚴重"]
        has_noisy = _has_any_keyword(noise_desc, noisy_words)
        results["noise_consistent"] = (
            has_noisy,
            f"SNR={snr}dB → noise desc should mention loudness. Got: {noise_desc[:80]}"
        )
    elif snr > 20:
        quiet_words = ["low", "quiet", "minimal", "mild", "slight",
                       "安靜", "輕微", "低"]
        has_quiet = _has_any_keyword(noise_desc, quiet_words)
        results["noise_consistent"] = (
            has_quiet,
            f"SNR={snr}dB → noise should be mild. Got: {noise_desc[:80]}"
        )

    # 語音描述
    speech_desc = _safe_str(getattr(percept, 'speech_description', ''))
    results["speech_present"] = (
        len(speech_desc) > 10,
        f"Speech description length={len(speech_desc)}"
    )

    # 環境描述（composite 輸出是 environment_description）
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
    L5 約束檢查：場景理解是否合理？

    檢查項目:
    - scene_keyword_match: 場景描述包含預期關鍵字
    - scene_confidence: 信心度在合理範圍
    - reverb_awareness: 高 RT60 時應提到迴響
    """
    results = {}
    scene = getattr(pred, 'scene', None)

    if scene is None:
        return {"l5_available": (False, "No scene understanding produced")}

    # 場景關鍵字（composite 輸出是 situation）
    scene_text = _safe_str(getattr(scene, 'situation', ''))
    keywords = getattr(example, 'expect_scene_keywords', [])
    if keywords:
        results["scene_keyword_match"] = (
            _has_any_keyword(scene_text, keywords),
            f"Expected one of {keywords[:4]}... in: {scene_text[:80]}"
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

    # 迴響感知
    rt60 = float(example.rt60_s)
    if rt60 > 1.5:
        reverb_words = ["reverb", "echo", "hall", "迴響", "回音", "殘響"]
        results["reverb_awareness"] = (
            _has_any_keyword(scene_text, reverb_words),
            f"RT60={rt60}s → scene should mention reverb. Got: {scene_text[:80]}"
        )

    return results


# ===== L6: Strategy Generation =====

def check_l6_strategy(example, pred):
    """
    L6 約束檢查：策略是否物理上合理？

    檢查項目:
    - nr_appropriate: 噪音場景 → 降噪, 安靜場景 → 不過度降噪
    - dsp_params_valid: DSP 參數結構完整
    - strategy_reasoning: 策略有推理過程
    """
    results = {}
    strategy = getattr(pred, 'strategy', None)
    dsp = getattr(pred, 'dsp_params', None)

    if strategy is None and dsp is None:
        return {"l6_available": (False, "No strategy or DSP params produced")}

    # 降噪強度 vs 場景吵雜程度
    expect_strong_nr = getattr(example, 'expect_strong_nr', None)
    if expect_strong_nr is not None and strategy is not None:
        nr_agg = getattr(strategy, 'nr_aggressiveness', None)
        if nr_agg is None:
            nr_agg = getattr(strategy, 'adjusted_nr_aggressiveness', None)
        if nr_agg is not None:
            try:
                agg = float(nr_agg)
                if expect_strong_nr:
                    results["nr_appropriate"] = (
                        agg > 0.4,
                        f"Noisy scene but NR aggressiveness={agg:.2f}, expected > 0.4"
                    )
                else:
                    results["nr_appropriate"] = (
                        agg < 0.7,
                        f"Quiet scene but NR aggressiveness={agg:.2f}, expected < 0.7"
                    )
            except (ValueError, TypeError):
                pass

    # DSP 參數結構
    if dsp is not None:
        has_beam = hasattr(dsp, 'beam_weights') and dsp.beam_weights is not None
        has_mask = hasattr(dsp, 'noise_mask') and dsp.noise_mask is not None
        has_filter = hasattr(dsp, 'filter_coeffs') and dsp.filter_coeffs is not None
        results["dsp_params_valid"] = (
            has_beam and has_mask and has_filter,
            f"beam={has_beam}, mask={has_mask}, filter={has_filter}"
        )

    # 策略推理
    if strategy is not None:
        reasoning = _safe_str(getattr(strategy, 'combined_reasoning', ''))
        results["strategy_reasoning"] = (
            len(reasoning) > 30,
            f"Reasoning length={len(reasoning)}"
        )

    return results


# ===== L7: Intent & Preference =====

def check_l7_intent(example, pred):
    """
    L7 約束檢查：使用者動作是否被正確處理？

    檢查項目:
    - action_respected: 有 user_action 時應觸發 full pipeline
    - depth_appropriate: execution_depth 合理
    """
    results = {}

    # execution depth
    depth = _safe_str(getattr(pred, 'execution_depth', 'unknown'))
    expect_full = getattr(example, 'expect_full_depth', False)
    user_action = str(getattr(example, 'user_action', 'none'))

    if expect_full or (user_action != 'none'):
        results["action_respected"] = (
            depth == 'full',
            f"User action='{user_action}' but depth='{depth}', expected 'full'"
        )

    if depth in ('fast', 'medium', 'full'):
        results["depth_valid"] = (True, f"depth='{depth}'")
    else:
        results["depth_valid"] = (False, f"depth='{depth}' not in (fast/medium/full)")

    return results


# ===== Pipeline: End-to-End =====

def check_pipeline(example, pred):
    """
    全流程約束檢查：把 L4-L7 的結果匯總。

    額外檢查:
    - output_complete: 有 DSP 參數輸出
    - no_crash: 執行沒有 crash
    """
    results = {}

    # 有 DSP 輸出
    dsp = getattr(pred, 'dsp_params', None)
    results["output_complete"] = (
        dsp is not None,
        "DSP params present" if dsp else "No DSP params"
    )

    # 匯總各層
    results.update(check_l4_perceptual(example, pred))
    results.update(check_l5_scene(example, pred))
    results.update(check_l6_strategy(example, pred))
    results.update(check_l7_intent(example, pred))

    return results


def compute_score(check_results):
    """從 check_results dict 算出 0-1 分數。"""
    if not check_results:
        return 0.0
    passed = sum(1 for v, _ in check_results.values() if v)
    return passed / len(check_results)
