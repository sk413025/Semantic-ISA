import json
from typing import Optional
from contextlib import nullcontext
import numpy as np
import dspy
from asir.types import RawSignal, DSPParameterSet
from asir.primitives.signal import prim_sample_audio, prim_beamform
from asir.primitives.features import comp_extract_full_features
from asir.primitives.intent import ParseIntentSig, UpdatePreferencesSig
from asir.composites import (
    FullPerceptualDescription, SceneWithHistory,
    GenerateFullStrategy, comp_strategy_to_dsp_params,
)
from asir.multimodal import raw_signal_to_audio, generate_spectrogram_image
from asir.routing.pipeline import PipelineRoutingSig


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

    # ★ Phase 5: LM 多模態能力資料庫（用於自動偵測）
    AUDIO_CAPABLE_LMS = {
        "gpt-4o-audio-preview", "gpt-4o-mini-audio-preview",
        "gemini-2.0-flash", "gemini-2.5-pro", "gemini-3-pro",
        "gemini-2.0-flash-exp",
    }
    VISION_CAPABLE_LMS = {
        "gpt-4o", "gpt-4o-mini", "gpt-4o-audio-preview",
        "gpt-4o-mini-audio-preview",
        "gemini-2.0-flash", "gemini-2.5-pro", "gemini-3-pro",
        "gemini-2.0-flash-exp",
        "claude-3-5-sonnet", "claude-3-opus", "claude-4-sonnet",
        "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-latest",
        "claude-sonnet-4-20250514",
    }

    def __init__(self,
                 fast_lm=None,    # 第四層用的小模型（低延遲）
                 strong_lm=None,  # 第五、六層用的大模型（強推理）
                 enable_multimodal: bool = True,  # ★ Phase 3: 多模態開關
                 ):
        super().__init__()

        # === Semantic Scheduler: 不同層用不同模型 ===
        self.fast_lm = fast_lm
        self.strong_lm = strong_lm
        self.enable_multimodal = enable_multimodal

        # ★ Phase 5: 自動偵測 LM 多模態能力
        self._fast_lm_supports_audio = self._check_lm_capability(fast_lm, 'audio')
        self._fast_lm_supports_vision = self._check_lm_capability(fast_lm, 'vision')
        self._strong_lm_supports_vision = self._check_lm_capability(strong_lm, 'vision')

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

    @classmethod
    def _check_lm_capability(cls, lm, capability: str) -> bool:
        """
        ★ Phase 5: 偵測 LM 是否支援音訊/視覺輸入。
        透過 model name 比對已知的能力資料庫。
        """
        if lm is None:
            return False
        try:
            model_name = str(getattr(lm, 'model', ''))
            # 移除 provider prefix（如 "openai/" → ""）
            short_name = model_name.split('/')[-1] if '/' in model_name else model_name
            if capability == 'audio':
                return any(k in short_name for k in cls.AUDIO_CAPABLE_LMS)
            elif capability == 'vision':
                return any(k in short_name for k in cls.VISION_CAPABLE_LMS)
        except Exception:
            pass
        return False

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

        # ═══ Phase 2 + 5: 多模態資料生成（成本感知）═══
        # ★ 根據 execution_depth 決定多模態預算：
        #   fast → 不生成（已在上面 return）
        #   medium → 只生成頻譜圖（Image，便宜）
        #   full → 頻譜圖 + 音訊（Audio，貴）
        audio_clip = None
        spectrogram_img = None

        if self.enable_multimodal:
            # medium + full 都生成頻譜圖
            if self._fast_lm_supports_vision or self._strong_lm_supports_vision:
                spectrogram_img = generate_spectrogram_image(
                    raw_signal, title=f"Frame (SNR≈{features.snr_db}dB)"
                )
            # 只有 full 才生成音訊（Phase 5 成本控制）
            if depth == "full" and self._fast_lm_supports_audio:
                audio_clip = raw_signal_to_audio(raw_signal)

        # ═══ 第四層：感知描述（LLM — 用小模型 + 多模態）═══
        fast_ctx = dspy.context(lm=self.fast_lm) if self.fast_lm else nullcontext()
        try:
            with fast_ctx:
                percept = self.perceptual_desc(
                    acoustic_features=features,
                    user_context=user_profile,
                    audio_clip=audio_clip,        # ★ Phase 3
                    spectrogram=spectrogram_img,   # ★ Phase 3
                )
        except Exception as e:
            percept = dspy.Prediction(
                noise_description='[{"type":"unknown","direction":"unknown","temporal":"unknown","severity":"moderate"}]',
                speech_description="Speakers: 1, Target: front, Intelligibility: unknown",
                environment_description="Type: unknown, Character: noisy",
                confidence=0.3
            )

        # ═══ 第五層：場景理解（LLM — 用大模型 + 頻譜圖）═══
        strong_ctx = dspy.context(lm=self.strong_lm) if self.strong_lm else nullcontext()
        # ★ Phase 3: L5 只用頻譜圖（不傳音訊，省成本）
        l5_spectrogram = spectrogram_img if self._strong_lm_supports_vision else None
        try:
            with strong_ctx:
                scene = self.scene_understanding(
                    percept=percept,
                    user_profile=user_profile,
                    recent_scenes=self.scene_history,
                    spectrogram=l5_spectrogram,  # ★ Phase 3
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

            # ★ v0.8: 移除硬編碼英文 gate（"dissatisfied"/"satisfied"）。
            # UpdatePreferencesSig 是 LLM ChainOfThought，
            # 語意決定是否需要更新偏好，取代 keyword matching。
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
            except (json.JSONDecodeError, TypeError, ValueError):
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
