import json
from contextlib import nullcontext
from typing import Optional

import dspy
import numpy as np

from asir.composites import (
    FullPerceptualDescription,
    GenerateFullStrategy,
    SceneWithHistory,
    comp_strategy_to_dsp_params,
)
from asir.multimodal import generate_spectrogram_image, raw_signal_to_audio
from asir.primitives.features import comp_extract_full_features
from asir.primitives.intent import ParseIntentSig, UpdatePreferencesSig
from asir.primitives.signal import prim_beamform, prim_sample_audio
from asir.routing.pipeline import PipelineRoutingSig
from asir.types import DSPParameterSet, RawSignal


class AcousticSemanticHarness(dspy.Module):
    """
    Top-level composite: the full seven-layer ASIR pipeline.

    [COMP] intent_aware_strategy
    = pipeline_router -> choose execution_depth
      -> Layer 1 (sampling) -> Layer 2 (DSP) -> Layer 3 (features)
      -> Layer 4 (percept) -> Layer 5 (scene) -> Layer 6 (strategy)
      -> translate back to Layer 2 DSP parameters

    The harness ties together four subsystem roles:
    1. Semantic Linker: format adaptation between layer outputs and inputs
    2. Semantic Runtime: context-window handling and recovery behavior
    3. Semantic Scheduler: model selection and latency budgeting
    4. Persistent Store: long-term preferences, scene history, and cached results
    """

    # Known multimodal model capabilities for auto-detection.
    AUDIO_CAPABLE_LMS = {
        "gpt-4o-audio-preview",
        "gpt-4o-mini-audio-preview",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "gemini-3-pro",
        "gemini-2.0-flash-exp",
    }
    VISION_CAPABLE_LMS = {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-audio-preview",
        "gpt-4o-mini-audio-preview",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "gemini-3-pro",
        "gemini-2.0-flash-exp",
        "claude-3-5-sonnet",
        "claude-3-opus",
        "claude-4-sonnet",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest",
        "claude-sonnet-4-20250514",
    }

    def __init__(
        self,
        fast_lm=None,  # Smaller low-latency model for Layer 4.
        strong_lm=None,  # Stronger reasoning model for Layers 5 and 6.
        enable_multimodal: bool = True,  # Global multimodal switch.
    ):
        super().__init__()

        # Semantic scheduler: use different models at different layers.
        self.fast_lm = fast_lm
        self.strong_lm = strong_lm
        self.enable_multimodal = enable_multimodal

        # Auto-detect multimodal capability from model names.
        self._fast_lm_supports_audio = self._check_lm_capability(fast_lm, "audio")
        self._fast_lm_supports_vision = self._check_lm_capability(fast_lm, "vision")
        self._strong_lm_supports_vision = self._check_lm_capability(strong_lm, "vision")

        # Semantic modules, all of which are GEPA-optimizable.
        self.perceptual_desc = FullPerceptualDescription()
        self.scene_understanding = SceneWithHistory()
        self.strategy_gen = GenerateFullStrategy()
        self.parse_intent = dspy.ChainOfThought(ParseIntentSig)
        self.update_prefs = dspy.ChainOfThought(UpdatePreferencesSig)
        self.pipeline_router = dspy.ChainOfThought(PipelineRoutingSig)

        # Persistent store.
        self.scene_history: list[str] = []
        self.feedback_history: list[str] = []
        self.current_preferences = {
            "noise_tolerance": "medium",
            "processing_preference": "natural",
            "environment_awareness": "moderate",
            "known_situations": [
                "wet market: enhance the front, preserve environmental awareness"
            ],
        }
        self.current_dsp_params: Optional[DSPParameterSet] = None

        # Cached intermediate results used by fast / medium execution paths.
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
        Detect whether an LM supports audio or vision input by checking its name
        against a small capability registry.
        """

        if lm is None:
            return False
        try:
            model_name = str(getattr(lm, "model", ""))
            short_name = model_name.split("/")[-1] if "/" in model_name else model_name
            if capability == "audio":
                return any(k in short_name for k in cls.AUDIO_CAPABLE_LMS)
            if capability == "vision":
                return any(k in short_name for k in cls.VISION_CAPABLE_LMS)
        except Exception:
            pass
        return False

    def _estimate_signal_change(self, raw_signal: RawSignal) -> float:
        """Estimate the frame-to-frame signal-change magnitude in [0, 1]."""

        current_energy = np.mean(np.abs(raw_signal.samples[0]))
        if self._last_signal_energy == 0:
            self._last_signal_energy = current_energy
            return 1.0  # First frame: treat it as a large change and force full execution.
        delta = abs(current_energy - self._last_signal_energy)
        normalized = min(1.0, delta / (self._last_signal_energy + 1e-8))
        self._last_signal_energy = current_energy
        return float(normalized)

    def forward(
        self,
        raw_signal: Optional[RawSignal] = None,
        user_action: str = "none",
        audiogram_json: str = '{"250":30,"500":35,"1000":40,"2000":50,"4000":60}',
        user_profile: str = (
            "72-year-old man with bilateral moderate sensorineural hearing loss "
            "who prefers natural sound"
        ),
    ) -> dspy.Prediction:
        """
        Run one full control-loop step.

        Multi-rate execution is chosen dynamically by the pipeline router:
        - fast (<10 ms): L1->L2 only, using cached params
        - medium (<500 ms): L1->L5, refreshing the scene but reusing cached strategy
        - full (>500 ms): L1->L7, refreshing the entire stack
        """

        # Layer 1: physical sensing. This always runs.
        if raw_signal is None:
            raw_signal = prim_sample_audio(duration_ms=32.0)

        # Layer 2: fast DSP path with cached parameters.
        if self.current_dsp_params is not None:
            processed = prim_beamform(raw_signal, self.current_dsp_params.beam_weights[0] * 30)
        else:
            processed = raw_signal.samples[0]
        _ = processed

        # Pipeline router: decide how deep this frame should run.
        signal_change = self._estimate_signal_change(raw_signal)
        routing = self.pipeline_router(
            signal_change_magnitude=signal_change,
            last_scene_confidence=self._last_scene_conf,
            last_strategy_confidence=self._last_strategy_conf,
            user_action=user_action,
            frames_since_full_update=self._frames_since_full,
        )

        depth = str(routing.execution_depth).strip().lower()

        # Fast path: return cached results immediately.
        if (
            depth == "fast"
            and self.current_dsp_params is not None
            and self._cached_percept is not None
        ):
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
                routing_reasoning=routing.routing_reasoning,
            )

        # Layer 3: deterministic feature extraction.
        features = comp_extract_full_features(raw_signal)

        # Phase 2 + 5: cost-aware multimodal artifact generation.
        # fast   -> nothing (already returned above)
        # medium -> spectrogram only
        # full   -> spectrogram + audio
        audio_clip = None
        spectrogram_img = None

        if self.enable_multimodal:
            if self._fast_lm_supports_vision or self._strong_lm_supports_vision:
                spectrogram_img = generate_spectrogram_image(
                    raw_signal,
                    title=f"Frame (SNR≈{features.snr_db}dB)",
                )
            if depth == "full" and self._fast_lm_supports_audio:
                audio_clip = raw_signal_to_audio(raw_signal)

        # Layer 4: perceptual description using the fast model.
        fast_ctx = dspy.context(lm=self.fast_lm) if self.fast_lm else nullcontext()
        try:
            with fast_ctx:
                percept = self.perceptual_desc(
                    acoustic_features=features,
                    user_context=user_profile,
                    audio_clip=audio_clip,
                    spectrogram=spectrogram_img,
                )
        except Exception:
            percept = dspy.Prediction(
                noise_description='[{"type":"unknown","direction":"unknown","temporal":"unknown","severity":"moderate"}]',
                speech_description="Speakers: 1, Target: front, Intelligibility: unknown",
                environment_description="Type: unknown, Character: noisy",
                confidence=0.3,
            )

        # Layer 5: scene understanding using the strong model and, when available, a spectrogram.
        strong_ctx = dspy.context(lm=self.strong_lm) if self.strong_lm else nullcontext()
        l5_spectrogram = spectrogram_img if self._strong_lm_supports_vision else None
        try:
            with strong_ctx:
                scene = self.scene_understanding(
                    percept=percept,
                    user_profile=user_profile,
                    recent_scenes=self.scene_history,
                    spectrogram=l5_spectrogram,
                )
        except Exception:
            scene = dspy.Prediction(
                situation="Unable to determine scene",
                challenges_json="[]",
                preservation_notes_json="[]",
                confidence=0.2,
            )

        # Persistent store: record scene history and cache.
        self.scene_history.append(scene.situation)
        if len(self.scene_history) > 20:
            self.scene_history = self.scene_history[-20:]
        self._last_scene_conf = float(scene.confidence)
        self._cached_percept = percept
        self._cached_scene = scene

        # Medium path: refresh the scene but reuse cached strategy.
        if (
            depth == "medium"
            and not routing.force_strategy_update
            and self._cached_strategy is not None
            and self.current_dsp_params is not None
        ):
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
                routing_reasoning=routing.routing_reasoning,
            )

        # Full path: continue through L6 + L7.
        if user_action != "none":
            _intent = self.parse_intent(
                user_action=user_action,
                current_scene=scene.situation,
                user_history=json.dumps(self.current_preferences, ensure_ascii=False),
            )
            _ = _intent

            # v0.8 removed hardcoded English keyword gates such as
            # "dissatisfied" / "satisfied". UpdatePreferencesSig now decides
            # semantically whether preferences should change.
            pref_update = self.update_prefs(
                current_preferences=json.dumps(self.current_preferences, ensure_ascii=False),
                user_feedback=user_action,
                current_scene=scene.situation,
                feedback_history=" | ".join(self.feedback_history[-5:]),
            )
            self.feedback_history.append(f"{user_action} in {scene.situation[:50]}")
            try:
                updated = json.loads(pref_update.updated_preferences_json)
                self.current_preferences.update(updated)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # Layer 6: strategy generation. Internal per-PRIM fallbacks live inside GenerateFullStrategy.
        prefs_str = json.dumps(self.current_preferences, ensure_ascii=False)

        # Build a fresh context manager here because the previous one is single-use.
        l6_ctx = dspy.context(lm=self.strong_lm) if self.strong_lm else nullcontext()
        with l6_ctx:
            strategy = self.strategy_gen(
                scene=scene,
                user_prefs_str=prefs_str,
                audiogram_json=audiogram_json,
            )

        # Semantic -> physical translation.
        dsp_params = comp_strategy_to_dsp_params(strategy)

        # Persistent store: cache all final outputs.
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
            routing_reasoning=routing.routing_reasoning,
        )
