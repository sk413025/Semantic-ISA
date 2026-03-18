ARCHITECTURE_MAP = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                    ACOUSTIC SEMANTIC IR (ASIR) v0.3                     ║
║   DSPy + GEPA + Method A (Routing) + Multimodal (Audio/Image)          ║
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
║  Layer 4: Perceptual Description  ★ +Audio +Image       │              ║
║  ┌──────────┴──────────────────────────────────────┐    │              ║
║  │ [PRIM] DescribeNoiseSig    → fast_lm +🎵+🖼️    │    │              ║
║  │ [PRIM] DescribeSpeechSig   → fast_lm +🎵+🖼️    │    │              ║
║  │ [PRIM] DescribeEnvSig      → fast_lm +🎵+🖼️    │    │              ║
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
║  GEPA Optimizer (v0.3):                                                ║
║  ├─ Optimizable targets:                                               ║
║  │   ├─ 9 original PRIMs  (can be frozen in Phase 1)                  ║
║  │   └─ 5 new ROUTERs     (★ Method A learnable routing)             ║
║  ├─ Feedback metric:   create_acoustic_feedback_metric()               ║
║  │   └─ Encodes:  物理約束 + 聽力學原則 + routing 合理性              ║
║  ├─ Per-predictor feedback: PRIM + ROUTER 各有獨立回饋                ║
║  ├─ Reflection:  用 strong_lm 反思失敗案例，自動改進 prompt           ║
║  └─ ★ Phase 4: MultiModalInstructionProposer (v0.3)                  ║
║       反思 LM 同時看到 Audio/Image + 錯誤分析 → 更精準改進 prompt     ║
║                                                                         ║
║  Multimodal Pipeline (v0.3):                                           ║
║  ├─ Phase 1: raw_signal_to_audio()  → dspy.Audio (原始音訊)           ║
║  ├─ Phase 2: generate_spectrogram() → dspy.Image (頻譜圖)             ║
║  ├─ Phase 3: L4/L5 Signatures 加入 Optional[Audio/Image] 輸入        ║
║  ├─ Phase 4: GEPA + MultiModalInstructionProposer                     ║
║  └─ Phase 5: 成本感知路由 (fast=text, medium=image, full=audio+image) ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
