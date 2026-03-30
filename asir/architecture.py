ARCHITECTURE_MAP = r"""
+=============================================================================+
|                    ACOUSTIC SEMANTIC IR (ASIR) v0.3                         |
|      DSPy + GEPA + Routed Execution + Multimodal Audio / Image Inputs       |
+=============================================================================+
|                                                                             |
|  Pipeline Router (PipelineRoutingSig)                                       |
|  +-----------------------------------------------------------------------+  |
|  | [ROUTING] pipeline_router -> fast | medium | full                    |  |
|  | Determines how deep each frame runs: fast=L1-2, medium=L1-5,         |  |
|  | full=L1-7.                                                           |  |
|  +--------------------------------------+--------------------------------+  |
|                                         |                                   |
|  Layer 7: Intent and Preferences        |                                   |
|  +--------------------------------------+--------------------------------+  |
|  | [PRIM] ParseIntentSig       -> dspy.ChainOfThought -> LLM            |  |
|  | [PRIM] UpdatePreferencesSig -> dspy.ChainOfThought -> LLM            |  |
|  +--------------------------------------+--------------------------------+  |
|                                         |                                   |
|  Layer 6: Strategy Generation           |  Front-end planning + back-end    |
|  +--------------------------------------+--------------------------------+  |
|  | [ROUTING] strategy_planner -> plan beam / NR coordination            |  |
|  |      v enriched context (coordination plan + budget)                 |  |
|  | [PRIM] GenerateBeamformingSig -> LLM + physics                       |  |
|  | [PRIM] GenerateNRSig         -> LLM                                  |  |
|  | [PRIM] prim_generate_gain()  -> deterministic (NAL-NL2)              |  |
|  |      v three outputs                                                    |
|  | [ROUTING] strategy_integrator -> conflict checks, NR refinement,     |  |
|  |                                   confidence calibration             |  |
|  | [COMP] GenerateFullStrategy  -> dspy.Module orchestrating the above   |  |
|  |                                                                       |  |
|  | comp_strategy_to_dsp_params(): semantic-to-physical translation       |  |
|  +-----------+---------------------------------------------+-------------+  |
|              |                                             |                |
|==============+========= SEMANTIC BOUNDARY (Shannon Cut) ====+================|
|              |                                             |                |
|  Layer 5: Scene Understanding                              |                |
|  +-----------+-------------------------+                    |                |
|  | [PRIM] ReasonAboutSceneSig -> strong_lm                 |                |
|  | [ROUTING] scene_router -> decide whether contradiction  |                |
|  |                           resolution is needed          |                |
|  | [PRIM] resolve_contradictions (conditional)             |                |
|  | [COMP] SceneWithHistory                                 |                |
|  +-----------+-------------------------+                    |                |
|              |                                             |                |
|  Layer 4: Perceptual Description   +Audio +Image           |                |
|  +-----------+-----------------------------------------+   |                |
|  | [PRIM] DescribeNoiseSig  -> fast_lm + audio + image |   |                |
|  | [PRIM] DescribeSpeechSig -> fast_lm + audio + image |   |                |
|  | [PRIM] DescribeEnvSig    -> fast_lm + audio + image |   |                |
|  |      v three descriptions                             |   |                |
|  | [ROUTING] aggregate_router -> dynamic weighting +     |   |                |
|  |                                confidence             |   |                |
|  | [COMP] FullPerceptualDescription                      |   |                |
|  +-----------+-----------------------------------------+   |                |
|              |                                             |                |
|==============+============== SEMANTIC BOUNDARY ============+================|
|              |                                                              |
|  Layer 3: Acoustic Features (deterministic)                                 |
|  +-----------+-----------------------------------------+                    |
|  | [PRIM] prim_extract_mfcc()                          |                    |
|  | [PRIM] prim_estimate_snr()                          |                    |
|  | [PRIM] prim_estimate_rt60()                         |                    |
|  | [COMP] comp_extract_full_features()                 |                    |
|  +-----------+-----------------------------------------+                    |
|              |                                                              |
|  Layer 2: Signal Processing (deterministic)                                 |
|  +-----------+-----------------------------------------+                    |
|  | [PRIM] prim_fft()                                   |                    |
|  | [PRIM] prim_estimate_noise_psd()                    |                    |
|  | [PRIM] prim_beamform() <- DSPParameterSet           |                    |
|  | [COMP] comp_spectral_subtract()                     |                    |
|  +-----------+-----------------------------------------+                    |
|              |                                                              |
|  Layer 1: Physical Sensing (deterministic, hardware-facing)                 |
|  +-----------------------------------------------------+                    |
|  | [PRIM] prim_sample_audio()                          |                    |
|  +-----------------------------------------------------+                    |
|                                                                             |
+=============================================================================+
|  HARNESS = AcousticSemanticHarness(dspy.Module)                             |
|  - Semantic linker: string serialization and format adaptation in forward() |
|  - Semantic runtime: try/except fallback + dspy.context                     |
|  - Semantic scheduler: pipeline_router + fast_lm / strong_lm                |
|  - Persistent store: scene history, cached results, preferences             |
|                                                                             |
|  GEPA optimizer                                                             |
|  - Optimizable targets: 9 original PRIMs + 5 ROUTERs                       |
|  - Feedback metric: create_acoustic_feedback_metric()                       |
|  - Encodes physical constraints, hearing-aid principles, and routing logic  |
|  - Per-predictor feedback for both PRIM and ROUTER modules                  |
|  - Reflection loop improves prompts with strong_lm                          |
|  - MultimodalInstructionProposer supports audio/image-aware prompt updates  |
|                                                                             |
|  Multimodal pipeline                                                        |
|  - raw_signal_to_audio()  -> dspy.Audio                                     |
|  - generate_spectrogram() -> dspy.Image                                     |
|  - L4/L5 signatures accept optional audio/image inputs                      |
|  - GEPA can optimize with multimodal instruction proposal                   |
|  - Cost-aware routing: fast=text, medium=image, full=audio+image            |
+=============================================================================+
"""
