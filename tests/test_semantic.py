"""
L4-L7 Semantic Layer Tests — 語意推理品質驗證

需要 OPENAI_API_KEY，沒有 API key 會自動 skip。
用 asir/eval/ 的場景定義、metrics 和 composites 跑 10 個場景。
結果記錄到 MLflow (experiment: asir-eval-pytest)。

跑法:
  PYTHONUTF8=1 python -X utf8 -m pytest tests/test_semantic.py -v
"""
import os
import json
import pytest

from asir.eval.examples import create_eval_examples
from asir.eval.run import build_features, _build_trace
from asir.eval.metrics import (
    check_l4_perceptual, check_l5_scene,
    check_l6_strategy, check_dsp_output, check_l7_routing,
    compute_score,
)

EVAL_EXAMPLES = create_eval_examples()
SCENARIO_IDS = [e.scenario for e in EVAL_EXAMPLES]


def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())


_load_env()

# Skip entire module if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skip semantic tests",
)


# ===== Module-scoped fixture: run all scenarios once =====

_cached_results = {}


@pytest.fixture(scope="module")
def all_results():
    """Run L4-L7 pipeline for all 10 scenarios once, cache results."""
    if _cached_results:
        return _cached_results

    import dspy
    import mlflow
    from asir.composites import (
        FullPerceptualDescription, SceneWithHistory,
        GenerateFullStrategy, comp_strategy_to_dsp_params,
    )
    from asir.routing.pipeline import PipelineRoutingSig

    api_key = os.environ["OPENAI_API_KEY"]
    fast_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    strong_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    dspy.configure(lm=fast_lm)

    perceptual_desc = FullPerceptualDescription()
    scene_understanding = SceneWithHistory()
    strategy_gen = GenerateFullStrategy()
    pipeline_router = dspy.ChainOfThought(PipelineRoutingSig)

    scenario_traces = {}

    for ex in EVAL_EXAMPLES:
        scenario = ex.scenario
        features = build_features(ex)

        with dspy.context(lm=fast_lm):
            percept = perceptual_desc(
                acoustic_features=features,
                user_context=str(ex.user_profile),
            )

        with dspy.context(lm=strong_lm):
            scene = scene_understanding(
                percept=percept,
                user_profile=str(ex.user_profile),
                recent_scenes=[],
            )

        prefs = {"noise_tolerance": "medium", "processing_preference": "natural"}
        user_action = str(getattr(ex, 'user_action', 'none'))

        if user_action != "none":
            from asir.primitives.intent import UpdatePreferencesSig
            with dspy.context(lm=strong_lm):
                pref_update = dspy.ChainOfThought(UpdatePreferencesSig)(
                    current_preferences=json.dumps(prefs, ensure_ascii=False),
                    user_feedback=user_action,
                    current_scene=scene.situation,
                    feedback_history="",
                )
            try:
                updated = json.loads(pref_update.updated_preferences_json)
                prefs.update(updated)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        prefs_str = json.dumps(prefs, ensure_ascii=False)
        with dspy.context(lm=strong_lm):
            strategy = strategy_gen(
                scene=scene,
                user_prefs_str=prefs_str,
                audiogram_json=str(ex.audiogram_json),
            )

        dsp_params = comp_strategy_to_dsp_params(strategy)

        with dspy.context(lm=fast_lm):
            routing = pipeline_router(
                signal_change_magnitude=1.0,
                last_scene_confidence=0.5,
                last_strategy_confidence=0.5,
                user_action=user_action,
                frames_since_full_update=10,
            )

        pred = dspy.Prediction(
            percept=percept,
            scene=scene,
            strategy=strategy,
            dsp_params=dsp_params,
            execution_depth=str(routing.execution_depth).strip().lower(),
            current_preferences=prefs if isinstance(prefs, dict) else None,
        )

        _cached_results[scenario] = {
            "example": ex,
            "pred": pred,
            "checks": {
                "L4": check_l4_perceptual(ex, pred),
                "L5": check_l5_scene(ex, pred),
                "L6": check_l6_strategy(ex, pred),
                "DSP": check_dsp_output(ex, pred),
                "L7": check_l7_routing(ex, pred),
            },
            "trace": _build_trace(pred),
        }

    # Log to MLflow
    mlflow.set_experiment("asir-eval-pytest")
    with mlflow.start_run(run_name="pytest_semantic"):
        all_failures = []
        for scenario, data in _cached_results.items():
            for layer, checks in data["checks"].items():
                score = compute_score(checks)
                mlflow.log_metric(f"{scenario}_{layer}", score)
                for check_name, (passed, detail) in checks.items():
                    if not passed:
                        all_failures.append({
                            "scenario": scenario, "layer": layer,
                            "check": check_name, "detail": detail,
                        })
        mlflow.log_metric("num_failures", len(all_failures))
        mlflow.log_dict({
            "scenarios": {
                s: {"trace": d["trace"], "checks": {
                    layer: {name: {"passed": p, "detail": det}
                            for name, (p, det) in checks.items()}
                    for layer, checks in d["checks"].items()
                }}
                for s, d in _cached_results.items()
            },
            "failures": all_failures,
        }, "pytest_eval_results.json")

    return _cached_results


# ===== L4 Perceptual Description =====

class TestL4Perceptual:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_l4_checks_pass(self, all_results, scenario):
        """L4 噪音/語音/環境描述是否跟聲學特徵一致。"""
        checks = all_results[scenario]["checks"]["L4"]
        failed = [(name, detail) for name, (passed, detail) in checks.items() if not passed]
        assert not failed, \
            f"L4 failures for {scenario}: " + "; ".join(f"{n}: {d}" for n, d in failed)


# ===== L5 Scene Understanding =====

class TestL5Scene:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_l5_checks_pass(self, all_results, scenario):
        """L5 場景理解是否反映噪音/迴響/聲源數。"""
        checks = all_results[scenario]["checks"]["L5"]
        failed = [(name, detail) for name, (passed, detail) in checks.items() if not passed]
        assert not failed, \
            f"L5 failures for {scenario}: " + "; ".join(f"{n}: {d}" for n, d in failed)


# ===== L6 Strategy =====

class TestL6Strategy:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_l6_checks_pass(self, all_results, scenario):
        """L6 策略推理是否合理（NR 強度、推理長度）。"""
        checks = all_results[scenario]["checks"]["L6"]
        failed = [(name, detail) for name, (passed, detail) in checks.items() if not passed]
        assert not failed, \
            f"L6 failures for {scenario}: " + "; ".join(f"{n}: {d}" for n, d in failed)


# ===== DSP Output =====

class TestDSPOutput:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_dsp_checks_pass(self, all_results, scenario):
        """DSP 參數是否符合物理約束（增益、波束、降噪、壓縮）。"""
        checks = all_results[scenario]["checks"]["DSP"]
        failed = [(name, detail) for name, (passed, detail) in checks.items() if not passed]
        assert not failed, \
            f"DSP failures for {scenario}: " + "; ".join(f"{n}: {d}" for n, d in failed)


# ===== L7 Routing & Preferences =====

class TestL7Routing:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_l7_checks_pass(self, all_results, scenario):
        """L7 路由和偏好更新是否正確。"""
        checks = all_results[scenario]["checks"]["L7"]
        failed = [(name, detail) for name, (passed, detail) in checks.items() if not passed]
        assert not failed, \
            f"L7 failures for {scenario}: " + "; ".join(f"{n}: {d}" for n, d in failed)


# ===== Cross-scenario: E9 vs E10 比較 =====

class TestMarketScenarioPair:
    """菜市場旗艦場景：E9 (自動) vs E10 (太悶了) 的差異。"""

    def test_e10_nr_differs_from_e9(self, all_results):
        """太悶了回饋應導致 E10 的 NR 不同於 E9。"""
        e9 = all_results["wet_market_vendor"]["trace"].get("L6", {})
        e10 = all_results["market_too_muffled"]["trace"].get("L6", {})
        nr9 = e9.get("nr_aggressiveness")
        nr10 = e10.get("nr_aggressiveness")
        if nr9 is not None and nr10 is not None:
            assert nr9 != nr10, \
                f"E9 NR={nr9} should differ from E10 NR={nr10} (user said 太悶了)"

    def test_e10_preferences_updated(self, all_results):
        """E10 (太悶了) 的偏好應有更新。"""
        prefs = all_results["market_too_muffled"]["pred"].current_preferences
        default = {"noise_tolerance": "medium", "processing_preference": "natural"}
        changed = any(
            prefs.get(k) != v for k, v in default.items()
        ) or len(prefs) > len(default)
        assert changed, f"E10 prefs should differ from defaults: {prefs}"
