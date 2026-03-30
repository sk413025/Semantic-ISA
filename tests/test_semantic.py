"""
L4-L7 semantic layer tests for reasoning quality.

These tests require `OPENAI_API_KEY`; if the key is missing the module is
skipped automatically.

The suite runs all 10 scenarios using the scenario definitions, metrics, and
composites from `asir/eval/`, then logs summary artifacts to MLflow
(`experiment: asir-eval-pytest`).

Run with:
  PYTHONUTF8=1 python -X utf8 -m pytest tests/test_semantic.py -v
"""

import json
import os

import pytest

from asir.eval.examples import create_eval_examples
from asir.eval.metrics import (
    check_dsp_output,
    check_l4_perceptual,
    check_l5_scene,
    check_l6_strategy,
    check_l7_routing,
    compute_score,
)
from asir.eval.run import _build_trace, build_features

EVAL_EXAMPLES = create_eval_examples()
SCENARIO_IDS = [e.scenario for e in EVAL_EXAMPLES]


def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_load_env()

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skip semantic tests",
)

_cached_results = {}


@pytest.fixture(scope="module")
def all_results():
    """Run the semantic pipeline once for all scenarios and cache results."""
    if _cached_results:
        return _cached_results

    import dspy
    import mlflow

    from asir.composites import (
        FullPerceptualDescription,
        GenerateFullStrategy,
        SceneWithHistory,
        comp_strategy_to_dsp_params,
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
        user_action = str(getattr(ex, "user_action", "none"))

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

    mlflow.set_experiment("asir-eval-pytest")
    with mlflow.start_run(run_name="pytest_semantic"):
        all_failures = []
        for scenario, data in _cached_results.items():
            for layer, checks in data["checks"].items():
                score = compute_score(checks)
                mlflow.log_metric(f"{scenario}_{layer}", score)
                for check_name, (passed, detail) in checks.items():
                    if not passed:
                        all_failures.append(
                            {
                                "scenario": scenario,
                                "layer": layer,
                                "check": check_name,
                                "detail": detail,
                            }
                        )
        mlflow.log_metric("num_failures", len(all_failures))
        mlflow.log_dict(
            {
                "scenarios": {
                    s: {
                        "trace": d["trace"],
                        "checks": {
                            layer: {
                                name: {"passed": p, "detail": det}
                                for name, (p, det) in checks.items()
                            }
                            for layer, checks in d["checks"].items()
                        },
                    }
                    for s, d in _cached_results.items()
                },
                "failures": all_failures,
            },
            "pytest_eval_results.json",
        )

    return _cached_results


class TestL4Perceptual:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_l4_checks_pass(self, all_results, scenario):
        """Verify L4 descriptions remain consistent with acoustic features."""
        checks = all_results[scenario]["checks"]["L4"]
        failed = [(name, detail) for name, (passed, detail) in checks.items() if not passed]
        assert not failed, (
            f"L4 failures for {scenario}: "
            + "; ".join(f"{n}: {d}" for n, d in failed)
        )


class TestL5Scene:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_l5_checks_pass(self, all_results, scenario):
        """Verify L5 scene understanding reflects noise, reverb, and sources."""
        checks = all_results[scenario]["checks"]["L5"]
        failed = [(name, detail) for name, (passed, detail) in checks.items() if not passed]
        assert not failed, (
            f"L5 failures for {scenario}: "
            + "; ".join(f"{n}: {d}" for n, d in failed)
        )


class TestL6Strategy:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_l6_checks_pass(self, all_results, scenario):
        """Verify L6 strategy reasoning stays coherent with scene demands."""
        checks = all_results[scenario]["checks"]["L6"]
        failed = [(name, detail) for name, (passed, detail) in checks.items() if not passed]
        assert not failed, (
            f"L6 failures for {scenario}: "
            + "; ".join(f"{n}: {d}" for n, d in failed)
        )


class TestDSPOutput:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_dsp_checks_pass(self, all_results, scenario):
        """Verify DSP parameters obey physical and hearing-aid constraints."""
        checks = all_results[scenario]["checks"]["DSP"]
        failed = [(name, detail) for name, (passed, detail) in checks.items() if not passed]
        assert not failed, (
            f"DSP failures for {scenario}: "
            + "; ".join(f"{n}: {d}" for n, d in failed)
        )


class TestL7Routing:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_l7_checks_pass(self, all_results, scenario):
        """Verify routing and preference updates behave as expected."""
        checks = all_results[scenario]["checks"]["L7"]
        failed = [(name, detail) for name, (passed, detail) in checks.items() if not passed]
        assert not failed, (
            f"L7 failures for {scenario}: "
            + "; ".join(f"{n}: {d}" for n, d in failed)
        )


class TestMarketScenarioPair:
    """Compare the two flagship market scenarios side by side."""

    def test_e10_nr_differs_from_e9(self, all_results):
        """'Too muffled' feedback should change E10 NR relative to E9."""
        e9 = all_results["wet_market_vendor"]["trace"].get("L6", {})
        e10 = all_results["market_too_muffled"]["trace"].get("L6", {})
        nr9 = e9.get("nr_aggressiveness")
        nr10 = e10.get("nr_aggressiveness")
        if nr9 is not None and nr10 is not None:
            assert nr9 != nr10, (
                f"E9 NR={nr9} should differ from E10 NR={nr10} "
                "(user reported the output was too muffled)"
            )

    def test_e10_preferences_updated(self, all_results):
        """The E10 feedback case should update user preferences."""
        prefs = all_results["market_too_muffled"]["pred"].current_preferences
        default = {"noise_tolerance": "medium", "processing_preference": "natural"}
        changed = any(prefs.get(k) != v for k, v in default.items()) or len(prefs) > len(default)
        assert changed, f"E10 preferences should differ from defaults: {prefs}"
