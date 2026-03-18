"""
Integration Tests — 真實音檔 → 完整 L1-L7 Harness 管線

需要 OPENAI_API_KEY + 場景 WAV 檔，沒有會自動 skip。
用 asir/eval/ 的場景定義、音檔和 metrics 做端對端驗證。
結果記錄到 MLflow (experiment: asir-integration-pytest)。

跑法:
  PYTHONUTF8=1 python -X utf8 -m pytest tests/test_integration.py -v
"""
import os
import json
import pytest
from pathlib import Path

from asir.eval.examples import create_eval_examples
from asir.eval.metrics import (
    check_l4_perceptual, check_l5_scene,
    check_l6_strategy, check_dsp_output, check_l7_routing,
    compute_score,
)
from asir.primitives.signal import prim_load_audio

EVAL_EXAMPLES = create_eval_examples()
SCENARIO_IDS = [e.scenario for e in EVAL_EXAMPLES]
SCENARIO_DIR = Path(__file__).parent.parent / "asir" / "eval" / "audio" / "scenarios"


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

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skip integration tests",
)


_cached_results = {}


@pytest.fixture(scope="module")
def all_results():
    """Run full harness for all 10 scenarios with real audio, cache results."""
    if _cached_results:
        return _cached_results

    import dspy
    import mlflow
    from asir.harness import AcousticSemanticHarness

    api_key = os.environ["OPENAI_API_KEY"]
    fast_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    strong_lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    dspy.configure(lm=fast_lm)

    harness = AcousticSemanticHarness(
        fast_lm=fast_lm, strong_lm=strong_lm, enable_multimodal=True,
    )

    for ex in EVAL_EXAMPLES:
        scenario = ex.scenario
        wav_path = SCENARIO_DIR / f"{scenario}.wav"
        if not wav_path.exists():
            _cached_results[scenario] = None
            continue

        signal = prim_load_audio(str(wav_path))
        if isinstance(signal, tuple):
            signal = signal[0]

        user_action = str(getattr(ex, 'user_action', 'none'))
        result = harness(
            raw_signal=signal,
            user_action=user_action,
            user_profile=str(ex.user_profile),
            audiogram_json=str(ex.audiogram_json),
        )

        pred = dspy.Prediction(
            percept=result.percept,
            scene=result.scene,
            strategy=result.strategy,
            dsp_params=result.dsp_params,
            execution_depth=str(
                getattr(result, 'execution_depth', 'full')
            ).strip().lower(),
            current_preferences=getattr(result, 'current_preferences', None),
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
        }

        # Reset harness state between scenarios
        harness.current_preferences = {
            "noise_tolerance": "medium",
            "processing_preference": "natural",
            "environment_awareness": "moderate",
            "known_situations": ["菜市場: 增強正前方, 保留環境感"],
        }
        harness.feedback_history = []

    # Log to MLflow
    mlflow.set_experiment("asir-integration-pytest")
    with mlflow.start_run(run_name="pytest_integration"):
        all_failures = []
        for scenario, data in _cached_results.items():
            if data is None:
                continue
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
        mlflow.log_dict({"failures": all_failures}, "pytest_integration_results.json")

    return _cached_results


# ===== Per-layer tests =====

class TestIntegrationL4:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_l4(self, all_results, scenario):
        data = all_results.get(scenario)
        if data is None:
            pytest.skip(f"WAV not found for {scenario}")
        checks = data["checks"]["L4"]
        failed = [(n, d) for n, (p, d) in checks.items() if not p]
        assert not failed, f"L4 {scenario}: " + "; ".join(f"{n}: {d}" for n, d in failed)


class TestIntegrationL5:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_l5(self, all_results, scenario):
        data = all_results.get(scenario)
        if data is None:
            pytest.skip(f"WAV not found for {scenario}")
        checks = data["checks"]["L5"]
        failed = [(n, d) for n, (p, d) in checks.items() if not p]
        assert not failed, f"L5 {scenario}: " + "; ".join(f"{n}: {d}" for n, d in failed)


class TestIntegrationL6:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_l6(self, all_results, scenario):
        data = all_results.get(scenario)
        if data is None:
            pytest.skip(f"WAV not found for {scenario}")
        checks = data["checks"]["L6"]
        failed = [(n, d) for n, (p, d) in checks.items() if not p]
        assert not failed, f"L6 {scenario}: " + "; ".join(f"{n}: {d}" for n, d in failed)


class TestIntegrationDSP:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_dsp(self, all_results, scenario):
        data = all_results.get(scenario)
        if data is None:
            pytest.skip(f"WAV not found for {scenario}")
        checks = data["checks"]["DSP"]
        failed = [(n, d) for n, (p, d) in checks.items() if not p]
        assert not failed, f"DSP {scenario}: " + "; ".join(f"{n}: {d}" for n, d in failed)


class TestIntegrationL7:
    @pytest.mark.parametrize("scenario", SCENARIO_IDS)
    def test_l7(self, all_results, scenario):
        data = all_results.get(scenario)
        if data is None:
            pytest.skip(f"WAV not found for {scenario}")
        checks = data["checks"]["L7"]
        failed = [(n, d) for n, (p, d) in checks.items() if not p]
        assert not failed, f"L7 {scenario}: " + "; ".join(f"{n}: {d}" for n, d in failed)
