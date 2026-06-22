from src.ab_testing.cuped import run_cuped_ab_inference, simulate_fixed_ab_with_covariate
from src.ab_testing.inference import run_ab_inference
from src.experiments.cuped_power_study import run_cuped_power_study


def test_cuped_inference_runs_on_synthetic_logs() -> None:
    logs = simulate_fixed_ab_with_covariate(
        p_control=0.05,
        p_treatment=0.08,
        horizon=500,
        seed=3,
    )
    naive = run_ab_inference(logs, alpha=0.05)
    cuped = run_cuped_ab_inference(logs, alpha=0.05)
    assert naive.n_control + naive.n_treatment == 500
    assert cuped.n_control + cuped.n_treatment == 500
    assert 0.0 <= cuped.p_value <= 1.0


def test_cuped_power_study_runs_small() -> None:
    summary, detail = run_cuped_power_study(
        horizon=300,
        n_trials=6,
        alpha=0.05,
        null_ctr=0.05,
        effect_ctrs=[0.08],
    )
    assert not summary.empty
    assert set(summary["method"]) == {"naive_ab", "cuped_ab"}
    assert len(detail) == 6 * 2 * 2
