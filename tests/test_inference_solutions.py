import pandas as pd



from src.ab_testing.bandit_logs import simulate_thompson_logs_with_propensity

from src.ab_testing.weighted_inference import run_weighted_ab_inference

from src.experiments.inference_solutions_validity import run_inference_solutions_study





def test_weighted_inference_runs_on_synthetic_logs() -> None:

    frame = pd.DataFrame(

        [

            {"arm": 0, "reward": 0.0, "propensity_control": 0.5, "propensity_treatment": 0.5},

            {"arm": 1, "reward": 1.0, "propensity_control": 0.5, "propensity_treatment": 0.5},

            {"arm": 0, "reward": 1.0, "propensity_control": 0.5, "propensity_treatment": 0.5},

            {"arm": 1, "reward": 0.0, "propensity_control": 0.5, "propensity_treatment": 0.5},

        ]

    )

    result = run_weighted_ab_inference(frame, n_bootstrap=50, bootstrap_seed=1)

    assert result.n_observations == 4

    assert 0.0 <= result.p_value <= 1.0





def test_thompson_logs_include_propensity() -> None:

    logs = simulate_thompson_logs_with_propensity(

        p_control=0.05,

        p_treatment=0.06,

        horizon=100,

        seed=7,

    )

    assert len(logs) == 100

    assert {"propensity_control", "propensity_treatment"}.issubset(logs.columns)





def test_inference_solutions_study_runs_small() -> None:

    summary, detail = run_inference_solutions_study(

        horizon=500,

        n_trials=8,

        alpha=0.05,

        null_ctr=0.05,

        effect_ctrs=[0.06, 0.08],

    )

    assert not summary.empty

    assert set(summary["method"]).issuperset({"naive_ab_full_ts", "ips_weighted_ab", "naive_ab_fixed_only"})

    assert len(detail) == 8 * 3 * 3

