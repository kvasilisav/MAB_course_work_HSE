import pandas as pd

from src.experiments.synthetic_scenarios import SCENARIOS, bootstrap_regret_comparisons, run_scenario


def test_synthetic_scenarios_run_small() -> None:
    results, summary = run_scenario(
        scenario_name="small_gap",
        reward_probs=SCENARIOS["small_gap"],
        horizon=200,
        seeds=2,
    )
    assert len(SCENARIOS) == 3
    assert not results.empty
    assert set(summary["scenario"]) == {"small_gap"}
    assert summary["policy_name"].nunique() == 4


def test_synthetic_scenarios_bootstrap_summary() -> None:
    _, summary = run_scenario(
        scenario_name="large_gap",
        reward_probs=SCENARIOS["large_gap"],
        horizon=200,
        seeds=5,
    )
    bootstrap = bootstrap_regret_comparisons(summary, n_bootstrap=100)
    assert not bootstrap.empty
    assert {"regret_diff_ci_low", "regret_diff_ci_high"}.issubset(bootstrap.columns)
