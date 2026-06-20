import pandas as pd

from src.experiments.synthetic_scenarios import SCENARIOS, run_scenario


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
