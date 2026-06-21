import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    cumulative_regret,
    cumulative_reward,
    empirical_arm_posterior_means,
    suboptimal_arm_share,
    time_to_first_oracle_optimal,
    time_to_stable_best_arm,
    time_to_stable_mode_arm,
)
from src.evaluation.summary import summarize_runs


def test_cumulative_reward_is_monotonic_for_binary_rewards() -> None:
    rewards = np.array([1, 0, 1, 1, 0], dtype=float)
    cum = cumulative_reward(rewards)
    assert np.all(np.diff(cum) >= 0)


def test_regret_is_zero_when_rewards_match_oracle() -> None:
    rewards = np.array([0.2, 0.8, 0.6], dtype=float)
    regret = cumulative_regret(rewards, rewards)
    assert np.allclose(regret, 0.0)


def test_suboptimal_share_bounds() -> None:
    chosen = np.array([0, 1, 1, 2], dtype=int)
    optimal = np.array([0, 1, 2, 2], dtype=int)
    value = suboptimal_arm_share(chosen, optimal)
    assert 0.0 <= value <= 1.0


def test_time_to_first_oracle_optimal() -> None:
    chosen = np.array([0, 1, 2, 1, 1], dtype=int)
    optimal = np.array([1, 1, 1, 1, 1], dtype=int)
    assert time_to_first_oracle_optimal(chosen, optimal) == 1


def test_time_to_stable_mode_arm_uses_mode_not_oracle() -> None:
    chosen = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert time_to_stable_mode_arm(chosen, threshold=0.95) == 1


def test_time_to_stable_best_arm_uses_posterior_means() -> None:
    chosen = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int)
    rewards = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    posterior = empirical_arm_posterior_means(chosen, rewards, n_arms=2)
    assert time_to_stable_best_arm(posterior, threshold=0.95) is not None


def test_summarize_runs_exports_two_time_metrics() -> None:
    results = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "seed": 0,
                "policy_name": "fixed_ab",
                "chosen_arm": 0,
                "reward": 1.0,
                "oracle_best_arm": 0,
                "oracle_best_reward": 1.0,
            },
            {
                "run_id": "r1",
                "seed": 0,
                "policy_name": "fixed_ab",
                "chosen_arm": 1,
                "reward": 0.0,
                "oracle_best_arm": 0,
                "oracle_best_reward": 1.0,
            },
        ]
    )
    summary = summarize_runs(results)
    assert "time_to_first_oracle_optimal" in summary.columns
    assert "time_to_stable_best_arm" in summary.columns
    assert "time_to_best_arm" not in summary.columns
    assert summary.loc[0, "time_to_first_oracle_optimal"] == 0
