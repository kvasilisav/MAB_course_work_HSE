import numpy as np

from src.evaluation.metrics import cumulative_regret, cumulative_reward, suboptimal_arm_share


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
