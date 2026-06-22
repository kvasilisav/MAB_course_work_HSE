import numpy as np

from src.evaluation.bootstrap import bootstrap_percentile_ci, bootstrap_snips_ci


def test_bootstrap_percentile_ci_contains_mean() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    low, high = bootstrap_percentile_ci(values, n_bootstrap=500, seed=1)
    assert low <= values.mean() <= high


def test_bootstrap_snips_ci_reasonable() -> None:
    weights = np.array([1.0, 1.0, 0.0, 2.0])
    rewards = np.array([0.0, 1.0, 1.0, 1.0])
    low, high = bootstrap_snips_ci(weights, rewards, n_bootstrap=300, seed=2)
    point = float((weights * rewards).sum() / weights.sum())
    assert low <= point <= high
