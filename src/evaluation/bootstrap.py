from __future__ import annotations

import numpy as np


def bootstrap_percentile_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or n_bootstrap <= 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    n = arr.size
    stats = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample = arr[rng.integers(0, n, size=n)]
        stats[idx] = float(sample.mean())

    low = float(np.percentile(stats, 100.0 * alpha / 2.0))
    high = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def bootstrap_snips_ci(
    weights: np.ndarray,
    rewards: np.ndarray,
    *,
    n_bootstrap: int,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bootstrap CI for SNIPS by resampling logged rows with replacement."""
    w = np.asarray(weights, dtype=float)
    r = np.asarray(rewards, dtype=float)
    if w.size == 0 or n_bootstrap <= 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    n = w.size
    stats = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sample_w = w[sample_idx]
        sample_r = r[sample_idx]
        weight_sum = float(sample_w.sum())
        stats[idx] = float((sample_w * sample_r).sum() / weight_sum) if weight_sum > 0.0 else 0.0

    low = float(np.percentile(stats, 100.0 * alpha / 2.0))
    high = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return low, high
