from __future__ import annotations

import numpy as np

from src.bandits.epsilon_greedy import EpsilonGreedyPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy


def thompson_sampling_propensities(
    policy: ThompsonSamplingPolicy,
    *,
    n_mc: int = 500,
    seed: int | None = None,
) -> np.ndarray:
    """Monte Carlo estimate of P(select arm a) under current Beta posteriors."""
    rng = np.random.default_rng(seed)
    alphas = np.asarray(policy._alphas, dtype=float)
    betas = np.asarray(policy._betas, dtype=float)
    samples = rng.beta(alphas, betas, size=(n_mc, policy.n_arms))
    choices = np.argmax(samples, axis=1)
    counts = np.bincount(choices, minlength=policy.n_arms).astype(float)
    propensities = counts / float(n_mc)
    floor = 1e-6
    propensities = np.maximum(propensities, floor)
    propensities /= propensities.sum()
    return propensities


def epsilon_greedy_propensities(policy: EpsilonGreedyPolicy) -> np.ndarray:
    """Exact propensity vector before arm draw (2+ arms)."""
    n_arms = policy.n_arms
    eps = policy.epsilon
    propensities = np.full(n_arms, eps / n_arms, dtype=float)
    greedy_arm = int(np.argmax(policy._means))
    if policy._counts.sum() == 0:
        return np.full(n_arms, 1.0 / n_arms, dtype=float)
    propensities[greedy_arm] += 1.0 - eps
    return propensities
