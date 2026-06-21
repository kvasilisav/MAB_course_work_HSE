from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_reward(rewards: np.ndarray) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=float)
    return np.cumsum(rewards)


def cumulative_regret(rewards: np.ndarray, oracle_rewards: np.ndarray) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=float)
    oracle_rewards = np.asarray(oracle_rewards, dtype=float)
    if rewards.shape != oracle_rewards.shape:
        raise ValueError("rewards and oracle_rewards must have equal shapes.")
    return np.cumsum(oracle_rewards - rewards)


def suboptimal_arm_share(chosen_arms: np.ndarray, optimal_arms: np.ndarray) -> float:
    chosen_arms = np.asarray(chosen_arms, dtype=int)
    optimal_arms = np.asarray(optimal_arms, dtype=int)
    if chosen_arms.shape != optimal_arms.shape:
        raise ValueError("chosen_arms and optimal_arms must have equal shapes.")
    if chosen_arms.size == 0:
        return 0.0
    return float(np.mean(chosen_arms != optimal_arms))


def time_to_first_oracle_optimal(chosen_arms: np.ndarray, optimal_arms: np.ndarray) -> int | None:
    chosen = np.asarray(chosen_arms, dtype=int)
    optimal = np.asarray(optimal_arms, dtype=int)
    if chosen.size == 0:
        return None
    if chosen.shape != optimal.shape:
        raise ValueError("chosen_arms and optimal_arms must have equal shapes.")
    matches = np.flatnonzero(chosen == optimal)
    if matches.size == 0:
        return None
    return int(matches[0])


def time_to_stable_mode_arm(chosen_arms: pd.Series, threshold: float = 0.95) -> int | None:
    """First step when the most frequently chosen arm (mode) stays stable.

    This is a diagnostic for allocation stability, not the oracle-best arm.
    """
    if chosen_arms.empty:
        return None
    mode_arm = int(chosen_arms.mode().iloc[0])
    for idx, arm in enumerate(chosen_arms.to_list()):
        if int(arm) != mode_arm:
            continue
        suffix = chosen_arms.iloc[idx:]
        if (suffix == mode_arm).mean() >= threshold:
            return idx
    return None


def empirical_arm_posterior_means(
    chosen_arms: np.ndarray,
    rewards: np.ndarray,
    n_arms: int,
    *,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> pd.DataFrame:
    if n_arms < 1:
        raise ValueError("n_arms must be positive.")
    chosen = np.asarray(chosen_arms, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    if chosen.shape != rewards.shape:
        raise ValueError("chosen_arms and rewards must have equal shapes.")

    prior_mean = alpha_prior / (alpha_prior + beta_prior)
    clicks = np.zeros(n_arms, dtype=float)
    pulls = np.zeros(n_arms, dtype=float)
    rows: list[dict[int, float]] = []
    for arm, reward in zip(chosen, rewards):
        if not 0 <= arm < n_arms:
            raise ValueError("arm index is out of bounds.")
        pulls[arm] += 1.0
        if reward >= 1.0:
            clicks[arm] += 1.0
        rows.append(
            {
                a: float((clicks[a] + alpha_prior) / (pulls[a] + alpha_prior + beta_prior))
                if pulls[a] > 0.0
                else float(prior_mean)
                for a in range(n_arms)
            }
        )
    return pd.DataFrame(rows)


def time_to_stable_best_arm(
    posterior_or_means: pd.DataFrame,
    threshold: float = 0.95,
) -> int | None:
    if posterior_or_means.empty:
        return None
    winners = posterior_or_means.idxmax(axis=1)
    if winners.empty:
        return None
    final_winner = winners.iloc[-1]
    stability = (winners == final_winner).astype(float).rolling(window=20, min_periods=5).mean()
    stable_idx = stability[stability >= threshold].index
    if len(stable_idx) == 0:
        return None
    first = stable_idx[0]
    return int(first) if isinstance(first, (int, np.integer)) else None
