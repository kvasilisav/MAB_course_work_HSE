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
