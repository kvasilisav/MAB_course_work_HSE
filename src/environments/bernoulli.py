from __future__ import annotations

import numpy as np


class BernoulliBanditEnv:
    def __init__(
        self,
        reward_probs: list[float] | np.ndarray,
        horizon: int = 10000,
        seed: int | None = None,
    ) -> None:
        probs = np.asarray(reward_probs, dtype=float)
        if probs.ndim != 1 or probs.size < 2:
            raise ValueError("reward_probs must define at least 2 arms.")
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError("reward_probs must be within [0, 1].")
        self.reward_probs = probs
        self.horizon = horizon
        self.n_arms = int(probs.size)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._step_id = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._step_id = 0

    def current_context(self) -> np.ndarray | None:
        return None

    def step(
        self,
        arm: int,
        context: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float | int]]:
        if not 0 <= arm < self.n_arms:
            raise ValueError("arm index is out of bounds.")
        if self._step_id >= self.horizon:
            raise RuntimeError("Environment horizon exceeded.")

        self._step_id += 1
        p = float(self.reward_probs[arm])
        reward = float(self._rng.binomial(1, p))
        best_arm = int(np.argmax(self.reward_probs))
        best_reward = float(self.reward_probs[best_arm])

        return reward, {"best_arm": best_arm, "best_reward": best_reward}

    def optimal_expected_reward(self, context: np.ndarray | None = None) -> float:
        return float(np.max(self.reward_probs))
