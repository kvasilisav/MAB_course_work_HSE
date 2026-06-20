from __future__ import annotations

import math

import numpy as np

from .base import BasePolicyState


class UCB1Policy:
    name = "ucb1"

    def __init__(self, n_arms: int, exploration: float = 2.0, seed: int | None = None) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be at least 2.")
        if exploration <= 0:
            raise ValueError("exploration constant must be positive.")
        self.n_arms = n_arms
        self.exploration = exploration
        self._state = BasePolicyState(n_arms=n_arms, seed=seed)
        self._rng = self._state.build_rng()
        self._counts = np.zeros(n_arms, dtype=int)
        self._means = np.zeros(n_arms, dtype=float)
        self._total_steps = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._state.seed = seed
        self._rng = self._state.build_rng()
        self._counts = np.zeros(self.n_arms, dtype=int)
        self._means = np.zeros(self.n_arms, dtype=float)
        self._total_steps = 0

    def select_arm(self, context: np.ndarray | None = None) -> int:
        for arm in range(self.n_arms):
            if self._counts[arm] == 0:
                return arm

        log_term = math.log(max(self._total_steps, 1))
        bonuses = np.sqrt((self.exploration * log_term) / self._counts)
        scores = self._means + bonuses
        return int(np.argmax(scores))

    def update(
        self,
        arm: int,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        if not 0 <= arm < self.n_arms:
            raise ValueError("arm index is out of bounds.")
        self._total_steps += 1
        self._counts[arm] += 1
        n = self._counts[arm]
        self._means[arm] += (reward - self._means[arm]) / n

    def snapshot(self) -> dict[str, float | int | list[float]]:
        return {
            "n_arms": self.n_arms,
            "exploration": self.exploration,
            "total_steps": self._total_steps,
            "counts": self._counts.astype(float).tolist(),
            "means": self._means.tolist(),
        }
