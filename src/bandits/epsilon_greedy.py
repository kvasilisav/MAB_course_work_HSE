from __future__ import annotations

import numpy as np

from .base import BasePolicyState


class EpsilonGreedyPolicy:
    name = "epsilon_greedy"

    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int | None = None) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be at least 2.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        self.n_arms = n_arms
        self.epsilon = epsilon
        self._state = BasePolicyState(n_arms=n_arms, seed=seed)
        self._rng = self._state.build_rng()
        self._counts = np.zeros(n_arms, dtype=int)
        self._means = np.zeros(n_arms, dtype=float)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._state.seed = seed
        self._rng = self._state.build_rng()
        self._counts = np.zeros(self.n_arms, dtype=int)
        self._means = np.zeros(self.n_arms, dtype=float)

    def select_arm(self, context: np.ndarray | None = None) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_arms))
        return int(np.argmax(self._means))

    def update(
        self,
        arm: int,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        if not 0 <= arm < self.n_arms:
            raise ValueError("arm index is out of bounds.")
        self._counts[arm] += 1
        n = self._counts[arm]
        self._means[arm] += (reward - self._means[arm]) / n

    def snapshot(self) -> dict[str, float | int | list[float]]:
        return {
            "n_arms": self.n_arms,
            "epsilon": self.epsilon,
            "counts": self._counts.astype(float).tolist(),
            "means": self._means.tolist(),
        }
