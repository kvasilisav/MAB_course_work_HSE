from __future__ import annotations

import numpy as np

from .base import BasePolicyState


class FixedABPolicy:
    name = "fixed_ab"

    def __init__(
        self,
        n_arms: int,
        probabilities: list[float] | None = None,
        seed: int | None = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError("FixedABPolicy requires at least 2 arms.")
        self.n_arms = n_arms
        if probabilities is None:
            probabilities = [1.0 / n_arms] * n_arms
        if len(probabilities) != n_arms:
            raise ValueError("probabilities length must match n_arms.")
        probs = np.asarray(probabilities, dtype=float)
        if np.any(probs < 0):
            raise ValueError("probabilities must be non-negative.")
        if np.isclose(probs.sum(), 0.0):
            raise ValueError("probabilities must sum to positive value.")
        self._probabilities = (probs / probs.sum()).tolist()
        self._state = BasePolicyState(n_arms=n_arms, seed=seed)
        self._rng = self._state.build_rng()
        self._counts = np.zeros(n_arms, dtype=int)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._state.seed = seed
        self._rng = self._state.build_rng()
        self._counts = np.zeros(self.n_arms, dtype=int)

    def select_arm(self, context: np.ndarray | None = None) -> int:
        arm = int(self._rng.choice(self.n_arms, p=self._probabilities))
        self._counts[arm] += 1
        return arm

    def update(
        self,
        arm: int,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        # Fixed A/B does not adapt traffic allocation.
        return None

    def snapshot(self) -> dict[str, float | int | list[float]]:
        return {
            "n_arms": self.n_arms,
            "probabilities": list(self._probabilities),
            "counts": self._counts.astype(float).tolist(),
        }
