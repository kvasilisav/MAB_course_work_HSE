from __future__ import annotations

import numpy as np

from .base import BasePolicyState


class LinUCBPolicy:
    name = "linucb"

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
        regularization: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be at least 2.")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.regularization = regularization
        self._state = BasePolicyState(n_arms=n_arms, seed=seed)
        self._rng = self._state.build_rng()
        self._a = np.array([np.eye(context_dim) * regularization for _ in range(n_arms)], dtype=float)
        self._b = np.zeros((n_arms, context_dim), dtype=float)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._state.seed = seed
        self._rng = self._state.build_rng()
        self._a = np.array(
            [np.eye(self.context_dim) * self.regularization for _ in range(self.n_arms)],
            dtype=float,
        )
        self._b = np.zeros((self.n_arms, self.context_dim), dtype=float)

    def select_arm(self, context: np.ndarray | None = None) -> int:
        if context is None:
            context = np.ones(self.context_dim, dtype=float)
        x = np.asarray(context, dtype=float).reshape(-1)
        if x.size != self.context_dim:
            raise ValueError("context dimension does not match policy context_dim.")

        scores = np.zeros(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            theta = np.linalg.solve(self._a[arm], self._b[arm])
            mean = float(theta @ x)
            variance_term = float(x @ np.linalg.solve(self._a[arm], x))
            bonus = self.alpha * float(np.sqrt(max(0.0, variance_term)))
            scores[arm] = mean + bonus
        return int(np.argmax(scores))

    def update(
        self,
        arm: int,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        if not 0 <= arm < self.n_arms:
            raise ValueError("arm index is out of bounds.")
        if context is None:
            context = np.ones(self.context_dim, dtype=float)
        x = np.asarray(context, dtype=float).reshape(-1)
        if x.size != self.context_dim:
            raise ValueError("context dimension does not match policy context_dim.")
        self._a[arm] += np.outer(x, x)
        self._b[arm] += reward * x

    def snapshot(self) -> dict[str, float | int | list[float]]:
        return {
            "n_arms": self.n_arms,
            "context_dim": self.context_dim,
            "alpha": self.alpha,
            "a_trace": [float(np.trace(item)) for item in self._a],
            "b_norm": [float(np.linalg.norm(item)) for item in self._b],
        }
