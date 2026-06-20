from __future__ import annotations

import numpy as np

from .base import BasePolicyState


class ThompsonSamplingPolicy:
    name = "thompson_sampling"

    def __init__(
        self,
        n_arms: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be at least 2.")
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("priors must be positive.")
        self.n_arms = n_arms
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self._state = BasePolicyState(n_arms=n_arms, seed=seed)
        self._rng = self._state.build_rng()
        self._alphas = np.full(n_arms, alpha_prior, dtype=float)
        self._betas = np.full(n_arms, beta_prior, dtype=float)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._state.seed = seed
        self._rng = self._state.build_rng()
        self._alphas = np.full(self.n_arms, self.alpha_prior, dtype=float)
        self._betas = np.full(self.n_arms, self.beta_prior, dtype=float)

    def select_arm(self, context: np.ndarray | None = None) -> int:
        samples = self._rng.beta(self._alphas, self._betas)
        return int(np.argmax(samples))

    def update(
        self,
        arm: int,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        if not 0 <= arm < self.n_arms:
            raise ValueError("arm index is out of bounds.")
        if reward >= 1.0:
            self._alphas[arm] += 1.0
        else:
            self._betas[arm] += 1.0

    def snapshot(self) -> dict[str, float | int | list[float]]:
        return {
            "n_arms": self.n_arms,
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "alphas": self._alphas.tolist(),
            "betas": self._betas.tolist(),
        }
