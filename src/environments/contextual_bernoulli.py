from __future__ import annotations

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


class ContextualBernoulliBanditEnv:
    """Bernoulli bandit where click probability depends on context and arm."""

    def __init__(
        self,
        n_arms: int = 5,
        context_dim: int = 4,
        horizon: int = 5000,
        seed: int | None = None,
        theta: np.ndarray | None = None,
        biases: np.ndarray | None = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be at least 2.")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.horizon = horizon
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._step_id = 0
        self._current_context: np.ndarray | None = None

        if theta is None:
            rng = np.random.default_rng(seed)
            self._theta = rng.normal(loc=0.0, scale=0.8, size=(n_arms, context_dim))
        else:
            theta_arr = np.asarray(theta, dtype=float)
            if theta_arr.shape != (n_arms, context_dim):
                raise ValueError("theta must have shape (n_arms, context_dim).")
            self._theta = theta_arr

        if biases is None:
            self._biases = np.linspace(-0.5, 0.5, n_arms)
        else:
            bias_arr = np.asarray(biases, dtype=float).reshape(-1)
            if bias_arr.size != n_arms:
                raise ValueError("biases must have length n_arms.")
            self._biases = bias_arr

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._step_id = 0
        self._current_context = None

    def _sample_context(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=self.context_dim)

    def _arm_probs(self, context: np.ndarray) -> np.ndarray:
        logits = self._theta @ context + self._biases
        probs = _sigmoid(logits)
        return np.clip(probs, 0.02, 0.20)

    def current_context(self) -> np.ndarray | None:
        if self._step_id >= self.horizon:
            return None
        if self._current_context is None:
            self._current_context = self._sample_context()
        return self._current_context.copy()

    def step(
        self,
        arm: int,
        context: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float | int]]:
        if not 0 <= arm < self.n_arms:
            raise ValueError("arm index is out of bounds.")
        if self._step_id >= self.horizon:
            raise RuntimeError("Environment horizon exceeded.")

        if context is None:
            context = self.current_context()
        if context is None:
            raise RuntimeError("No context available for this step.")
        x = np.asarray(context, dtype=float).reshape(-1)
        probs = self._arm_probs(x)
        p = float(probs[arm])
        reward = float(self._rng.binomial(1, p))
        best_arm = int(np.argmax(probs))
        best_reward = float(probs[best_arm])

        self._step_id += 1
        self._current_context = None
        return reward, {"best_arm": best_arm, "best_reward": best_reward}

    def optimal_expected_reward(self, context: np.ndarray | None = None) -> float:
        if context is None:
            context = self.current_context()
        if context is None:
            return float(np.max(_sigmoid(self._biases)))
        x = np.asarray(context, dtype=float).reshape(-1)
        return float(np.max(self._arm_probs(x)))
