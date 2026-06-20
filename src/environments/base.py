from __future__ import annotations

from typing import Protocol

import numpy as np


class BanditEnvironment(Protocol):
    n_arms: int

    def reset(self, seed: int | None = None) -> None: ...

    def current_context(self) -> np.ndarray | None: ...

    def step(
        self,
        arm: int,
        context: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float | int]]: ...

    def optimal_expected_reward(self, context: np.ndarray | None = None) -> float: ...
