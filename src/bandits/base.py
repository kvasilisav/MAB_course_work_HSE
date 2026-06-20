from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class BanditPolicy(Protocol):
    name: str
    n_arms: int

    def reset(self, seed: int | None = None) -> None: ...

    def select_arm(self, context: np.ndarray | None = None) -> int: ...

    def update(
        self,
        arm: int,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None: ...

    def snapshot(self) -> dict[str, float | int | list[float]]: ...


@dataclass
class BasePolicyState:
    n_arms: int
    seed: int | None = None

    def build_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)
