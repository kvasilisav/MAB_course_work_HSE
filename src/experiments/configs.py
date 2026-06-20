from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.bandits.base import BanditPolicy
from src.environments.base import BanditEnvironment


@dataclass(slots=True)
class ExperimentConfig:
    run_id: str
    seed: int
    horizon: int
    policy_factory: Callable[[int, int], BanditPolicy]
    environment_factory: Callable[[int], BanditEnvironment]
    mode: str = "synthetic"
    batch_size: int = 1
    assignments_path: str | Path | None = None
