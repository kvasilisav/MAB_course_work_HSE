from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EventRecord:
    event_id: int
    timestamp: str | int
    user_id: str | int
    candidate_arms: list[int]
    chosen_arm: int | None
    observed_reward: float | None
    propensity: float | None = None
    context: np.ndarray | None = None


@dataclass(slots=True)
class AssignmentRecord:
    run_id: str
    event_id: int
    policy_name: str
    chosen_arm: int
    reward: float
    oracle_best_arm: int
    oracle_best_reward: float
    batch_id: int
