from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipeline.event_stream import iter_events
from src.pipeline.loader import load_events
from src.pipeline.schemas import EventRecord


class LoggedClicksBanditEnv:
    """Dataset-backed environment with lightweight replay-style rewards."""

    def __init__(
        self,
        events: list[EventRecord],
        n_arms: int | None = None,
        arm_ctr_priors: dict[int, float] | None = None,
        seed: int | None = None,
    ) -> None:
        if not events:
            raise ValueError("events must not be empty.")
        self.events = list(iter_events(events))
        inferred_arms = max(max(event.candidate_arms) for event in self.events) + 1
        self.n_arms = inferred_arms if n_arms is None else n_arms
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._idx = 0
        self._arm_ctr_priors = arm_ctr_priors or self._build_default_priors(self.events, self.n_arms)

    @classmethod
    def from_csv(cls, path: str | Path, seed: int | None = None) -> "LoggedClicksBanditEnv":
        return cls(events=load_events(path), seed=seed)

    @staticmethod
    def _build_default_priors(events: list[EventRecord], n_arms: int) -> dict[int, float]:
        successes = np.zeros(n_arms, dtype=float)
        totals = np.zeros(n_arms, dtype=float)
        for event in events:
            if event.chosen_arm is None or event.observed_reward is None:
                continue
            if 0 <= event.chosen_arm < n_arms:
                successes[event.chosen_arm] += event.observed_reward
                totals[event.chosen_arm] += 1.0
        priors: dict[int, float] = {}
        for arm in range(n_arms):
            # Smoothed CTR prior: Beta(1,1) posterior mean.
            priors[arm] = float((successes[arm] + 1.0) / (totals[arm] + 2.0))
        return priors

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._idx = 0

    def current_context(self) -> np.ndarray | None:
        if self._idx >= len(self.events):
            return None
        return self.events[self._idx].context

    def _current_event(self) -> EventRecord:
        if self._idx >= len(self.events):
            raise RuntimeError("Dataset event stream is exhausted.")
        return self.events[self._idx]

    def step(
        self,
        arm: int,
        context: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float | int]]:
        event = self._current_event()
        if arm not in event.candidate_arms:
            arm = int(event.candidate_arms[0])

        if event.chosen_arm is not None and event.observed_reward is not None and arm == event.chosen_arm:
            reward = float(event.observed_reward)
        else:
            ctr = float(self._arm_ctr_priors.get(arm, 0.5))
            reward = float(self._rng.binomial(1, min(max(ctr, 0.0), 1.0)))

        best_arm = int(max(event.candidate_arms, key=lambda item: self._arm_ctr_priors.get(item, 0.5)))
        best_reward = float(self._arm_ctr_priors.get(best_arm, 0.5))
        self._idx += 1
        return reward, {"best_arm": best_arm, "best_reward": best_reward, "event_id": event.event_id}

    def optimal_expected_reward(self, context: np.ndarray | None = None) -> float:
        event = self._current_event()
        return float(max(self._arm_ctr_priors.get(arm, 0.5) for arm in event.candidate_arms))
