from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.bandits.base import BanditPolicy
from src.pipeline.schemas import EventRecord


@dataclass(slots=True)
class ReplayEvaluationResult:
    policy_name: str
    seed: int
    total_events: int
    accepted_events: int
    acceptance_rate: float
    accepted_mean_reward: float
    ips_estimate: float
    snips_estimate: float
    effective_sample_size: float


def _safe_propensity(event: EventRecord, *, propensity_floor: float) -> float | None:
    if event.propensity is None:
        return None
    if event.propensity <= 0.0:
        return None
    return max(float(event.propensity), propensity_floor)


def _valid_events(events: list[EventRecord]) -> list[EventRecord]:
    rows: list[EventRecord] = []
    for event in events:
        if event.chosen_arm is None or event.observed_reward is None:
            continue
        if event.propensity is None or event.propensity <= 0.0:
            continue
        rows.append(event)
    return rows


def run_replay_evaluation(
    *,
    events: list[EventRecord],
    policy_factory: Callable[[int, int], BanditPolicy],
    seed: int,
    freeze_policy: bool = True,
    propensity_floor: float = 0.01,
    shuffle_events: bool = True,
) -> ReplayEvaluationResult:
    if not events:
        raise ValueError("events must not be empty.")
    n_arms = max(max(row.candidate_arms) for row in events) + 1
    policy = policy_factory(n_arms, seed)
    policy.reset(seed=seed)

    stream = _valid_events(events)
    if not stream:
        raise ValueError("No valid logged rows with chosen_arm, reward, and positive propensity were found.")

    if shuffle_events:
        rng = np.random.default_rng(seed)
        stream = list(stream)
        rng.shuffle(stream)

    accepted_rewards: list[float] = []
    weights: list[float] = []
    weighted_rewards: list[float] = []
    accepted = 0
    valid_logged_rows = 0

    for event in stream:
        propensity = _safe_propensity(event, propensity_floor=propensity_floor)
        if propensity is None:
            continue
        valid_logged_rows += 1

        context = event.context
        chosen_by_target = policy.select_arm(context)
        indicator = 1.0 if chosen_by_target == event.chosen_arm else 0.0
        weight = indicator / propensity
        weights.append(weight)
        weighted_rewards.append(weight * float(event.observed_reward))

        if indicator > 0.0:
            accepted += 1
            reward = float(event.observed_reward)
            accepted_rewards.append(reward)
            if not freeze_policy:
                policy.update(chosen_by_target, reward, context)

    if valid_logged_rows == 0:
        raise ValueError("No valid logged rows with chosen_arm, reward, and positive propensity were found.")

    total_events = len(events)
    accepted_mean_reward = float(np.mean(accepted_rewards)) if accepted_rewards else 0.0
    ips_estimate = float(np.sum(weighted_rewards) / valid_logged_rows)
    weight_sum = float(np.sum(weights))
    snips_estimate = float(np.sum(weighted_rewards) / weight_sum) if weight_sum > 0 else 0.0
    weight_sq_sum = float(np.sum(np.square(weights)))
    ess = float((weight_sum * weight_sum) / weight_sq_sum) if weight_sq_sum > 0 else 0.0

    return ReplayEvaluationResult(
        policy_name=getattr(policy, "name", "unknown_policy"),
        seed=seed,
        total_events=total_events,
        accepted_events=accepted,
        acceptance_rate=float(accepted / total_events) if total_events > 0 else 0.0,
        accepted_mean_reward=accepted_mean_reward,
        ips_estimate=ips_estimate,
        snips_estimate=snips_estimate,
        effective_sample_size=ess,
    )


def compare_policies_replay(
    *,
    events: list[EventRecord],
    policy_factories: dict[str, Callable[[int, int], BanditPolicy]],
    seeds: int,
    freeze_policy: bool = True,
    propensity_floor: float = 0.01,
    shuffle_events: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in range(seeds):
        for _, policy_factory in policy_factories.items():
            result = run_replay_evaluation(
                events=events,
                policy_factory=policy_factory,
                seed=seed,
                freeze_policy=freeze_policy,
                propensity_floor=propensity_floor,
                shuffle_events=shuffle_events,
            )
            rows.append(
                {
                    "policy_name": result.policy_name,
                    "seed": result.seed,
                    "total_events": result.total_events,
                    "accepted_events": result.accepted_events,
                    "acceptance_rate": result.acceptance_rate,
                    "accepted_mean_reward": result.accepted_mean_reward,
                    "ips_estimate": result.ips_estimate,
                    "snips_estimate": result.snips_estimate,
                    "effective_sample_size": result.effective_sample_size,
                }
            )
    return pd.DataFrame(rows).sort_values(["policy_name", "seed"]).reset_index(drop=True)
