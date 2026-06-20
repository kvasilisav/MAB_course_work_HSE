from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.bandits.fixed_ab import FixedABPolicy
from src.bandits.linucb import LinUCBPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.bandits.ucb1 import UCB1Policy


def _build_factory(name: str, n_arms: int, context_dim: int) -> Callable[[], object]:
    if name == "fixed_ab":
        return lambda: FixedABPolicy(n_arms=n_arms, seed=42)
    if name == "thompson_sampling":
        return lambda: ThompsonSamplingPolicy(n_arms=n_arms, seed=42)
    if name == "ucb1":
        return lambda: UCB1Policy(n_arms=n_arms, seed=42)
    if name == "linucb":
        return lambda: LinUCBPolicy(n_arms=n_arms, context_dim=context_dim, alpha=1.0, seed=42)
    raise ValueError(f"Unknown policy: {name}")


def benchmark_policy(
    *,
    policy_name: str,
    n_arms: int,
    context_dim: int,
    n_warmup: int = 100,
    n_calls: int = 5000,
) -> dict[str, float | int | str]:
    factory = _build_factory(policy_name, n_arms, context_dim)
    policy = factory()
    policy.reset(seed=42)
    rng = np.random.default_rng(42)
    context = rng.uniform(-1.0, 1.0, size=context_dim) if context_dim > 1 else None

    for _ in range(n_warmup):
        arm = policy.select_arm(context)
        policy.update(arm, 1.0, context)

    policy.reset(seed=42)
    select_start = time.perf_counter()
    last_arm = 0
    for _ in range(n_calls):
        last_arm = policy.select_arm(context)
    select_elapsed = time.perf_counter() - select_start

    policy.reset(seed=42)
    update_start = time.perf_counter()
    for _ in range(n_calls):
        policy.update(last_arm, 1.0, context)
    update_elapsed = time.perf_counter() - update_start

    select_us = (select_elapsed / n_calls) * 1_000_000.0
    update_us = (update_elapsed / n_calls) * 1_000_000.0
    return {
        "policy_name": policy_name,
        "n_arms": n_arms,
        "context_dim": context_dim,
        "n_calls": n_calls,
        "select_arm_us_mean": select_us,
        "update_us_mean": update_us,
        "total_decision_us_mean": select_us + update_us,
    }


def run_latency_matrix(
    *,
    n_calls: int = 5000,
    scenarios: list[tuple[int, int]] | None = None,
    policies: list[str] | None = None,
) -> pd.DataFrame:
    if scenarios is None:
        scenarios = [(5, 1), (80, 7)]
    if policies is None:
        policies = ["fixed_ab", "thompson_sampling", "ucb1", "linucb"]
    rows: list[dict[str, float | int | str]] = []
    for n_arms, context_dim in scenarios:
        for policy_name in policies:
            if policy_name == "linucb" and context_dim <= 0:
                continue
            rows.append(
                benchmark_policy(
                    policy_name=policy_name,
                    n_arms=n_arms,
                    context_dim=max(1, context_dim),
                    n_calls=n_calls,
                )
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark per-decision latency of bandit policies.")
    parser.add_argument("--n-calls", type=int, default=5000)
    parser.add_argument("--output-path", default="outputs/extended/policy_latency.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = run_latency_matrix(n_calls=args.n_calls)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
