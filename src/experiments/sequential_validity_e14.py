from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ab_testing.sequential import (
    run_always_valid_msprt_inference,
    run_fixed_horizon_inference,
    run_group_sequential_obf_inference,
    run_naive_peeking_inference,
)
from src.bandits.fixed_ab import FixedABPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.environments.bernoulli import BernoulliBanditEnv

DEFAULT_LOOK_FRACTIONS = [0.25, 0.5, 0.75, 1.0]


def _simulate_logs(policy: object, env: BernoulliBanditEnv, horizon: int) -> pd.DataFrame:
    env.reset()
    policy.reset()
    rows: list[dict[str, float | str | int]] = []
    for _ in range(horizon):
        arm = policy.select_arm()
        reward, _ = env.step(arm)
        rows.append(
            {
                "group": "treatment" if arm == 1 else "control",
                "reward": float(reward),
                "arm": int(arm),
            }
        )
        policy.update(arm, reward)
    return pd.DataFrame(rows)


def _evaluate_method(
    *,
    method_name: str,
    method_fn: Callable[[pd.DataFrame], object],
    policy_factory: Callable[[int], object],
    p0: float,
    p1: float,
    horizon: int,
    n_trials: int,
    desc: str | None = None,
) -> dict[str, float | str]:
    rejects = 0
    stop_fractions: list[float] = []
    trial_iter = tqdm(range(n_trials), desc=desc, unit="trial", leave=False) if desc else range(n_trials)

    for seed in trial_iter:
        env = BernoulliBanditEnv([p0, p1], horizon=horizon, seed=seed)
        policy = policy_factory(seed + 1_000_003)
        logs = _simulate_logs(policy, env, horizon)
        if (logs["group"] == "control").sum() == 0 or (logs["group"] == "treatment").sum() == 0:
            continue
        result = method_fn(logs)
        if result.reject_null:
            rejects += 1
        stop_fractions.append(float(result.stop_fraction))

    evaluated = len(stop_fractions)
    return {
        "method": method_name,
        "rejection_rate": rejects / evaluated if evaluated else 0.0,
        "mean_stop_fraction": float(sum(stop_fractions) / len(stop_fractions)) if stop_fractions else 1.0,
        "n_trials_evaluated": evaluated,
    }


def run_sequential_validity_study(
    *,
    horizon: int,
    n_trials: int,
    alpha: float,
    null_ctr: float,
    effect_ctr: float,
    look_fractions: list[float] | None = None,
    msprt_tau: float = 0.02,
    msprt_min_per_arm: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    looks = look_fractions or DEFAULT_LOOK_FRACTIONS
    summary_rows: list[dict[str, float | str | int]] = []
    detail_rows: list[dict[str, float | str | int | bool]] = []

    settings = [("null", null_ctr, null_ctr), ("effect", null_ctr, effect_ctr)]
    policies = {
        "fixed_ab": lambda seed: FixedABPolicy(n_arms=2, probabilities=[0.5, 0.5], seed=seed),
        "thompson_sampling": lambda seed: ThompsonSamplingPolicy(n_arms=2, seed=seed),
    }
    methods: list[tuple[str, Callable[[pd.DataFrame], object]]] = [
        ("fixed_horizon", lambda logs: run_fixed_horizon_inference(logs, alpha=alpha)),
        (
            "naive_peek",
            lambda logs: run_naive_peeking_inference(logs, look_fractions=looks, alpha=alpha),
        ),
        (
            "group_sequential_obf",
            lambda logs: run_group_sequential_obf_inference(logs, look_fractions=looks, alpha=alpha),
        ),
        (
            "always_valid_msprt",
            lambda logs: run_always_valid_msprt_inference(
                logs,
                alpha=alpha,
                tau=msprt_tau,
                min_per_arm=msprt_min_per_arm,
            ),
        ),
    ]

    for scenario, p0, p1 in settings:
        target = alpha if scenario == "null" else 0.8
        for policy_name, policy_factory in policies.items():
            for method_name, method_fn in methods:
                desc = f"E14 {scenario} | {policy_name} | {method_name}"
                stats = _evaluate_method(
                    method_name=method_name,
                    method_fn=method_fn,
                    policy_factory=policy_factory,
                    p0=p0,
                    p1=p1,
                    horizon=horizon,
                    n_trials=n_trials,
                    desc=desc,
                )
                tqdm.write(
                    f"  done: {desc} -> rejection_rate={stats['rejection_rate']:.3f}, "
                    f"mean_stop_fraction={stats['mean_stop_fraction']:.3f}"
                )
                summary_rows.append(
                    {
                        "scenario": scenario,
                        "policy": policy_name,
                        "CTR_A": p0,
                        "CTR_B": p1,
                        "method": method_name,
                        "rejection_rate": stats["rejection_rate"],
                        "mean_stop_fraction": stats["mean_stop_fraction"],
                        "target": target,
                        "n_trials": stats["n_trials_evaluated"],
                    }
                )

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    return summary, detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E14: sequential / peeking inference vs fixed horizon on fixed and adaptive logs.",
    )
    parser.add_argument("--horizon", type=int, default=20_000)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null-ctr", type=float, default=0.05)
    parser.add_argument("--effect-ctr", type=float, default=0.06)
    parser.add_argument("--msprt-tau", type=float, default=0.02)
    parser.add_argument("--msprt-min-per-arm", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/sequential_valid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"E14 sequential validity: horizon={args.horizon}, trials={args.trials}, "
        f"alpha={args.alpha}, looks={DEFAULT_LOOK_FRACTIONS}, msprt_tau={args.msprt_tau}"
    )
    summary, detail = run_sequential_validity_study(
        horizon=args.horizon,
        n_trials=args.trials,
        alpha=args.alpha,
        null_ctr=args.null_ctr,
        effect_ctr=args.effect_ctr,
        msprt_tau=args.msprt_tau,
        msprt_min_per_arm=args.msprt_min_per_arm,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "e14_summary.csv", index=False)
    detail.to_csv(out_dir / "e14_detail.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
