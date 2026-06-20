from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ab_testing.inference import run_ab_inference
from src.bandits.fixed_ab import FixedABPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.environments.bernoulli import BernoulliBanditEnv


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


def _evaluate_policy_rejection_rate(
    *,
    policy_factory,
    p0: float,
    p1: float,
    horizon: int,
    n_trials: int,
    alpha: float,
    desc: str | None = None,
) -> float:
    rejects = 0
    trial_iter = tqdm(range(n_trials), desc=desc, unit="trial", leave=False) if desc else range(n_trials)
    for seed in trial_iter:
        env = BernoulliBanditEnv([p0, p1], horizon=horizon, seed=seed)
        policy = policy_factory(seed + 1_000_003)
        logs = _simulate_logs(policy, env, horizon)
        if (logs["group"] == "control").sum() == 0 or (logs["group"] == "treatment").sum() == 0:
            continue
        result = run_ab_inference(logs, alpha=alpha)
        if result.reject_null:
            rejects += 1
    return rejects / n_trials


def run_validity_study(
    *,
    horizon: int,
    n_trials: int,
    alpha: float,
    null_ctr: float,
    effect_ctrs: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    effect_settings = [("null", null_ctr, null_ctr)]
    for effect_ctr in effect_ctrs:
        effect_settings.append((f"effect_{effect_ctr:.4f}", null_ctr, effect_ctr))
    policies = {
        "fixed_ab": lambda seed: FixedABPolicy(n_arms=2, probabilities=[0.5, 0.5], seed=seed),
        "thompson_sampling_naive_inference": lambda seed: ThompsonSamplingPolicy(n_arms=2, seed=seed),
    }

    for scenario, p0, p1 in effect_settings:
        target = alpha if scenario == "null" else 0.8
        for policy_name, policy_factory in policies.items():
            desc = f"E4 {scenario} | {policy_name}"
            rejection_rate = _evaluate_policy_rejection_rate(
                policy_factory=policy_factory,
                p0=p0,
                p1=p1,
                horizon=horizon,
                n_trials=n_trials,
                alpha=alpha,
                desc=desc,
            )
            tqdm.write(f"  done: {desc} -> rejection_rate={rejection_rate:.3f}")
            rows.append(
                {
                    "scenario": scenario,
                    "policy": policy_name,
                    "p_control": p0,
                    "p_treatment": p1,
                    "rejection_rate": rejection_rate,
                    "target_rate": target,
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B inference validity study against adaptive MAB allocation.")
    parser.add_argument("--horizon", type=int, default=20000)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null-ctr", type=float, default=0.05)
    parser.add_argument("--effect-ctr", type=float, default=None)
    parser.add_argument(
        "--effect-ctrs",
        default="0.06",
        help="Comma-separated treatment CTRs for effect scenarios (e.g. 0.06,0.08).",
    )
    parser.add_argument("--output-path", default="outputs/ab_validity/summary.csv")
    return parser.parse_args()


def _parse_effect_ctrs(args: argparse.Namespace) -> list[float]:
    if args.effect_ctr is not None:
        return [float(args.effect_ctr)]
    return [float(item.strip()) for item in args.effect_ctrs.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    effect_ctrs = _parse_effect_ctrs(args)
    print(
        f"E4 validity study: horizon={args.horizon}, trials={args.trials}, "
        f"alpha={args.alpha}, null_ctr={args.null_ctr}, effect_ctrs={effect_ctrs}"
    )
    frame = run_validity_study(
        horizon=args.horizon,
        n_trials=args.trials,
        alpha=args.alpha,
        null_ctr=args.null_ctr,
        effect_ctrs=effect_ctrs,
    )
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(frame)


if __name__ == "__main__":
    main()
