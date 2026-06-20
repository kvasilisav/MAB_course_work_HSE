from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ab_testing.inference import run_ab_inference
from src.bandits.epsilon_greedy import EpsilonGreedyPolicy
from src.bandits.fixed_ab import FixedABPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.bandits.ucb1 import UCB1Policy
from src.environments.bernoulli import BernoulliBanditEnv


def _collect_policy_logs(policy: object, env: BernoulliBanditEnv, horizon: int) -> pd.DataFrame:
    env.reset()
    policy.reset()
    rows: list[dict[str, float | int | str]] = []
    for _ in range(horizon):
        arm = int(policy.select_arm())
        reward, _ = env.step(arm)
        rows.append(
            {
                "group": "treatment" if arm == 1 else "control",
                "reward": float(reward),
            }
        )
        policy.update(arm, reward)
    return pd.DataFrame(rows)


def run_single_experiment_suite(*, p0: float, p1: float, horizon: int, seed: int, alpha: float) -> pd.DataFrame:
    policy_factories = {
        "fixed_ab": lambda s: FixedABPolicy(n_arms=2, probabilities=[0.5, 0.5], seed=s),
        "epsilon_greedy": lambda s: EpsilonGreedyPolicy(n_arms=2, epsilon=0.1, seed=s),
        "ucb1": lambda s: UCB1Policy(n_arms=2, seed=s),
        "thompson_sampling": lambda s: ThompsonSamplingPolicy(n_arms=2, seed=s),
    }
    rows: list[dict[str, float | int | str]] = []
    for idx, (name, factory) in enumerate(policy_factories.items()):
        env = BernoulliBanditEnv([p0, p1], horizon=horizon, seed=seed + idx)
        policy = factory(seed + 1_000_003 + idx)
        logs = _collect_policy_logs(policy, env, horizon)
        if (logs["group"] == "control").sum() == 0 or (logs["group"] == "treatment").sum() == 0:
            continue
        result = run_ab_inference(logs, alpha=alpha)
        rows.append(
            {
                "policy": name,
                "n_control": result.n_control,
                "n_treatment": result.n_treatment,
                "control_rate": result.control_rate,
                "treatment_rate": result.treatment_rate,
                "ate": result.ate,
                "p_value": result.p_value,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "reject_null": result.reject_null,
            }
        )
    return pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare naive A/B inference applied to fixed and adaptive policies.")
    parser.add_argument("--p0", type=float, default=0.05)
    parser.add_argument("--p1", type=float, default=0.06)
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-path", default="outputs/ab_vs_mab_inference/summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = run_single_experiment_suite(p0=args.p0, p1=args.p1, horizon=args.horizon, seed=args.seed, alpha=args.alpha)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(frame)


if __name__ == "__main__":
    main()
