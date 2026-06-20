from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.bandits.epsilon_greedy import EpsilonGreedyPolicy
from src.bandits.fixed_ab import FixedABPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.bandits.ucb1 import UCB1Policy
from src.environments.bernoulli import BernoulliBanditEnv
from src.evaluation.summary import summarize_runs
from src.experiments.configs import ExperimentConfig
from src.experiments.runner import compare_policies

SCENARIOS: dict[str, list[float]] = {
  # E1 baseline: best arm 5%, spread ~2 п.п.
    "baseline": [0.03, 0.035, 0.04, 0.05, 0.045],
    # Arms within ~0.8 п.п. — exploration дороже.
    "small_gap": [0.046, 0.048, 0.05, 0.052, 0.054],
    # Один явный лидер 15% CTR — exploitation выигрывает быстрее.
    "large_gap": [0.01, 0.02, 0.03, 0.15, 0.04],
}

POLICY_FACTORIES = {
    "fixed_ab": lambda n_arms, seed: FixedABPolicy(n_arms=n_arms, seed=seed),
    "epsilon_greedy": lambda n_arms, seed: EpsilonGreedyPolicy(n_arms=n_arms, epsilon=0.1, seed=seed),
    "ucb1": lambda n_arms, seed: UCB1Policy(n_arms=n_arms, seed=seed),
    "thompson_sampling": lambda n_arms, seed: ThompsonSamplingPolicy(n_arms=n_arms, seed=seed),
}


def run_scenario(
    *,
    scenario_name: str,
    reward_probs: list[float],
    horizon: int,
    seeds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs: list[ExperimentConfig] = []
    for seed in range(seeds):
        for policy_name, policy_factory in POLICY_FACTORIES.items():
            configs.append(
                ExperimentConfig(
                    run_id=f"{scenario_name}_{policy_name}_seed{seed}",
                    seed=seed,
                    horizon=horizon,
                    policy_factory=policy_factory,
                    environment_factory=lambda s, rp=reward_probs, h=horizon: BernoulliBanditEnv(
                        reward_probs=rp,
                        horizon=h,
                        seed=s,
                    ),
                )
            )
    results = compare_policies(configs)
    results["scenario"] = scenario_name
    summary = summarize_runs(results)
    summary["scenario"] = scenario_name
    return results, summary


def run_all_scenarios(*, horizon: int, seeds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for name, probs in SCENARIOS.items():
        results, summary = run_scenario(
            scenario_name=name,
            reward_probs=probs,
            horizon=horizon,
            seeds=seeds,
        )
        result_frames.append(results)
        summary_frames.append(summary)
    return pd.concat(result_frames, ignore_index=True), pd.concat(summary_frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic gap scenarios (baseline / small_gap / large_gap).")
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/synthetic_scenarios")
    parser.add_argument("--list-scenarios", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_scenarios:
        print(json.dumps(SCENARIOS, indent=2))
        return

    results, summary = run_all_scenarios(horizon=args.horizon, seeds=args.seeds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    pivot = (
        summary.groupby(["scenario", "policy_name"])[["final_cumulative_regret", "suboptimal_share"]]
        .mean()
        .reset_index()
    )
    pivot.to_csv(out_dir / "summary_by_scenario.csv", index=False)
    print(pivot.sort_values(["scenario", "final_cumulative_regret"]).to_string(index=False))


if __name__ == "__main__":
    main()
