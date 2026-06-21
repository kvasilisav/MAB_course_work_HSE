from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.bandits.epsilon_greedy import EpsilonGreedyPolicy
from src.bandits.fixed_ab import FixedABPolicy
from src.bandits.linucb import LinUCBPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.bandits.ucb1 import UCB1Policy
from src.environments.bernoulli import BernoulliBanditEnv
from src.environments.contextual_bernoulli import ContextualBernoulliBanditEnv
from src.environments.logged_clicks import LoggedClicksBanditEnv
from src.evaluation.summary import summarize_runs
from src.experiments.configs import ExperimentConfig
from src.experiments.runner import compare_policies
from src.ope.replay import compare_policies_replay
from src.pipeline.loader import load_events

DEFAULT_POLICIES = ["fixed_ab", "epsilon_greedy", "ucb1", "thompson_sampling"]


def _parse_policy_list(raw: str | None, include_linucb: bool) -> list[str]:
    if raw:
        policies = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        policies = list(DEFAULT_POLICIES)
        if include_linucb:
            policies.append("linucb")
    return policies


def _build_policy_factory(
    policy_name: str,
    *,
    context_dim: int = 1,
    linucb_alpha: float = 1.0,
):
    mapping = {
        "fixed_ab": lambda n_arms, seed: FixedABPolicy(n_arms=n_arms, seed=seed),
        "epsilon_greedy": lambda n_arms, seed: EpsilonGreedyPolicy(n_arms=n_arms, epsilon=0.1, seed=seed),
        "ucb1": lambda n_arms, seed: UCB1Policy(n_arms=n_arms, seed=seed),
        "thompson_sampling": lambda n_arms, seed: ThompsonSamplingPolicy(n_arms=n_arms, seed=seed),
        "linucb": lambda n_arms, seed: LinUCBPolicy(
            n_arms=n_arms,
            context_dim=context_dim,
            alpha=linucb_alpha,
            seed=seed,
        ),
    }
    if policy_name not in mapping:
        raise ValueError(f"Unknown policy: {policy_name}")
    return mapping[policy_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed A/B and bandit policies.")
    parser.add_argument("--mode", default="synthetic", choices=["synthetic", "batch", "ope"])
    parser.add_argument("--env", default="bernoulli", choices=["bernoulli", "contextual"])
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--include-linucb", action="store_true")
    parser.add_argument("--policies", default=None, help="Comma-separated policy names.")
    parser.add_argument("--context-dim", type=int, default=4)
    parser.add_argument("--linucb-alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--events-path", default="data/processed/obd_events.csv")
    parser.add_argument("--assignments-path", default=None)
    parser.add_argument("--reward-probs", default="[0.03, 0.035, 0.04, 0.05, 0.045]")
    parser.add_argument("--propensity-floor", type=float, default=0.01)
    parser.add_argument("--no-freeze-policy", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = _parse_policy_list(args.policies, args.include_linucb)

    reward_probs = [float(item) for item in json.loads(args.reward_probs)]
    events = load_events(args.events_path) if args.mode in {"batch", "ope"} else []
    context_dim = max(1, int(args.context_dim))
    if events and events[0].context is not None:
        context_dim = int(events[0].context.size)

    run_horizon = args.horizon
    if args.mode in {"batch", "ope"}:
        run_horizon = len(events)

    policy_factories = {
        name: _build_policy_factory(
            name,
            context_dim=context_dim,
            linucb_alpha=args.linucb_alpha,
        )
        for name in policies
    }
    if args.mode == "ope":
        replay_summary = compare_policies_replay(
            events=events,
            policy_factories=policy_factories,
            seeds=args.seeds,
            freeze_policy=not args.no_freeze_policy,
            propensity_floor=args.propensity_floor,
            shuffle_events=not args.no_shuffle,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        replay_summary.to_csv(output_dir / "ope_summary.csv", index=False)
        print(
            replay_summary.groupby("policy_name")[
                ["acceptance_rate", "ips_estimate", "snips_estimate", "effective_sample_size"]
            ].mean().sort_values("snips_estimate", ascending=False)
        )
        return

    configs: list[ExperimentConfig] = []
    for seed in range(args.seeds):
        for policy_name in policies:
            if args.assignments_path is not None:
                assignments_path = args.assignments_path
            else:
                assignments_path = str(
                    Path(args.output_dir) / f"assignments_{policy_name}_seed{seed}.csv"
                )
            if args.mode == "batch":
                environment_factory = (
                    lambda s, loaded_events=events: LoggedClicksBanditEnv(events=loaded_events, seed=s)
                )
            elif args.env == "contextual":
                environment_factory = (
                    lambda s, cd=context_dim, h=args.horizon: ContextualBernoulliBanditEnv(
                        n_arms=5,
                        context_dim=cd,
                        horizon=h,
                        seed=s,
                    )
                )
            else:
                environment_factory = (
                    lambda s, rp=reward_probs, h=args.horizon: BernoulliBanditEnv(
                        reward_probs=rp,
                        horizon=h,
                        seed=s,
                    )
                )
            configs.append(
                ExperimentConfig(
                    run_id=f"{policy_name}_seed{seed}",
                    seed=seed,
                    horizon=run_horizon,
                    mode=args.mode,
                    batch_size=args.batch_size,
                    assignments_path=assignments_path,
                    policy_factory=policy_factories[policy_name],
                    environment_factory=environment_factory,
                )
            )

    results = compare_policies(configs)
    summary = summarize_runs(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    if not summary.empty:
        pivot = summary.groupby("policy_name")[["final_cumulative_reward", "final_cumulative_regret", "suboptimal_share"]].mean()
        print(pivot.sort_values("final_cumulative_regret"))
    else:
        print("No rows were produced. Verify environment inputs.")


if __name__ == "__main__":
    main()
