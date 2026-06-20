from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ab_testing.bandit_logs import logs_to_ab_frame, simulate_thompson_logs_with_propensity
from src.ab_testing.inference import run_ab_inference
from src.ab_testing.weighted_inference import run_weighted_ab_inference
from src.bandits.epsilon_greedy import EpsilonGreedyPolicy
from src.bandits.fixed_ab import FixedABPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.environments.bernoulli import BernoulliBanditEnv
from src.evaluation.summary import summarize_runs
from src.experiments.configs import ExperimentConfig
from src.experiments.runner import compare_policies


def _simulate_fixed_ab_logs(
    *,
    p_control: float,
    p_treatment: float,
    horizon: int,
    seed: int,
) -> pd.DataFrame:
    env = BernoulliBanditEnv([p_control, p_treatment], horizon=horizon, seed=seed)
    policy = FixedABPolicy(n_arms=2, probabilities=[0.5, 0.5], seed=seed + 17)
    env.reset()
    policy.reset()
    rows: list[dict[str, float | int | str]] = []
    for step in range(horizon):
        arm = int(policy.select_arm())
        reward, _ = env.step(arm)
        rows.append(
            {
                "step": step,
                "arm": arm,
                "reward": float(reward),
                "group": "treatment" if arm == 1 else "control",
            }
        )
        policy.update(arm, reward)
    return pd.DataFrame(rows)


def _run_online_comparison(
    *,
    p_control: float,
    p_treatment: float,
    horizon: int,
    n_seeds: int,
    epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reward_probs = [p_control, p_treatment]
    policy_specs: list[tuple[str, object]] = [
        ("fixed_ab", lambda n_arms, seed: FixedABPolicy(n_arms=n_arms, probabilities=[0.5, 0.5], seed=seed)),
        (
            "thompson_sampling",
            lambda n_arms, seed: ThompsonSamplingPolicy(n_arms=n_arms, seed=seed),
        ),
        (
            "epsilon_greedy",
            lambda n_arms, seed: EpsilonGreedyPolicy(n_arms=n_arms, epsilon=epsilon, seed=seed),
        ),
    ]

    configs: list[ExperimentConfig] = []
    for seed in range(n_seeds):
        for policy_name, factory in policy_specs:
            configs.append(
                ExperimentConfig(
                    run_id=f"{policy_name}_seed{seed}",
                    seed=seed,
                    horizon=horizon,
                    mode="synthetic",
                    batch_size=200,
                    assignments_path=None,
                    policy_factory=factory,
                    environment_factory=lambda s, rp=reward_probs, h=horizon: BernoulliBanditEnv(
                        reward_probs=rp,
                        horizon=h,
                        seed=s,
                    ),
                )
            )
    results = compare_policies(configs)
    summary = summarize_runs(results)
    return results, summary


def _run_inference_block(
    *,
    p_control: float,
    p_treatment: float,
    horizon: int,
    n_trials: int,
    alpha: float,
    propensity_mc_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str | bool]] = []
    for seed in tqdm(range(n_trials), desc="E12 inference", unit="trial"):
        ts_logs = simulate_thompson_logs_with_propensity(
            p_control=p_control,
            p_treatment=p_treatment,
            horizon=horizon,
            seed=seed,
            propensity_mc_samples=propensity_mc_samples,
        )
        fixed_logs = _simulate_fixed_ab_logs(
            p_control=p_control,
            p_treatment=p_treatment,
            horizon=horizon,
            seed=seed + 200_003,
        )
        evaluations = [
            ("naive_ab_full_ts", run_ab_inference(logs_to_ab_frame(ts_logs), alpha=alpha)),
            ("naive_ab_fixed_only", run_ab_inference(logs_to_ab_frame(fixed_logs), alpha=alpha)),
            ("ips_weighted_ab", run_weighted_ab_inference(ts_logs, alpha=alpha, bootstrap_seed=seed)),
        ]
        for method, result in evaluations:
            rows.append(
                {
                    "method": method,
                    "reject_null": bool(result.reject_null),
                    "p_value": float(result.p_value),
                    "seed": seed,
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("method", as_index=False)
        .agg(
            rejection_rate=("reject_null", "mean"),
            mean_p_value=("p_value", "mean"),
            trials=("reject_null", "count"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    return summary, detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E12: pairwise product A/B — online bandits vs fixed_ab + inference protocols.",
    )
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--p-control", type=float, default=0.05)
    parser.add_argument("--p-treatment", type=float, default=0.08, help="Treatment CTR (+3 pp default).")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--inference-trials", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--propensity-mc", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/product_ab")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"E12 product A/B: horizon={args.horizon}, seeds={args.seeds}, "
        f"CTR={args.p_control:.3f}/{args.p_treatment:.3f}"
    )
    results, summary = _run_online_comparison(
        p_control=args.p_control,
        p_treatment=args.p_treatment,
        horizon=args.horizon,
        n_seeds=args.seeds,
        epsilon=args.epsilon,
    )
    results.to_csv(out_dir / "e12_online_results.csv", index=False)
    summary.to_csv(out_dir / "e12_online_summary.csv", index=False)

    inference_summary, inference_detail = _run_inference_block(
        p_control=args.p_control,
        p_treatment=args.p_treatment,
        horizon=args.horizon,
        n_trials=args.inference_trials,
        alpha=args.alpha,
        propensity_mc_samples=args.propensity_mc,
    )
    inference_summary.to_csv(out_dir / "e12_inference_summary.csv", index=False)
    inference_detail.to_csv(out_dir / "e12_inference_detail.csv", index=False)

    scenario = {
        "scenario": "product_ab_pair",
        "n_arms": 2,
        "p_control": args.p_control,
        "p_treatment": args.p_treatment,
        "delta_ctr_pp": (args.p_treatment - args.p_control) * 100,
        "horizon": args.horizon,
        "seeds": args.seeds,
        "policies": ["fixed_ab", "thompson_sampling", "epsilon_greedy"],
    }
    (out_dir / "e12_scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")

    print("\nOnline summary (mean):")
    print(
        summary.groupby("policy_name")[
            ["final_cumulative_reward", "final_cumulative_regret", "suboptimal_share"]
        ].mean()
    )
    print("\nInference summary:")
    print(inference_summary.to_string(index=False))


if __name__ == "__main__":
    main()
