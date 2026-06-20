"""Generate figures for coursework (outputs/figures/)."""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.bandits.epsilon_greedy import EpsilonGreedyPolicy
from src.bandits.fixed_ab import FixedABPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.environments.bernoulli import BernoulliBanditEnv
from src.environments.logged_clicks import LoggedClicksBanditEnv
from src.evaluation.plotting import plot_metric_by_policy
from src.evaluation.summary import summarize_runs
from src.experiments.configs import ExperimentConfig
from src.experiments.runner import compare_policies
from src.pipeline.loader import load_events

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_e8b_regret() -> Path:
    path = ROOT / "outputs/extended_full/e8b_contextual_synthetic/results.csv"
    df = pd.read_csv(path)
    fig = plot_metric_by_policy(df, "cumulative_regret")
    fig.suptitle("E8b: средний cumulative regret (контекстная синтетика)")
    return _save(fig, "fig_3_1_e8b_regret_curves.png")


def fig_ope_ess() -> Path:
    df = pd.read_csv(ROOT / "outputs/extended_full/ope_matrix_summary.csv")
    pivot = df.pivot(index="policy_name", columns="dataset", values="effective_sample_size")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("E2b/c: ESS по политике и подвыборке OBD")
    ax.set_xlabel("Политика")
    ax.set_ylabel("ESS")
    ax.legend(title="dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_3_2_ope_ess_matrix.png")


def fig_p1_latency() -> Path:
    df = pd.read_csv(ROOT / "outputs/extended_full/policy_latency.csv")
    subset = df[df["n_arms"] == 80].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(subset["policy_name"], subset["total_decision_us_mean"], color="steelblue")
    ax.set_yscale("log")
    ax.set_title("P1: задержка одного решения (80 рук, log scale)")
    ax.set_ylabel("мкс")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_3_3_p1_latency_80arms.png")


def fig_synthetic_gaps() -> Path:
    df = pd.read_csv(ROOT / "outputs/synthetic_scenarios/summary_by_scenario.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario, group in df.groupby("scenario"):
        ax.bar(
            [f"{scenario}\n{p}" for p in group["policy_name"]],
            group["final_cumulative_regret"],
            label=scenario,
        )
    ax.set_title("Синтетика: regret по сценариям CTR (baseline / small_gap / large_gap)")
    ax.set_ylabel("final cumulative regret")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save(fig, "fig_3_4_synthetic_gap_regret.png")


def fig_obd_ctr_top20() -> Path:
    stats = pd.read_csv(ROOT / "outputs/obd_pair/item_ctr_stats.csv").head(20)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(stats["item_id"].astype(str), stats["ctr"], color="coral")
    ax.set_title("OBD (10k random/all): CTR топ-20 товаров по показам")
    ax.set_xlabel("item_id")
    ax.set_ylabel("CTR")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_3_5_obd_ctr_top20.png")


def fig_epsilon_ablation() -> Path:
    reward_probs = [0.03, 0.035, 0.04, 0.05, 0.045]
    configs = []
    for seed, epsilon in itertools.product(range(10), [0.05, 0.1, 0.2]):
        configs.append(
            ExperimentConfig(
                run_id=f"eps{epsilon}_seed{seed}",
                seed=seed,
                horizon=5000,
                policy_factory=lambda n_arms, s, eps=epsilon: EpsilonGreedyPolicy(
                    n_arms=n_arms, epsilon=eps, seed=s
                ),
                environment_factory=lambda s, rp=reward_probs: BernoulliBanditEnv(
                    reward_probs=rp, horizon=5000, seed=s
                ),
            )
        )
    summary = summarize_runs(compare_policies(configs))
    summary["epsilon"] = summary["run_id"].str.extract(r"eps([0-9.]+)_")[0].astype(float)
    agg = summary.groupby("epsilon")["final_cumulative_regret"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    agg.plot(kind="bar", ax=ax, color="seagreen")
    ax.set_title("E6: regret vs ε (ε-greedy, 10 seed)")
    ax.set_ylabel("mean regret")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_3_6_epsilon_regret.png")


def fig_ts_priors() -> Path:
    reward_probs = [0.03, 0.035, 0.04, 0.05, 0.045]
    prior_grid = [(1.0, 1.0), (1.0, 5.0), (5.0, 1.0), (10.0, 10.0)]
    configs = []
    for seed, (a, b) in itertools.product(range(8), prior_grid):
        configs.append(
            ExperimentConfig(
                run_id=f"ts_a{a}_b{b}_seed{seed}",
                seed=seed,
                horizon=5000,
                policy_factory=lambda n_arms, s, alpha=a, beta=b: ThompsonSamplingPolicy(
                    n_arms=n_arms, alpha_prior=alpha, beta_prior=beta, seed=s
                ),
                environment_factory=lambda s, rp=reward_probs: BernoulliBanditEnv(
                    reward_probs=rp, horizon=5000, seed=s
                ),
            )
        )
    summary = summarize_runs(compare_policies(configs))
    summary["prior"] = summary["run_id"].str.extract(r"ts_a([0-9.]+)_b([0-9.]+)").agg("_".join, axis=1)
    agg = summary.groupby("prior")["final_cumulative_regret"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    agg.plot(kind="bar", ax=ax, color="mediumpurple")
    ax.set_title("Абляция TS priors (8 seed, E1-среда)")
    ax.set_ylabel("mean regret")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, "fig_3_7_ts_priors_regret.png")


def fig_batch_size() -> Path:
    events_path = ROOT / "data/processed/obd_events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing OBD events at {events_path}")
    events = load_events(events_path)[:10_000]

    batch_sizes = [1, 50, 200, 1000]
    configs = []
    for seed, batch_size in itertools.product(range(5), batch_sizes):
        for policy_name, factory in {
            "fixed_ab": lambda n_arms, s: FixedABPolicy(n_arms=n_arms, seed=s),
            "epsilon_greedy": lambda n_arms, s: EpsilonGreedyPolicy(n_arms=n_arms, epsilon=0.1, seed=s),
        }.items():
            configs.append(
                ExperimentConfig(
                    run_id=f"{policy_name}_bs{batch_size}_seed{seed}",
                    seed=seed,
                    horizon=len(events),
                    mode="batch",
                    batch_size=batch_size,
                    policy_factory=factory,
                    environment_factory=lambda s, ev=events: LoggedClicksBanditEnv(events=ev, seed=s),
                )
            )
    summary = summarize_runs(compare_policies(configs))
    summary["batch_size"] = summary["run_id"].str.extract(r"_bs(\d+)_")[0].astype(int)
    agg = summary.groupby(["policy_name", "batch_size"])["final_cumulative_regret"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    agg.plot(kind="line", marker="o", ax=ax)
    ax.set_title("Абляция batch_size (OBD batch, 10k событий)")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("mean regret")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_3_8_batch_size_regret.png")


def main() -> None:
    paths = [
        fig_e8b_regret(),
        fig_ope_ess(),
        fig_p1_latency(),
        fig_synthetic_gaps(),
        fig_obd_ctr_top20(),
        fig_epsilon_ablation(),
        fig_ts_priors(),
        fig_batch_size(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
