from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ab_testing.inference import run_ab_inference
from src.ab_testing.weighted_inference import run_weighted_ab_inference
from src.bandits.epsilon_greedy import EpsilonGreedyPolicy
from src.bandits.fixed_ab import FixedABPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.bandits.ucb1 import UCB1Policy
from src.data.obd_pairwise import convert_obd_to_pairwise_events
from src.ope.replay import compare_policies_replay
from src.pipeline.loader import load_events


def _build_policy_factory(policy_name: str):
    mapping = {
        "fixed_ab": lambda n_arms, seed: FixedABPolicy(
            n_arms=n_arms,
            probabilities=[0.5, 0.5],
            seed=seed,
        ),
        "epsilon_greedy": lambda n_arms, seed: EpsilonGreedyPolicy(n_arms=n_arms, epsilon=0.1, seed=seed),
        "ucb1": lambda n_arms, seed: UCB1Policy(n_arms=n_arms, seed=seed),
        "thompson_sampling": lambda n_arms, seed: ThompsonSamplingPolicy(n_arms=n_arms, seed=seed),
    }
    if policy_name not in mapping:
        raise ValueError(f"Unknown policy: {policy_name}")
    return mapping[policy_name]


def _pairwise_propensity_frame(events_df: pd.DataFrame) -> pd.DataFrame:
    frame = events_df.copy()
    p0_mean = float(frame.loc[frame["chosen_arm"] == 0, "propensity"].mean()) if (frame["chosen_arm"] == 0).any() else 0.01
    p1_mean = float(frame.loc[frame["chosen_arm"] == 1, "propensity"].mean()) if (frame["chosen_arm"] == 1).any() else 0.01
    p0_mean = max(p0_mean, 1e-6)
    p1_mean = max(p1_mean, 1e-6)

    frame["arm"] = frame["chosen_arm"].astype(int)
    frame["propensity_control"] = frame.apply(
        lambda row: float(row["propensity"]) if int(row["arm"]) == 0 else p0_mean,
        axis=1,
    )
    frame["propensity_treatment"] = frame.apply(
        lambda row: float(row["propensity"]) if int(row["arm"]) == 1 else p1_mean,
        axis=1,
    )
    return frame


def _run_pairwise_inference(events_df: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    ab_frame = events_df.rename(columns={"chosen_arm": "arm", "reward": "reward"})[
        ["arm", "reward"]
    ].copy()
    ab_frame["group"] = ab_frame["arm"].map({0: "control", 1: "treatment"})
    naive = run_ab_inference(ab_frame, alpha=alpha)

    ips_frame = _pairwise_propensity_frame(events_df)
    ips = run_weighted_ab_inference(ips_frame, alpha=alpha)

    agg = events_df.groupby("chosen_arm").agg(
        impressions=("reward", "count"),
        clicks=("reward", "sum"),
    )
    return pd.DataFrame(
        [
            {
                "method": "naive_ab_pairwise",
                "reject_null": naive.reject_null,
                "p_value": naive.p_value,
                "ate": naive.ate,
                "n_control": naive.n_control,
                "n_treatment": naive.n_treatment,
            },
            {
                "method": "ips_weighted_pairwise",
                "reject_null": ips.reject_null,
                "p_value": ips.p_value,
                "ate": ips.ate_ips,
                "ess": ips.effective_sample_size,
                "n_observations": ips.n_observations,
            },
            {
                "method": "observed_ctr_arm0",
                "ctr": float(agg.loc[0, "clicks"] / agg.loc[0, "impressions"]) if 0 in agg.index else 0.0,
                "impressions": int(agg.loc[0, "impressions"]) if 0 in agg.index else 0,
            },
            {
                "method": "observed_ctr_arm1",
                "ctr": float(agg.loc[1, "clicks"] / agg.loc[1, "impressions"]) if 1 in agg.index else 0.0,
                "impressions": int(agg.loc[1, "impressions"]) if 1 in agg.index else 0,
            },
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E13: OBD pairwise OPE + 2-arm inference.")
    parser.add_argument("--input-path", default="data/raw/obd_random_all.csv")
    parser.add_argument("--pair-selection-path", default="outputs/obd_pair/pair_selection.csv")
    parser.add_argument("--policies", default="fixed_ab,thompson_sampling,epsilon_greedy,ucb1")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", default="outputs/obd_pair")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_df = pd.read_csv(args.pair_selection_path)
    item_a = int(pair_df.iloc[0]["item_a"])
    item_b = int(pair_df.iloc[0]["item_b"])

    raw = pd.read_csv(args.input_path)
    events_df = convert_obd_to_pairwise_events(raw, item_a=item_a, item_b=item_b)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "pairwise_events.csv"
    events_df.to_csv(events_path, index=False)

    events = load_events(events_path)
    policy_names = [p.strip() for p in args.policies.split(",") if p.strip()]
    factories = {name: _build_policy_factory(name) for name in policy_names}
    ope_summary = compare_policies_replay(events=events, policy_factories=factories, seeds=args.seeds)
    ope_summary["item_a"] = item_a
    ope_summary["item_b"] = item_b
    ope_summary["n_pair_events"] = len(events_df)
    ope_summary.to_csv(out_dir / "ope_summary.csv", index=False)

    inference = _run_pairwise_inference(events_df, alpha=args.alpha)
    inference["item_a"] = item_a
    inference["item_b"] = item_b
    inference.to_csv(out_dir / "inference_summary.csv", index=False)

    meta = {
        "item_a": item_a,
        "item_b": item_b,
        "n_pair_events": len(events_df),
        "policies": policy_names,
        "seeds": args.seeds,
    }
    (out_dir / "e13_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Pairwise OPE on {len(events_df)} events (items {item_a} vs {item_b})")
    print(
        ope_summary.groupby("policy_name")[
            ["snips_estimate", "acceptance_rate", "effective_sample_size"]
        ].mean()
    )
    print("\nPairwise inference:")
    print(inference.to_string(index=False))


if __name__ == "__main__":
    main()
