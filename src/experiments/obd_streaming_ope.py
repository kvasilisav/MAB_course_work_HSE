from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Callable
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from src.bandits.base import BanditPolicy
from src.bandits.epsilon_greedy import EpsilonGreedyPolicy
from src.bandits.fixed_ab import FixedABPolicy
from src.bandits.linucb import LinUCBPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.bandits.ucb1 import UCB1Policy
from scripts.prepare_open_bandit import obd_download_url


CAMPAIGN_N_ARMS = {
    "all": 80,
    "men": 34,
    "women": 46,
}


class StreamingReplayState:
    def __init__(self, policy_name: str, seed: int, policy: BanditPolicy) -> None:
        self.policy_name = policy_name
        self.seed = seed
        self.policy = policy
        self.total_events = 0
        self.valid_logged_rows = 0
        self.accepted_events = 0
        self.accepted_reward_sum = 0.0
        self.weight_sum = 0.0
        self.weighted_reward_sum = 0.0
        self.weight_sq_sum = 0.0

    def observe(
        self,
        *,
        logged_arm: int,
        reward: float,
        propensity: float,
        context: np.ndarray | None,
    ) -> None:
        self.total_events += 1
        if propensity <= 0.0:
            return
        self.valid_logged_rows += 1
        chosen_arm = int(self.policy.select_arm(context))
        if chosen_arm != logged_arm:
            self.weight_sq_sum += 0.0
            return

        weight = 1.0 / propensity
        self.accepted_events += 1
        self.accepted_reward_sum += reward
        self.weight_sum += weight
        self.weighted_reward_sum += weight * reward
        self.weight_sq_sum += weight * weight
        self.policy.update(chosen_arm, reward, context)

    def to_row(self) -> dict[str, float | int | str]:
        denominator = self.total_events if self.total_events > 0 else 1
        accepted_mean_reward = (
            self.accepted_reward_sum / self.accepted_events
            if self.accepted_events > 0
            else 0.0
        )
        ips_estimate = self.weighted_reward_sum / denominator
        snips_estimate = (
            self.weighted_reward_sum / self.weight_sum
            if self.weight_sum > 0.0
            else 0.0
        )
        ess = (
            (self.weight_sum * self.weight_sum) / self.weight_sq_sum
            if self.weight_sq_sum > 0.0
            else 0.0
        )
        return {
            "policy_name": self.policy_name,
            "seed": self.seed,
            "total_events": self.total_events,
            "valid_logged_rows": self.valid_logged_rows,
            "accepted_events": self.accepted_events,
            "acceptance_rate": self.accepted_events / denominator,
            "accepted_mean_reward": accepted_mean_reward,
            "ips_estimate": ips_estimate,
            "snips_estimate": snips_estimate,
            "effective_sample_size": ess,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run streaming OPE directly on Open Bandit Dataset CSV files. "
            "This is intended for partial/full OBD experiments where converting "
            "millions of rows into events.csv would be inefficient."
        )
    )
    parser.add_argument("--input-path", default=None, help="Local OBD CSV/CSV.GZ/ZIP path.")
    parser.add_argument("--input-url", default=None, help="Direct URL to an OBD CSV-like file.")
    parser.add_argument(
        "--download-small",
        action="store_true",
        help="Download the small GitHub OBD CSV if no local full file is available.",
    )
    parser.add_argument("--behavior-policy", default="random", choices=["random", "bts"])
    parser.add_argument("--campaign", default="all", choices=["all", "men", "women"])
    parser.add_argument("--raw-cache-path", default=None)
    parser.add_argument("--output-dir", default="outputs/obd_streaming_ope")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-arms", type=int, default=None)
    parser.add_argument("--include-linucb", action="store_true")
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Build context vectors for LinUCB. Non-contextual policies ignore this.",
    )
    return parser.parse_args()


def _default_raw_cache_path(behavior_policy: str, campaign: str) -> Path:
    return Path("data/raw") / f"obd_{behavior_policy}_{campaign}_stream.csv"


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input_path is not None:
        return Path(args.input_path)

    cache_path = (
        Path(args.raw_cache_path)
        if args.raw_cache_path is not None
        else _default_raw_cache_path(args.behavior_policy, args.campaign)
    )
    if cache_path.exists():
        return cache_path

    if args.input_url is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading OBD file from {args.input_url}")
        urlretrieve(args.input_url, cache_path)
        return cache_path

    if args.download_small:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        url = obd_download_url(args.behavior_policy, args.campaign)
        print(f"Downloading small OBD CSV from {url}")
        urlretrieve(url, cache_path)
        return cache_path

    raise FileNotFoundError(
        "No OBD input was provided. Pass --input-path for a local full/partial OBD "
        "CSV, --input-url for a direct file URL, or --download-small for the GitHub sample."
    )


def _hash_to_unit_float(value: object) -> float:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _build_context(row: pd.Series, affinity_columns: list[str], item_id: int) -> np.ndarray:
    position = float(row.get("position", 1))
    position_norm = (position - 1.0) / 2.0
    affinity_at_item = (
        float(row[affinity_columns[item_id]])
        if 0 <= item_id < len(affinity_columns)
        else 0.0
    )
    if affinity_columns:
        mean_affinity = float(row[affinity_columns].astype(float).mean())
    else:
        mean_affinity = 0.0
    hashed_features = [
        _hash_to_unit_float(row.get(f"user_feature_{idx}", ""))
        for idx in range(4)
        if f"user_feature_{idx}" in row.index
    ]
    return np.asarray([position_norm, affinity_at_item, mean_affinity, *hashed_features], dtype=float)


def _policy_factories(context_dim: int, include_linucb: bool) -> dict[str, Callable[[int, int], BanditPolicy]]:
    factories: dict[str, Callable[[int, int], BanditPolicy]] = {
        "fixed_ab": lambda n_arms, seed: FixedABPolicy(n_arms=n_arms, seed=seed),
        "epsilon_greedy": lambda n_arms, seed: EpsilonGreedyPolicy(
            n_arms=n_arms,
            epsilon=0.1,
            seed=seed,
        ),
        "ucb1": lambda n_arms, seed: UCB1Policy(n_arms=n_arms, seed=seed),
        "thompson_sampling": lambda n_arms, seed: ThompsonSamplingPolicy(n_arms=n_arms, seed=seed),
    }
    if include_linucb:
        factories["linucb"] = lambda n_arms, seed: LinUCBPolicy(
            n_arms=n_arms,
            context_dim=context_dim,
            alpha=1.0,
            seed=seed,
        )
    return factories


def run_streaming_obd_ope(
    *,
    input_path: Path,
    behavior_policy: str,
    campaign: str,
    n_arms: int | None,
    seeds: int,
    chunksize: int,
    max_events: int | None,
    include_context: bool,
    include_linucb: bool,
) -> pd.DataFrame:
    resolved_n_arms = n_arms if n_arms is not None else CAMPAIGN_N_ARMS[campaign]
    context_dim = 7 if include_context else 1
    factories = _policy_factories(context_dim=context_dim, include_linucb=include_linucb)
    states = [
        StreamingReplayState(
            policy_name=policy_name,
            seed=seed,
            policy=factory(resolved_n_arms, seed),
        )
        for seed in range(seeds)
        for policy_name, factory in factories.items()
    ]

    processed = 0
    reader = pd.read_csv(input_path, chunksize=chunksize, compression="infer")
    for chunk in reader:
        required = {"item_id", "click"}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"OBD CSV misses required columns: {sorted(missing)}")

        propensity_col = "propensity_score" if "propensity_score" in chunk.columns else "action_prob"
        if propensity_col not in chunk.columns:
            raise ValueError("OBD CSV must contain propensity_score (or action_prob).")

        if max_events is not None:
            remaining = max_events - processed
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        affinity_columns = [column for column in chunk.columns if column.startswith("user-item_affinity_")]
        if include_context:
            iterator = (
                (
                    int(row["item_id"]),
                    float(row["click"]),
                    float(row[propensity_col]),
                    _build_context(row, affinity_columns, int(row["item_id"])),
                )
                for _, row in chunk.iterrows()
            )
        else:
            iterator = zip(
                chunk["item_id"].astype(int).to_numpy(),
                chunk["click"].astype(float).to_numpy(),
                chunk[propensity_col].astype(float).to_numpy(),
                [None] * len(chunk),
            )

        for logged_arm, reward, propensity, context in iterator:
            for state in states:
                state.observe(
                    logged_arm=int(logged_arm),
                    reward=float(reward),
                    propensity=float(propensity),
                    context=context,
                )
        processed += len(chunk)
        if max_events is not None and processed >= max_events:
            break

    frame = pd.DataFrame([state.to_row() for state in states])
    frame.insert(0, "behavior_policy", behavior_policy)
    frame.insert(1, "campaign", campaign)
    frame.insert(2, "source_path", str(input_path))
    return frame


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_streaming_obd_ope(
        input_path=input_path,
        behavior_policy=args.behavior_policy,
        campaign=args.campaign,
        n_arms=args.n_arms,
        seeds=args.seeds,
        chunksize=args.chunksize,
        max_events=args.max_events,
        include_context=args.include_context,
        include_linucb=args.include_linucb,
    )
    summary.to_csv(output_dir / "ope_summary.csv", index=False)
    print(
        summary.groupby("policy_name")[
            ["acceptance_rate", "ips_estimate", "snips_estimate", "effective_sample_size"]
        ].mean().sort_values("snips_estimate", ascending=False)
    )
    print(f"Saved streaming OPE summary to {output_dir / 'ope_summary.csv'}")


if __name__ == "__main__":
    main()
