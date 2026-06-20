from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare lightweight click events for batch bandit experiments.")
    parser.add_argument("--input-path", required=True, help="Input CSV with at least user_id and arm_id columns.")
    parser.add_argument("--output-path", default="data/processed/events.csv")
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    frame = pd.read_csv(args.input_path).head(args.max_events).copy()

    if "user_id" not in frame.columns:
        frame["user_id"] = np.arange(len(frame))
    if "arm_id" not in frame.columns:
        # Fallback for generic tabular datasets.
        frame["arm_id"] = rng.integers(0, 5, size=len(frame))
    if "reward" not in frame.columns:
        frame["reward"] = rng.binomial(1, 0.1, size=len(frame))

    all_arms = sorted(int(value) for value in frame["arm_id"].unique())
    frame["event_id"] = np.arange(1, len(frame) + 1)
    frame["timestamp"] = frame["event_id"]
    frame["candidate_arms_json"] = [json.dumps(all_arms)] * len(frame)
    frame["chosen_arm"] = frame["arm_id"].astype(int)
    frame["context_json"] = [json.dumps([])] * len(frame)
    frame["propensity"] = 1.0 / max(len(all_arms), 1)

    result = frame[
        ["event_id", "timestamp", "user_id", "context_json", "candidate_arms_json", "chosen_arm", "reward", "propensity"]
    ]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Prepared {len(result)} events at {output_path}")


if __name__ == "__main__":
    main()
