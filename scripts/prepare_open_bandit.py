from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

OBD_RAW_BASE_URL = "https://raw.githubusercontent.com/st-tech/zr-obp/master/obd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Open Bandit Dataset (OBD) CSV into project events.csv format.",
    )
    parser.add_argument(
        "--behavior-policy",
        default="random",
        choices=["random", "bts"],
        help="Logging policy used in OBD (random or bts).",
    )
    parser.add_argument(
        "--campaign",
        default="all",
        choices=["all", "men", "women"],
        help="OBD campaign subset.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Local OBD CSV path. If omitted, --download is required unless default raw file exists.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the small OBD CSV from the zr-obp GitHub repository.",
    )
    parser.add_argument("--output-path", default="data/processed/obd_events.csv")
    parser.add_argument("--raw-cache-path", default=None, help="Where to store downloaded raw OBD CSV.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional row limit after loading.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed before truncation.")
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Encode OBD position, user hashes, and item affinity into context_json.",
    )
    return parser.parse_args()


def obd_download_url(behavior_policy: str, campaign: str) -> str:
    return f"{OBD_RAW_BASE_URL}/{behavior_policy}/{campaign}/{campaign}.csv"


def default_raw_cache_path(behavior_policy: str, campaign: str) -> Path:
    return Path("data/raw") / f"obd_{behavior_policy}_{campaign}.csv"


def _hash_to_unit_float(value: str) -> float:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _affinity_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("user-item_affinity_")]


def _build_context(row: pd.Series, affinity_columns: list[str], item_id: int) -> list[float]:
    position = float(row.get("position", 1))
    position_norm = (position - 1.0) / 2.0
    affinities = [float(row[column]) for column in affinity_columns]
    affinity_at_item = float(affinities[item_id]) if 0 <= item_id < len(affinities) else 0.0
    mean_affinity = float(np.mean(affinities)) if affinities else 0.0
    user_features = [
        _hash_to_unit_float(row.get(f"user_feature_{idx}", ""))
        for idx in range(4)
        if f"user_feature_{idx}" in row.index
    ]
    return [position_norm, affinity_at_item, mean_affinity, *user_features]


def convert_obd_frame(
    frame: pd.DataFrame,
    *,
    include_context: bool = True,
    seed: int = 42,
    max_events: int | None = None,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("OBD input frame is empty.")

    working = frame.copy()
    if "item_id" not in working.columns:
        raise ValueError("OBD CSV must contain item_id column.")
    if "click" not in working.columns:
        raise ValueError("OBD CSV must contain click column.")

    propensity_column = "propensity_score" if "propensity_score" in working.columns else "action_prob"
    if propensity_column not in working.columns:
        raise ValueError("OBD CSV must contain propensity_score (or action_prob).")

    working = working.reset_index(drop=True)
    if max_events is not None and len(working) > max_events:
        working = working.sample(n=max_events, random_state=seed).sort_index().reset_index(drop=True)

    n_arms = int(working["item_id"].max()) + 1
    all_arms = list(range(n_arms))
    affinity_columns = _affinity_columns(working)

    rows: list[dict[str, object]] = []
    for idx, row in working.iterrows():
        item_id = int(row["item_id"])
        context = _build_context(row, affinity_columns, item_id) if include_context else []
        user_id_source = "|".join(str(row.get(f"user_feature_{j}", "")) for j in range(4))
        rows.append(
            {
                "event_id": int(idx + 1),
                "timestamp": row.get("timestamp", idx + 1),
                "user_id": hashlib.md5(user_id_source.encode("utf-8")).hexdigest()[:16],
                "context_json": json.dumps(context),
                "candidate_arms_json": json.dumps(all_arms),
                "chosen_arm": item_id,
                "reward": float(row["click"]),
                "propensity": float(row[propensity_column]),
            }
        )

    return pd.DataFrame(rows)


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input_path is not None:
        return Path(args.input_path)

    cache_path = Path(args.raw_cache_path) if args.raw_cache_path else default_raw_cache_path(
        args.behavior_policy,
        args.campaign,
    )
    if cache_path.exists():
        return cache_path
    if args.download:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        url = obd_download_url(args.behavior_policy, args.campaign)
        print(f"Downloading OBD CSV from {url}")
        urlretrieve(url, cache_path)
        return cache_path

    raise FileNotFoundError(
        "OBD source file was not found. Pass --input-path, place the CSV in data/raw/, or use --download."
    )


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args)
    frame = pd.read_csv(input_path)
    result = convert_obd_frame(
        frame,
        include_context=args.include_context,
        seed=args.seed,
        max_events=args.max_events,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(
        f"Prepared {len(result)} OBD events at {output_path} "
        f"(behavior_policy={args.behavior_policy}, campaign={args.campaign}, source={input_path})"
    )


if __name__ == "__main__":
    main()
