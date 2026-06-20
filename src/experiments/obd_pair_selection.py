from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.obd_pairwise import item_ctr_table, select_top_similar_ctr_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA: select OBD item pair for pairwise A/B projection.")
    parser.add_argument("--input-path", default="data/raw/obd_random_all.csv")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-impressions", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/obd_pair")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_path)
    stats = item_ctr_table(frame)
    pair = select_top_similar_ctr_pair(
        frame,
        top_k=args.top_k,
        min_impressions=args.min_impressions,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / "item_ctr_stats.csv", index=False)

    pair_row = pd.DataFrame(
        [
            {
                "item_a": pair.item_a,
                "item_b": pair.item_b,
                "impressions_a": pair.impressions_a,
                "impressions_b": pair.impressions_b,
                "clicks_a": pair.clicks_a,
                "clicks_b": pair.clicks_b,
                "ctr_a": pair.ctr_a,
                "ctr_b": pair.ctr_b,
                "ctr_gap": pair.ctr_gap,
                "top_k_pool": pair.rank_pool,
                "min_impressions": args.min_impressions,
                "source": args.input_path,
            }
        ]
    )
    pair_row.to_csv(out_dir / "pair_selection.csv", index=False)
    (out_dir / "pair_selection.json").write_text(
        json.dumps(pair_row.iloc[0].to_dict(), indent=2),
        encoding="utf-8",
    )

    print(f"Selected pair: item {pair.item_a} vs {pair.item_b}")
    print(f"CTR: {pair.ctr_a:.4f} vs {pair.ctr_b:.4f} (gap {pair.ctr_gap:.4f})")
    print(f"Impressions: {pair.impressions_a} / {pair.impressions_b}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
