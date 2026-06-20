from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ItemPairSelection:
    item_a: int
    item_b: int
    impressions_a: int
    impressions_b: int
    clicks_a: int
    clicks_b: int
    ctr_a: float
    ctr_b: float
    ctr_gap: float
    rank_pool: int


def item_ctr_table(frame: pd.DataFrame) -> pd.DataFrame:
    if "item_id" not in frame.columns or "click" not in frame.columns:
        raise ValueError("frame must contain item_id and click columns.")
    stats = (
        frame.groupby("item_id", as_index=False)
        .agg(impressions=("click", "count"), clicks=("click", "sum"))
        .assign(ctr=lambda df: df["clicks"] / df["impressions"])
        .sort_values("impressions", ascending=False)
        .reset_index(drop=True)
    )
    stats["rank"] = np.arange(1, len(stats) + 1)
    return stats


def select_top_similar_ctr_pair(
    frame: pd.DataFrame,
    *,
    top_k: int = 20,
    min_impressions: int = 100,
    min_clicks: int = 1,
) -> ItemPairSelection:
    stats = item_ctr_table(frame)
    pool = stats[(stats["impressions"] >= min_impressions) & (stats["clicks"] >= min_clicks)].head(top_k)
    if len(pool) < 2:
        raise ValueError("Not enough items in the selection pool.")

    best: ItemPairSelection | None = None
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            row_a = pool.iloc[i]
            row_b = pool.iloc[j]
            gap = abs(float(row_a["ctr"]) - float(row_b["ctr"]))
            candidate = ItemPairSelection(
                item_a=int(row_a["item_id"]),
                item_b=int(row_b["item_id"]),
                impressions_a=int(row_a["impressions"]),
                impressions_b=int(row_b["impressions"]),
                clicks_a=int(row_a["clicks"]),
                clicks_b=int(row_b["clicks"]),
                ctr_a=float(row_a["ctr"]),
                ctr_b=float(row_b["ctr"]),
                ctr_gap=gap,
                rank_pool=top_k,
            )
            if best is None or gap < best.ctr_gap:
                best = candidate
    if best is None:
        raise ValueError("Failed to select an item pair.")
    return best


def convert_obd_to_pairwise_events(
    frame: pd.DataFrame,
    *,
    item_a: int,
    item_b: int,
) -> pd.DataFrame:
    pair_items = {int(item_a), int(item_b)}
    remap = {int(item_a): 0, int(item_b): 1}
    propensity_column = "propensity_score" if "propensity_score" in frame.columns else "action_prob"
    if propensity_column not in frame.columns:
        raise ValueError("OBD CSV must contain propensity_score (or action_prob).")

    subset = frame[frame["item_id"].isin(pair_items)].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError("No rows found for the selected item pair.")

    rows: list[dict[str, object]] = []
    for idx, row in subset.iterrows():
        original_item = int(row["item_id"])
        arm = remap[original_item]
        rows.append(
            {
                "event_id": int(idx + 1),
                "timestamp": row.get("timestamp", idx + 1),
                "user_id": f"pair_{original_item}_{idx}",
                "context_json": json.dumps([]),
                "candidate_arms_json": json.dumps([0, 1]),
                "chosen_arm": arm,
                "reward": float(row["click"]),
                "propensity": float(row[propensity_column]),
                "original_item_id": original_item,
            }
        )
    return pd.DataFrame(rows)
