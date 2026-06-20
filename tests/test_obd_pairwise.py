from __future__ import annotations

import pandas as pd

from src.data.obd_pairwise import item_ctr_table, select_top_similar_ctr_pair


def test_select_pair_from_synthetic_frame() -> None:
    rows = []
    for item_id, clicks in [(1, 1), (1, 0), (2, 1), (2, 1)]:
        rows.append({"item_id": item_id, "click": clicks})
    frame = pd.DataFrame(rows)
    pair = select_top_similar_ctr_pair(frame, top_k=5, min_impressions=1, min_clicks=1)
    assert pair.item_a in {1, 2}
    assert pair.item_b in {1, 2}
    assert pair.item_a != pair.item_b


def test_item_ctr_table_sorted_by_impressions() -> None:
    frame = pd.DataFrame(
        [
            {"item_id": 1, "click": 0},
            {"item_id": 1, "click": 1},
            {"item_id": 2, "click": 0},
        ]
    )
    stats = item_ctr_table(frame)
    assert int(stats.iloc[0]["item_id"]) == 1
    assert int(stats.iloc[0]["impressions"]) == 2
