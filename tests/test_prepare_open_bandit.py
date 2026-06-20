import json

import pandas as pd

from scripts.prepare_open_bandit import convert_obd_frame
from src.pipeline.loader import load_events


def test_convert_obd_frame_produces_valid_events() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2019-11-24 00:00:34",
                "item_id": 2,
                "position": 1,
                "click": 1,
                "propensity_score": 0.25,
                "user_feature_0": "abc",
                "user_feature_1": "def",
                "user_feature_2": "ghi",
                "user_feature_3": "jkl",
                "user-item_affinity_0": 0.1,
                "user-item_affinity_1": 0.2,
                "user-item_affinity_2": 0.3,
            },
            {
                "timestamp": "2019-11-24 00:01:00",
                "item_id": 1,
                "position": 2,
                "click": 0,
                "propensity_score": 0.25,
                "user_feature_0": "abc",
                "user_feature_1": "def",
                "user_feature_2": "ghi",
                "user_feature_3": "jkl",
                "user-item_affinity_0": 0.0,
                "user-item_affinity_1": 0.5,
                "user-item_affinity_2": 0.0,
            },
        ]
    )

    converted = convert_obd_frame(frame, include_context=True, seed=1, max_events=None)
    assert len(converted) == 2
    assert set(converted.columns) >= {
        "event_id",
        "timestamp",
        "user_id",
        "context_json",
        "candidate_arms_json",
        "chosen_arm",
        "reward",
        "propensity",
    }
    assert converted.iloc[0]["chosen_arm"] == 2
    assert converted.iloc[0]["reward"] == 1.0
    assert converted.iloc[0]["propensity"] == 0.25
    assert json.loads(converted.iloc[0]["candidate_arms_json"]) == [0, 1, 2]


def test_convert_obd_frame_roundtrip_through_loader(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": 1,
                "item_id": 0,
                "position": 1,
                "click": 0,
                "propensity_score": 0.5,
                "user_feature_0": "u",
                "user-item_affinity_0": 0.0,
            }
        ]
    )
    converted = convert_obd_frame(frame, include_context=True)
    path = tmp_path / "obd_events.csv"
    converted.to_csv(path, index=False)
    events = load_events(path)
    assert len(events) == 1
    assert events[0].propensity == 0.5
    assert events[0].candidate_arms == [0]
