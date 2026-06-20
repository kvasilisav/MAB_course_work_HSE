import json

import pandas as pd

from src.bandits.fixed_ab import FixedABPolicy
from src.experiments.obd_streaming_ope import run_streaming_obd_ope
from src.ope.replay import compare_policies_replay
from src.pipeline.loader import load_events


def test_replay_outputs_core_metrics(tmp_path) -> None:
    path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {
                "event_id": 1,
                "timestamp": 1,
                "user_id": "u1",
                "context_json": json.dumps([1.0, 0.0]),
                "candidate_arms_json": json.dumps([0, 1]),
                "chosen_arm": 0,
                "reward": 1,
                "propensity": 0.5,
            },
            {
                "event_id": 2,
                "timestamp": 2,
                "user_id": "u2",
                "context_json": json.dumps([0.0, 1.0]),
                "candidate_arms_json": json.dumps([0, 1]),
                "chosen_arm": 1,
                "reward": 0,
                "propensity": 0.5,
            },
        ]
    ).to_csv(path, index=False)

    events = load_events(path)
    frame = compare_policies_replay(
        events=events,
        policy_factories={"fixed_ab": lambda n_arms, seed: FixedABPolicy(n_arms=n_arms, seed=seed)},
        seeds=2,
    )
    assert not frame.empty
    assert {"ips_estimate", "snips_estimate", "acceptance_rate", "effective_sample_size"}.issubset(frame.columns)


def test_streaming_obd_ope_reads_obd_csv_chunks(tmp_path) -> None:
    path = tmp_path / "obd.csv"
    pd.DataFrame(
        [
            {
                "timestamp": 1,
                "item_id": 0,
                "position": 1,
                "click": 1,
                "propensity_score": 0.5,
            },
            {
                "timestamp": 2,
                "item_id": 1,
                "position": 2,
                "click": 0,
                "propensity_score": 0.5,
            },
        ]
    ).to_csv(path, index=False)

    frame = run_streaming_obd_ope(
        input_path=path,
        behavior_policy="random",
        campaign="all",
        n_arms=2,
        seeds=2,
        chunksize=1,
        max_events=None,
        include_context=False,
        include_linucb=False,
    )

    assert not frame.empty
    assert {"ips_estimate", "snips_estimate", "effective_sample_size"}.issubset(frame.columns)
    assert set(frame["policy_name"]) >= {"fixed_ab", "epsilon_greedy", "ucb1", "thompson_sampling"}
