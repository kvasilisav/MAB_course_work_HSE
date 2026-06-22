import json
import tempfile
from pathlib import Path

import pandas as pd

from src.bandits.fixed_ab import FixedABPolicy
from src.bandits.thompson_sampling import ThompsonSamplingPolicy
from src.experiments.obd_streaming_ope import StreamingReplayState, run_streaming_obd_ope
from src.ope.replay import compare_policies_replay, run_replay_evaluation
from src.pipeline.loader import load_events


def _load_events_from_frame(frame: pd.DataFrame) -> list:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "events.csv"
        frame.to_csv(path, index=False)
        return load_events(path)


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


def test_replay_bootstrap_snips_ci() -> None:
    events = _load_events_from_frame(
        pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "timestamp": 1,
                    "user_id": "u1",
                    "context_json": json.dumps([1.0]),
                    "candidate_arms_json": json.dumps([0, 1]),
                    "chosen_arm": 0,
                    "reward": 1.0,
                    "propensity": 0.5,
                },
                {
                    "event_id": 2,
                    "timestamp": 2,
                    "user_id": "u2",
                    "context_json": json.dumps([1.0]),
                    "candidate_arms_json": json.dumps([0, 1]),
                    "chosen_arm": 0,
                    "reward": 0.0,
                    "propensity": 0.5,
                },
                {
                    "event_id": 3,
                    "timestamp": 3,
                    "user_id": "u3",
                    "context_json": json.dumps([1.0]),
                    "candidate_arms_json": json.dumps([0, 1]),
                    "chosen_arm": 1,
                    "reward": 1.0,
                    "propensity": 0.5,
                },
            ]
        )
    )
    result = run_replay_evaluation(
        events=events,
        policy_factory=lambda n_arms, seed: FixedABPolicy(
            n_arms=n_arms,
            probabilities=[1.0, 0.0],
            seed=seed,
        ),
        seed=0,
        freeze_policy=True,
        shuffle_events=False,
        n_bootstrap=200,
    )
    assert result.snips_ci_low is not None
    assert result.snips_ci_high is not None
    assert result.snips_ci_low <= result.snips_estimate <= result.snips_ci_high


def test_freeze_policy_skips_updates() -> None:
    events = _load_events_from_frame(
        pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "timestamp": 1,
                    "user_id": "u1",
                    "context_json": json.dumps([1.0]),
                    "candidate_arms_json": json.dumps([0, 1]),
                    "chosen_arm": 0,
                    "reward": 1,
                    "propensity": 0.5,
                },
                {
                    "event_id": 2,
                    "timestamp": 2,
                    "user_id": "u2",
                    "context_json": json.dumps([1.0]),
                    "candidate_arms_json": json.dumps([0, 1]),
                    "chosen_arm": 0,
                    "reward": 1,
                    "propensity": 0.5,
                },
            ]
        )
    )

    policy_frozen = ThompsonSamplingPolicy(n_arms=2, seed=0)
    policy_frozen.reset(seed=0)

    class _FrozenHolder:
        policy = policy_frozen

    run_replay_evaluation(
        events=events,
        policy_factory=lambda n_arms, seed: _FrozenHolder.policy,
        seed=0,
        freeze_policy=True,
        shuffle_events=False,
    )
    assert policy_frozen.snapshot()["alphas"] == [1.0, 1.0]
    assert policy_frozen.snapshot()["betas"] == [1.0, 1.0]

    policy_unfrozen = ThompsonSamplingPolicy(n_arms=2, seed=0)
    policy_unfrozen.reset(seed=0)

    class _UnfrozenHolder:
        policy = policy_unfrozen

    run_replay_evaluation(
        events=events,
        policy_factory=lambda n_arms, seed: _UnfrozenHolder.policy,
        seed=0,
        freeze_policy=False,
        shuffle_events=False,
    )
    assert policy_unfrozen.snapshot()["alphas"][0] > 1.0


def test_ips_denominator_uses_valid_logged_rows() -> None:
    events = _load_events_from_frame(
        pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "timestamp": 1,
                    "user_id": "u1",
                    "context_json": json.dumps([1.0]),
                    "candidate_arms_json": json.dumps([0, 1]),
                    "chosen_arm": 0,
                    "reward": 1.0,
                    "propensity": 0.5,
                },
                {
                    "event_id": 2,
                    "timestamp": 2,
                    "user_id": "u2",
                    "context_json": json.dumps([1.0]),
                    "candidate_arms_json": json.dumps([0, 1]),
                    "chosen_arm": 0,
                    "reward": 1.0,
                    "propensity": 0.5,
                },
                {
                    "event_id": 3,
                    "timestamp": 3,
                    "user_id": "u3",
                    "context_json": json.dumps([1.0]),
                    "candidate_arms_json": json.dumps([0, 1]),
                    "chosen_arm": 0,
                    "reward": 1.0,
                    "propensity": 0.0,
                },
            ]
        )
    )

    result = run_replay_evaluation(
        events=events,
        policy_factory=lambda n_arms, seed: FixedABPolicy(
            n_arms=n_arms,
            probabilities=[1.0, 0.0],
            seed=seed,
        ),
        seed=0,
        freeze_policy=True,
        shuffle_events=False,
    )
    assert result.ips_estimate == 2.0


def test_propensity_floor_clips_weights() -> None:
    events = _load_events_from_frame(
        pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "timestamp": 1,
                    "user_id": "u1",
                    "context_json": json.dumps([1.0]),
                    "candidate_arms_json": json.dumps([0, 1]),
                    "chosen_arm": 0,
                    "reward": 1.0,
                    "propensity": 0.001,
                }
            ]
        )
    )

    clipped = run_replay_evaluation(
        events=events,
        policy_factory=lambda n_arms, seed: FixedABPolicy(
            n_arms=n_arms,
            probabilities=[1.0, 0.0],
            seed=seed,
        ),
        seed=0,
        freeze_policy=True,
        propensity_floor=0.01,
        shuffle_events=False,
    )
    unclipped = run_replay_evaluation(
        events=events,
        policy_factory=lambda n_arms, seed: FixedABPolicy(
            n_arms=n_arms,
            probabilities=[1.0, 0.0],
            seed=seed,
        ),
        seed=0,
        freeze_policy=True,
        propensity_floor=1e-6,
        shuffle_events=False,
    )
    assert clipped.ips_estimate == 100.0
    assert unclipped.ips_estimate == 1000.0


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


def test_streaming_ips_denominator_uses_valid_logged_rows() -> None:
    state = StreamingReplayState(
        policy_name="fixed_ab",
        seed=0,
        policy=FixedABPolicy(n_arms=2, probabilities=[1.0, 0.0], seed=0),
        freeze_policy=True,
        propensity_floor=0.01,
    )
    state.observe(logged_arm=0, reward=1.0, propensity=0.5, context=None)
    state.observe(logged_arm=0, reward=1.0, propensity=0.5, context=None)
    state.observe(logged_arm=0, reward=1.0, propensity=0.0, context=None)
    row = state.to_row()
    assert row["valid_logged_rows"] == 2
    assert row["ips_estimate"] == 2.0


def test_streaming_freeze_policy_skips_updates() -> None:
    policy = ThompsonSamplingPolicy(n_arms=2, seed=0)
    policy.reset(seed=0)
    state = StreamingReplayState(
        policy_name="thompson_sampling",
        seed=0,
        policy=policy,
        freeze_policy=True,
        propensity_floor=0.01,
    )
    for _ in range(5):
        state.observe(logged_arm=0, reward=1.0, propensity=0.5, context=None)
    assert policy.snapshot()["alphas"] == [1.0, 1.0]

    policy2 = ThompsonSamplingPolicy(n_arms=2, seed=0)
    policy2.reset(seed=0)
    state2 = StreamingReplayState(
        policy_name="thompson_sampling",
        seed=0,
        policy=policy2,
        freeze_policy=False,
        propensity_floor=0.01,
    )
    for _ in range(5):
        state2.observe(logged_arm=0, reward=1.0, propensity=0.5, context=None)
    assert policy2.snapshot()["alphas"][0] > 1.0
