import json

import pandas as pd

from src.bandits.epsilon_greedy import EpsilonGreedyPolicy
from src.pipeline.assignment_logger import AssignmentLogger
from src.pipeline.batch_update import apply_batch_updates
from src.pipeline.loader import load_events
from src.pipeline.schemas import AssignmentRecord, EventRecord


def test_loader_reads_minimal_events_csv(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "event_id": 1,
                "timestamp": "2026-01-01T00:00:00",
                "user_id": "u1",
                "context_json": json.dumps([0.1, 0.2]),
                "candidate_arms_json": json.dumps([0, 1, 2]),
                "chosen_arm": 1,
                "reward": 1.0,
            }
        ]
    )
    path = tmp_path / "events.csv"
    frame.to_csv(path, index=False)
    events = load_events(path)
    assert len(events) == 1
    assert events[0].candidate_arms == [0, 1, 2]


def test_assignment_logger_writes_required_columns(tmp_path) -> None:
    path = tmp_path / "assignments.csv"
    logger = AssignmentLogger(path)
    logger.write(
        [
            AssignmentRecord(
                run_id="run_1",
                event_id=1,
                policy_name="epsilon_greedy",
                chosen_arm=0,
                reward=1.0,
                oracle_best_arm=0,
                oracle_best_reward=1.0,
                batch_id=0,
            )
        ]
    )
    frame = pd.read_csv(path)
    assert {"run_id", "event_id", "policy_name", "chosen_arm", "reward", "batch_id"}.issubset(frame.columns)


def test_batch_update_applies_each_buffered_event() -> None:
    policy = EpsilonGreedyPolicy(n_arms=2, epsilon=0.0, seed=1)
    events = [
        EventRecord(
            event_id=1,
            timestamp=1,
            user_id=1,
            candidate_arms=[0, 1],
            chosen_arm=0,
            observed_reward=1.0,
            context=None,
        ),
        EventRecord(
            event_id=2,
            timestamp=2,
            user_id=1,
            candidate_arms=[0, 1],
            chosen_arm=1,
            observed_reward=0.0,
            context=None,
        ),
    ]
    count = apply_batch_updates(policy, [(events[0], 0, 1.0), (events[1], 1, 0.0)])
    assert count == 2
