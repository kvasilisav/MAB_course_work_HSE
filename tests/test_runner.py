import json

import pandas as pd

from src.bandits.epsilon_greedy import EpsilonGreedyPolicy
from src.environments.bernoulli import BernoulliBanditEnv
from src.environments.logged_clicks import LoggedClicksBanditEnv
from src.experiments.configs import ExperimentConfig
from src.experiments.runner import run_experiment
from src.pipeline.loader import load_events


def test_runner_synthetic_mode_produces_metrics() -> None:
    config = ExperimentConfig(
        run_id="synthetic_test",
        seed=7,
        horizon=100,
        mode="synthetic",
        policy_factory=lambda n_arms, seed: EpsilonGreedyPolicy(n_arms=n_arms, epsilon=0.1, seed=seed),
        environment_factory=lambda seed: BernoulliBanditEnv([0.1, 0.2, 0.3], horizon=100, seed=seed),
    )
    frame = run_experiment(config)
    assert not frame.empty
    assert {"cumulative_reward", "cumulative_regret", "batch_id"}.issubset(frame.columns)


def test_runner_batch_mode_logs_assignments(tmp_path) -> None:
    events_path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {
                "event_id": 1,
                "timestamp": 1,
                "user_id": "u1",
                "context_json": json.dumps([1.0, 0.0]),
                "candidate_arms_json": json.dumps([0, 1]),
                "chosen_arm": 1,
                "reward": 1.0,
            },
            {
                "event_id": 2,
                "timestamp": 2,
                "user_id": "u2",
                "context_json": json.dumps([0.0, 1.0]),
                "candidate_arms_json": json.dumps([0, 1]),
                "chosen_arm": 0,
                "reward": 0.0,
            },
        ]
    ).to_csv(events_path, index=False)

    assignments_path = tmp_path / "assignments.csv"
    config = ExperimentConfig(
        run_id="batch_test",
        seed=3,
        horizon=2,
        mode="batch",
        batch_size=1,
        assignments_path=assignments_path,
        policy_factory=lambda n_arms, seed: EpsilonGreedyPolicy(n_arms=n_arms, epsilon=0.1, seed=seed),
        environment_factory=lambda seed: LoggedClicksBanditEnv(events=load_events(events_path), seed=seed),
    )
    frame = run_experiment(config)
    assert len(frame) == 2
    assert assignments_path.exists()
