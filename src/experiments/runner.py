from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.evaluation.metrics import cumulative_regret, cumulative_reward
from src.evaluation.summary import summarize_runs
from src.experiments.configs import ExperimentConfig
from src.pipeline.assignment_logger import AssignmentLogger
from src.pipeline.batch_update import apply_batch_updates
from src.pipeline.schemas import AssignmentRecord


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    env = config.environment_factory(config.seed)
    policy_seed = int(config.seed + 1_000_003)
    policy = config.policy_factory(env.n_arms, policy_seed)
    env.reset(seed=config.seed)
    policy.reset(seed=policy_seed)

    logs: list[dict[str, float | int | str]] = []
    assignments: list[AssignmentRecord] = []
    buffer: list[tuple[object, int, float]] = []
    batch_size = max(1, int(config.batch_size))
    batch_id = 0

    logger = AssignmentLogger(config.assignments_path) if config.assignments_path else None

    for step in range(config.horizon):
        context = env.current_context()
        arm = policy.select_arm(context)
        reward, meta = env.step(arm, context)
        if config.mode == "synthetic":
            policy.update(arm, reward, context)
        else:
            event_ref = getattr(env, "events", [None])[step]
            buffer.append((event_ref, arm, reward))
            if len(buffer) >= batch_size:
                apply_batch_updates(policy, buffer)
                buffer.clear()
                batch_id += 1

        record_batch = batch_id if config.mode == "batch" else 0
        assignments.append(
            AssignmentRecord(
                run_id=config.run_id,
                event_id=int(meta.get("event_id", step)),
                policy_name=policy.name,
                chosen_arm=int(arm),
                reward=float(reward),
                oracle_best_arm=int(meta["best_arm"]),
                oracle_best_reward=float(meta["best_reward"]),
                batch_id=record_batch,
            )
        )
        logs.append(
            {
                "run_id": config.run_id,
                "seed": config.seed,
                "step": step,
                "policy_name": policy.name,
                "chosen_arm": int(arm),
                "reward": float(reward),
                "oracle_best_arm": int(meta["best_arm"]),
                "oracle_best_reward": float(meta["best_reward"]),
                "batch_id": record_batch,
            }
        )

    if config.mode == "batch" and buffer:
        apply_batch_updates(policy, buffer)
        buffer.clear()

    if logger is not None:
        logger.write(assignments)

    frame = pd.DataFrame(logs)
    if frame.empty:
        return frame
    frame["cumulative_reward"] = cumulative_reward(frame["reward"].to_numpy())
    frame["cumulative_regret"] = cumulative_regret(
        frame["reward"].to_numpy(),
        frame["oracle_best_reward"].to_numpy(),
    )
    return frame


def compare_policies(configs: Iterable[ExperimentConfig]) -> pd.DataFrame:
    frames = [run_experiment(cfg) for cfg in configs]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_policy_runs(results: pd.DataFrame) -> pd.DataFrame:
    return summarize_runs(results)
