from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import EventRecord


def _parse_optional_json_array(raw: str | float | int | None) -> list[int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, (int, np.integer)):
        return [int(raw)]
    text = str(raw).strip()
    if not text:
        return []
    value = json.loads(text)
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(value)]


def _parse_optional_context(raw: str | float | int | None) -> np.ndarray | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    value = json.loads(text)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    return np.asarray([value], dtype=float)


def load_events(path: str | Path) -> list[EventRecord]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"events file was not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"event_id", "timestamp", "user_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"events file misses columns: {sorted(missing)}")

    rows: list[EventRecord] = []
    for row in frame.itertuples(index=False):
        chosen_arm_raw = getattr(row, "chosen_arm", None)
        chosen_arm = int(chosen_arm_raw) if chosen_arm_raw is not None and not pd.isna(chosen_arm_raw) else None
        reward_raw = getattr(row, "reward", None)
        observed_reward = (
            float(reward_raw)
            if reward_raw is not None and not pd.isna(reward_raw)
            else None
        )
        propensity_raw = getattr(row, "propensity", None)
        propensity = (
            float(propensity_raw)
            if propensity_raw is not None and not pd.isna(propensity_raw)
            else None
        )

        candidate_arms = _parse_optional_json_array(getattr(row, "candidate_arms_json", None))
        if not candidate_arms:
            if chosen_arm is not None:
                candidate_arms = [chosen_arm]
            else:
                continue

        rows.append(
            EventRecord(
                event_id=int(getattr(row, "event_id")),
                timestamp=getattr(row, "timestamp"),
                user_id=getattr(row, "user_id"),
                candidate_arms=candidate_arms,
                chosen_arm=chosen_arm,
                observed_reward=observed_reward,
                propensity=propensity,
                context=_parse_optional_context(getattr(row, "context_json", None)),
            )
        )
    return rows
