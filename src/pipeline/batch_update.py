from __future__ import annotations

from collections.abc import Iterable

from src.pipeline.schemas import EventRecord


def apply_batch_updates(policy: object, buffered_rows: Iterable[tuple[EventRecord, int, float]]) -> int:
    count = 0
    for event, arm, reward in buffered_rows:
        policy.update(arm, reward, event.context)
        count += 1
    return count
