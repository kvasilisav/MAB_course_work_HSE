from __future__ import annotations

from collections.abc import Iterable, Iterator

from .schemas import EventRecord


def iter_events(events: Iterable[EventRecord]) -> Iterator[EventRecord]:
    for event in sorted(events, key=lambda row: row.event_id):
        yield event
