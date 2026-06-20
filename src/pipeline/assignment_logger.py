from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .schemas import AssignmentRecord


class AssignmentLogger:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, assignments: list[AssignmentRecord]) -> None:
        if not assignments:
            return
        frame = pd.DataFrame([asdict(row) for row in assignments])
        if self.output_path.exists():
            frame.to_csv(self.output_path, mode="a", header=False, index=False)
        else:
            frame.to_csv(self.output_path, index=False)
