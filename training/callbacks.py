"""Simple JSON metrics logger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricsCallback:
    def __init__(self, output_path: str = "artifacts/metrics.jsonl") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
