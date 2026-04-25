"""Transition dataset utilities for world model training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_transition(path: str | Path, transition: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(transition) + "\n")


def load_transitions(path: str | Path) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return transitions
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            transitions.append(json.loads(line))
    return transitions
