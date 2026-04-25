"""Reward components used for training/eval dashboards."""

from __future__ import annotations


def component_breakdown(observation: dict) -> dict[str, float]:
    reward = float(observation.get("reward", 0.0))
    error = observation.get("error")
    components = {
        "task_success": max(0.0, reward),
        "correctness": 1.0 if not error else 0.0,
        "safety": 0.0 if error else 1.0,
        "efficiency": max(0.0, 1.0 - float(observation.get("steps_remaining", 0)) / 40.0),
    }
    return components
