"""Transition-conditioned world model for action ranking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WorldModelStats:
    avg_reward_delta: float
    feature_action_stats: dict[str, dict[str, float]]
    transition_count: int


def feature_key(observation: dict[str, Any], action: dict[str, Any]) -> str:
    data = observation.get("data", {})
    score = float(data.get("benchmark_score", 0.0))
    active = len(observation.get("active_malicious_packages", []))
    secrets = len(observation.get("exposed_secrets", []))
    uncertainty = round(float(observation.get("uncertainty_score", 0.0)), 1)
    return (
        f"cmd={action.get('command', 'unknown')}"
        f"|score={round(score, 1)}"
        f"|active={active}"
        f"|secrets={secrets}"
        f"|uncertainty={uncertainty}"
    )


class LightweightWorldModel:
    def __init__(self, stats: WorldModelStats):
        self.stats = stats

    @classmethod
    def fit(cls, transitions: list[dict[str, Any]]) -> "LightweightWorldModel":
        if not transitions:
            return cls(WorldModelStats(avg_reward_delta=0.0, feature_action_stats={}, transition_count=0))

        stats: dict[str, list[float]] = {}
        reward_deltas: list[float] = []
        for item in transitions:
            observation = item.get("observation", {})
            action = item.get("action", {})
            next_observation = item.get("next_observation", {})
            before_score = float(observation.get("data", {}).get("benchmark_score", 0.0))
            after_score = float(next_observation.get("data", {}).get("benchmark_score", before_score))
            reward = float(item.get("reward", 0.0))
            delta = after_score - before_score
            reward_deltas.append(delta)
            key = feature_key(observation, action)
            stats.setdefault(key, []).append(delta + reward)

        feature_action_stats = {
            key: {
                "predicted_value": sum(values) / len(values),
                "sample_count": float(len(values)),
            }
            for key, values in stats.items()
        }
        avg_reward_delta = sum(reward_deltas) / len(reward_deltas)
        return cls(
            WorldModelStats(
                avg_reward_delta=avg_reward_delta,
                feature_action_stats=feature_action_stats,
                transition_count=len(transitions),
            )
        )

    def predict(self, observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        key = feature_key(observation, action)
        stats = self.stats.feature_action_stats.get(key, {})
        predicted_value = float(stats.get("predicted_value", self.stats.avg_reward_delta))
        sample_count = float(stats.get("sample_count", 0.0))
        confidence = min(1.0, sample_count / 5.0)
        return {
            "predicted_value": predicted_value,
            "predicted_benchmark_delta": predicted_value,
            "confidence": confidence,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "avg_reward_delta": self.stats.avg_reward_delta,
                    "feature_action_stats": self.stats.feature_action_stats,
                    "transition_count": self.stats.transition_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LightweightWorldModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            WorldModelStats(
                avg_reward_delta=float(data.get("avg_reward_delta", 0.0)),
                feature_action_stats={
                    key: {
                        "predicted_value": float(value.get("predicted_value", 0.0)),
                        "sample_count": float(value.get("sample_count", 0.0)),
                    }
                    for key, value in data.get("feature_action_stats", {}).items()
                },
                transition_count=int(data.get("transition_count", 0)),
            )
        )
