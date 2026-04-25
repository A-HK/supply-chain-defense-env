"""Reward and benchmark-score evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path


def summarize_runs(path: str = "artifacts/metrics.jsonl") -> dict:
    file_path = Path(path)
    rewards: list[float] = []
    scores: list[float] = []
    success_count = 0
    total = 0
    if not file_path.exists():
        return {"mean_reward": 0.0, "mean_score": 0.0, "success_rate": 0.0, "episodes": 0}
    for line in file_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("type") != "episode_end":
            continue
        total += 1
        reward = float(item.get("total_reward", 0.0))
        score = float(item.get("benchmark_score", 0.0))
        rewards.append(reward)
        scores.append(score)
        if item.get("success", False):
            success_count += 1
    mean_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
    mean_score = (sum(scores) / len(scores)) if scores else 0.0
    success_rate = (success_count / total) if total else 0.0
    return {
        "mean_reward": mean_reward,
        "mean_score": mean_score,
        "success_rate": success_rate,
        "episodes": total,
    }


if __name__ == "__main__":
    print(json.dumps(summarize_runs(), indent=2))
