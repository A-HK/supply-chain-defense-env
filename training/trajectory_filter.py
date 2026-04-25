"""StarPO-S trajectory filtering for GRPO training stability.

Implements uncertainty-based filtering from RAGEN (arXiv 2504.20073):
- Compute per-prompt reward standard deviation across G rollouts
- Keep only top keep_ratio of prompts by reward variance
- Prevents echo trap where model collapses into repetitive behavior
- Includes collapse detection via reward-std monitoring
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryFilter:
    """StarPO-S style trajectory filtering for GRPO.

    Keeps top `keep_ratio` of prompts ranked by reward standard deviation.
    Prompts where all rollouts succeed (std=0) or all fail (std=0) are
    uninformative for learning.
    """

    def __init__(self, keep_ratio: float = 0.5, min_std_threshold: float = 0.01,
                 collapse_window: int = 20, collapse_std_threshold: float = 0.02):
        self.keep_ratio = keep_ratio
        self.min_std_threshold = min_std_threshold
        self.collapse_window = collapse_window
        self.collapse_std_threshold = collapse_std_threshold
        self._reward_stds: list[float] = []
        self._filter_ratios: list[float] = []
        self._collapse_detected = False

    def filter_batch(self, prompt_ids: list[Any], rewards: list[float],
                     num_generations: int) -> list[int]:
        """Filter batch, keeping only most informative rollouts."""
        prompt_groups: dict[Any, list[tuple[int, float]]] = defaultdict(list)
        for idx, (pid, reward) in enumerate(zip(prompt_ids, rewards)):
            prompt_groups[pid].append((idx, reward))

        prompt_stds: list[tuple[Any, float, list[int]]] = []
        for pid, group in prompt_groups.items():
            indices = [g[0] for g in group]
            rews = [g[1] for g in group]
            std = float(np.std(rews)) if len(rews) > 1 else 0.0
            prompt_stds.append((pid, std, indices))

        prompt_stds.sort(key=lambda x: -x[1])
        n_keep = max(1, int(len(prompt_stds) * self.keep_ratio))
        keep_indices = []
        for _, std, indices in prompt_stds[:n_keep]:
            keep_indices.extend(indices)

        all_stds = [s for _, s, _ in prompt_stds]
        self._reward_stds.append(float(np.mean(all_stds)) if all_stds else 0.0)
        self._filter_ratios.append(len(keep_indices) / max(1, len(rewards)))
        return sorted(keep_indices)

    def check_collapse(self) -> dict[str, Any]:
        """Check for training collapse indicators."""
        result = {"collapse_detected": False, "reward_std_trend": "stable", "recommendation": "continue"}
        if len(self._reward_stds) < self.collapse_window:
            return result
        recent = self._reward_stds[-self.collapse_window:]
        mean_std = float(np.mean(recent))
        if mean_std < self.collapse_std_threshold:
            result["collapse_detected"] = True
            result["reward_std_trend"] = "collapsing"
            result["recommendation"] = "Reward std near zero. Increase temperature or reduce keep_ratio."
            self._collapse_detected = True
        result["mean_reward_std"] = mean_std
        return result

    def get_metrics(self) -> dict[str, float]:
        return {
            "starpo_reward_std": self._reward_stds[-1] if self._reward_stds else 0.0,
            "starpo_reward_std_ma": float(np.mean(self._reward_stds[-10:])) if self._reward_stds else 0.0,
            "starpo_filter_ratio": self._filter_ratios[-1] if self._filter_ratios else 1.0,
            "starpo_collapse_detected": float(self._collapse_detected),
        }


class CollapseDetector:
    """Lightweight collapse detector for use in TrainerCallback."""

    def __init__(self, window: int = 20):
        self.window = window
        self._rewards: list[float] = []
        self.alerts: list[str] = []

    def update(self, reward: float | None = None) -> str | None:
        if reward is not None:
            self._rewards.append(reward)
        if len(self._rewards) < self.window:
            return None
        recent = self._rewards[-self.window:]
        reward_std = float(np.std(recent))
        if reward_std < 0.001:
            alert = f"COLLAPSE: reward_std={reward_std:.6f} over {self.window} steps"
            self.alerts.append(alert)
            return alert
        return None
