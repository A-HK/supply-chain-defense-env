"""Adaptive curriculum scheduling."""

from __future__ import annotations


class AdaptiveCurriculum:
    def __init__(self) -> None:
        self.level = 1.0

    def update(self, success_rate: float) -> float:
        if success_rate > 0.8:
            self.level = min(1.25, self.level + 0.05)
        elif success_rate < 0.3:
            self.level = max(0.85, self.level - 0.05)
        return self.level
