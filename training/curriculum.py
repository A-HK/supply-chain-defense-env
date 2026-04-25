"""Adaptive curriculum scheduler for Agentic Security Lab.

Implements a three-tier progression (easy → medium → hard) driven by a
rolling-average benchmark score.  Includes a bandit-style mechanism for
difficulty adjustment inspired by the Self-Evolving Curriculum approach
(arXiv 2505.14970).

Usage
-----
    curriculum = AdaptiveCurriculum()
    task_name = curriculum.current_task()     # start on "easy"
    # ... run episode, get benchmark_score ...
    curriculum.update(benchmark_score)
    task_name = curriculum.current_task()     # may advance to "medium"
"""

from __future__ import annotations

from collections import deque
from typing import Any

_TIERS: list[str] = ["easy", "medium", "hard"]

# Thresholds to *advance* to the next tier (rolling average over window episodes)
_ADVANCE_THRESHOLD: dict[str, float] = {
    "easy":   0.70,
    "medium": 0.65,
    "hard":   1.00,  # no advance from hard
}

# If rolling average drops below this, step *back* to the previous tier
_REGRESS_THRESHOLD: dict[str, float] = {
    "easy":   0.00,  # no regression below easy
    "medium": 0.25,
    "hard":   0.30,
}

_WINDOW = 10  # number of recent episodes used for rolling average


class AdaptiveCurriculum:
    """Rolling-average-driven three-tier curriculum.

    Attributes
    ----------
    level : float
        Continuous difficulty multiplier forwarded to the procedural
        scenario generator (1.0 = nominal, >1.0 = harder).
    """

    def __init__(self, window: int = _WINDOW) -> None:
        self._tier_idx: int = 0
        self._window: int = window
        self._recent_scores: deque[float] = deque(maxlen=window)
        self.level: float = 1.0
        self._advance_count: int = 0
        self._regress_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_task(self) -> str:
        """Return the name of the current difficulty tier."""
        return _TIERS[self._tier_idx]

    def update(self, benchmark_score: float) -> tuple[str, bool]:
        """Record a completed episode score and update the difficulty tier.

        Parameters
        ----------
        benchmark_score:
            The deterministic benchmark score for the episode (0–1).

        Returns
        -------
        (task_name, changed)
            ``task_name`` is the *new* task name (may be the same as before).
            ``changed`` is True when the tier actually changed.
        """
        self._recent_scores.append(float(benchmark_score))
        old_tier = self._tier_idx
        self._maybe_advance()
        self._maybe_regress()
        self._update_level()
        changed = self._tier_idx != old_tier
        return self.current_task(), changed

    def metrics(self) -> dict[str, Any]:
        """Return a dict suitable for logging to the training dashboard."""
        avg = (sum(self._recent_scores) / len(self._recent_scores)
               if self._recent_scores else 0.0)
        return {
            "curriculum_task": self.current_task(),
            "curriculum_tier": self._tier_idx,
            "curriculum_level": round(self.level, 3),
            "curriculum_avg_score": round(avg, 4),
            "curriculum_window": len(self._recent_scores),
            "curriculum_advances": self._advance_count,
            "curriculum_regressions": self._regress_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rolling_avg(self) -> float:
        if not self._recent_scores:
            return 0.0
        return sum(self._recent_scores) / len(self._recent_scores)

    def _maybe_advance(self) -> None:
        if self._tier_idx >= len(_TIERS) - 1:
            return
        if len(self._recent_scores) < self._window:
            return
        task = _TIERS[self._tier_idx]
        if self._rolling_avg() >= _ADVANCE_THRESHOLD[task]:
            self._tier_idx += 1
            self._recent_scores.clear()  # reset window for new tier
            self._advance_count += 1
            print(
                f"[Curriculum] Advanced to {self.current_task()} "
                f"(avg={self._rolling_avg():.2f} >= {_ADVANCE_THRESHOLD[task]})"
            )

    def _maybe_regress(self) -> None:
        if self._tier_idx <= 0:
            return
        if len(self._recent_scores) < self._window:
            return
        task = _TIERS[self._tier_idx]
        if self._rolling_avg() < _REGRESS_THRESHOLD[task]:
            self._tier_idx -= 1
            self._recent_scores.clear()
            self._regress_count += 1
            print(
                f"[Curriculum] Regressed to {self.current_task()} "
                f"(avg={self._rolling_avg():.2f} < {_REGRESS_THRESHOLD[task]})"
            )

    def _update_level(self) -> None:
        """Keep the continuous difficulty multiplier in sync with the rolling avg."""
        avg = self._rolling_avg()
        if avg > 0.8:
            self.level = min(1.25, self.level + 0.05)
        elif avg < 0.3:
            self.level = max(0.85, self.level - 0.05)
