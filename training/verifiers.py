"""Hard verifiers and holdout checks."""

from __future__ import annotations


def verify_episode_done(observation: dict) -> bool:
    return bool(observation.get("done", False))


def verify_no_invalid_action(observation: dict) -> bool:
    return observation.get("error") in (None, "")


def holdout_pass(summary: dict) -> bool:
    return float(summary.get("mean_score", 0.0)) >= 0.6 and float(summary.get("success_rate", 0.0)) >= 0.35
