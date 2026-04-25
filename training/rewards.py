"""Multi-component reward rubric for Agentic Security Lab.

Each function is an independent scoring dimension.  Composing them gives a
richer, harder-to-game signal than a single scalar (as recommended by the
OpenEnv reward-design guide and the hackathon guidelines).

All functions accept a single observation dict (the JSON returned by /step or
/reset) and return a float in [0, 1] unless otherwise noted.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Individual rubric components
# ---------------------------------------------------------------------------

def containment_score(observation: dict[str, Any]) -> float:
    """Fraction of malicious packages correctly quarantined.

    Uses the score_breakdown emitted in observation.data; falls back to
    step reward as a proxy if breakdown is absent (e.g. during SFT warmup).
    """
    breakdown = observation.get("data", {}).get("score_breakdown", {})
    if breakdown:
        return float(breakdown.get("quarantine_ratio", 0.0))
    # Proxy: positive reward on a quarantine step implies progress
    return 1.0 if float(observation.get("reward", 0.0)) > 0.1 else 0.0


def secret_rotation_score(observation: dict[str, Any]) -> float:
    """Fraction of exposed secrets that have been rotated."""
    breakdown = observation.get("data", {}).get("score_breakdown", {})
    if breakdown:
        return float(breakdown.get("rotate_ratio", 0.0))
    return 1.0 if float(observation.get("reward", 0.0)) > 0.05 else 0.0


def notification_score(observation: dict[str, Any]) -> float:
    """Fraction of affected teams notified."""
    breakdown = observation.get("data", {}).get("score_breakdown", {})
    if breakdown:
        return float(breakdown.get("notify_ratio", 0.0))
    return 0.0


def containment_race_score(observation: dict[str, Any]) -> float:
    """1.0 if the attacker was stopped before exfiltration, 0.0 otherwise."""
    breakdown = observation.get("data", {}).get("score_breakdown", {})
    if breakdown:
        return float(breakdown.get("contain_ratio", 0.0))
    metrics = observation.get("data", {}).get("evaluator_metrics", {})
    if metrics.get("attacker_succeeded", False):
        return 0.0
    return 1.0


def precision_score(observation: dict[str, Any]) -> float:
    """Anti-false-positive penalty: penalises quarantining clean packages.

    Returns 1.0 when no false positives have been committed, decaying
    toward 0.0 as the false-positive count grows.
    """
    metrics = observation.get("data", {}).get("evaluator_metrics", {})
    fp = int(metrics.get("false_positives", 0))
    invalid = int(metrics.get("invalid_actions", 0))
    # Each false positive costs 0.15; each invalid action costs 0.05.
    penalty = min(1.0, fp * 0.15 + invalid * 0.05)
    return max(0.0, 1.0 - penalty)


def efficiency_score(observation: dict[str, Any]) -> float:
    """Reward for completing the incident response quickly.

    Computed as steps_remaining / max_steps so that faster containment
    yields a higher score.  Returns 0.0 once the episode is over with
    no remaining budget.
    """
    data = observation.get("data", {})
    max_steps = int(data.get("max_steps", 20))
    steps_remaining = int(observation.get("steps_remaining", 0))
    if max_steps <= 0:
        return 0.0
    return round(max(0.0, steps_remaining / max_steps), 4)


def format_score(observation: dict[str, Any]) -> float:
    """1.0 if the last action parsed without an error, 0.0 otherwise.

    Encourages the model to emit well-formed JSON tool calls rather than
    free-text responses that fall back to 'conclude'.
    """
    error = observation.get("error")
    return 0.0 if error else 1.0


# ---------------------------------------------------------------------------
# Composite breakdown (used by evaluate.py and the GRPO notebook dashboard)
# ---------------------------------------------------------------------------

def component_breakdown(observation: dict[str, Any]) -> dict[str, float]:
    """Return all rubric scores for a single observation.

    Keys are designed to be logged as individual columns so reward hacking
    becomes visible (a rising aggregate score with a collapsing containment
    or precision score is a red flag).
    """
    return {
        "containment": containment_score(observation),
        "secret_rotation": secret_rotation_score(observation),
        "notification": notification_score(observation),
        "containment_race": containment_race_score(observation),
        "precision": precision_score(observation),
        "efficiency": efficiency_score(observation),
        "format": format_score(observation),
    }


def composite_reward(observation: dict[str, Any]) -> float:
    """Weighted aggregate of all rubric components.

    Weights follow the benchmark formula (quarantine 0.35, rotate 0.35,
    notify 0.20, contain 0.10) with small bonuses for precision/efficiency
    to incentivise clean, fast responses.
    """
    c = component_breakdown(observation)
    score = (
        0.35 * c["containment"]
        + 0.35 * c["secret_rotation"]
        + 0.20 * c["notification"]
        + 0.10 * c["containment_race"]
        + 0.05 * c["precision"]        # anti-cheat bonus
        + 0.03 * c["efficiency"]       # speed bonus
        + 0.02 * c["format"]           # format compliance
    )
    return round(max(0.0, min(1.1, score)), 6)  # max slightly above 1.0 for perfect runs
