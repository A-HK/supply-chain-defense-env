"""Independent holdout verifiers for Agentic Security Lab.

These run *outside* the training reward loop and provide a stronger signal
than the in-episode rubric alone.  A model that games the training reward
without genuinely solving the task should fail at least some of these checks.

Design principles (from hackathon guidelines):
- Use hard programmatic checks, not LLM-as-judge
- Verify actual state transitions, not verbal claims
- Layered: outcome checks first, process checks second
- Keep a separate holdout evaluator distinct from the training reward
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Episode-level verifiers (run against a single completed observation)
# ---------------------------------------------------------------------------

def verify_episode_done(observation: dict[str, Any]) -> bool:
    """Confirm the episode actually terminated (not mid-episode scoring)."""
    return bool(observation.get("done", False))


def verify_no_invalid_action(observation: dict[str, Any]) -> bool:
    """Confirm the last action parsed without a server-side error."""
    return observation.get("error") in (None, "")


def verify_attacker_stopped(observation: dict[str, Any]) -> bool:
    """Critical check: the attacker must not have succeeded (exfiltrated)."""
    metrics = observation.get("data", {}).get("evaluator_metrics", {})
    return not bool(metrics.get("attacker_succeeded", False))


def verify_no_false_positives(observation: dict[str, Any]) -> bool:
    """Confirm the agent didn't quarantine any legitimate packages."""
    metrics = observation.get("data", {}).get("evaluator_metrics", {})
    return int(metrics.get("false_positives", 0)) == 0


def verify_quarantine_completeness(observation: dict[str, Any], threshold: float = 1.0) -> bool:
    """Confirm all required packages were quarantined (default: 100%).

    Pass threshold < 1.0 for partial-credit evaluation.
    """
    breakdown = observation.get("data", {}).get("score_breakdown", {})
    return float(breakdown.get("quarantine_ratio", 0.0)) >= threshold


def verify_secret_rotation(observation: dict[str, Any], threshold: float = 1.0) -> bool:
    """Confirm all (or a fraction of) exposed secrets were rotated."""
    breakdown = observation.get("data", {}).get("score_breakdown", {})
    return float(breakdown.get("rotate_ratio", 0.0)) >= threshold


def verify_notification(observation: dict[str, Any], threshold: float = 0.8) -> bool:
    """Confirm at least `threshold` fraction of affected teams were notified.

    Notification is the hardest action to get right (requires prior
    check_dependents), so we allow 80% by default.
    """
    breakdown = observation.get("data", {}).get("score_breakdown", {})
    return float(breakdown.get("notify_ratio", 0.0)) >= threshold


def verify_benchmark_score(observation: dict[str, Any], threshold: float = 0.7) -> bool:
    """Composite benchmark score must exceed `threshold`."""
    score = float(observation.get("data", {}).get("benchmark_score", 0.0))
    return score >= threshold


def verify_no_spam(observation: dict[str, Any], max_invalid: int = 3) -> bool:
    """Detect degenerate looping: too many invalid / repeated actions is a
    signal that the model is pattern-matching rather than reasoning."""
    metrics = observation.get("data", {}).get("evaluator_metrics", {})
    return int(metrics.get("invalid_actions", 0)) <= max_invalid


# ---------------------------------------------------------------------------
# Trajectory-level verifiers (run against a full episode trajectory)
# ---------------------------------------------------------------------------

def verify_investigation_before_action(trajectory: list[dict[str, Any]]) -> bool:
    """Process check: the agent must investigate (scan/inspect) before quarantine.

    A model that jumps straight to quarantine without any investigation is
    guessing.  Any valid trajectory should have at least one scan_logs or
    inspect_package before the first quarantine.
    """
    seen_investigation = False
    for step in trajectory:
        cmd = step.get("command", "")
        if cmd in ("scan_logs", "inspect_package"):
            seen_investigation = True
        if cmd == "quarantine" and not seen_investigation:
            return False
    return True


def verify_no_premature_conclude(trajectory: list[dict[str, Any]]) -> bool:
    """The agent must not call conclude at step 1 or 2 (trivial skip)."""
    for i, step in enumerate(trajectory[:3]):
        if step.get("command") == "conclude":
            return False
    return True


def verify_critical_secrets_rotated(
    trajectory: list[dict[str, Any]],
    final_observation: dict[str, Any],
) -> bool:
    """Confirm that at least one secret was rotated before conclude is called.

    Prevents the degenerate strategy of quarantining the package but never
    revoking stolen credentials.
    """
    rotated_any = any(step.get("command") == "rotate_secret" for step in trajectory)
    return rotated_any


# ---------------------------------------------------------------------------
# Aggregate holdout check (used by self_improve.py and evaluate.py)
# ---------------------------------------------------------------------------

def holdout_pass(summary: dict[str, Any]) -> bool:
    """Return True if the episode batch meets the minimum quality bar.

    Requires:
    - Mean benchmark score >= 0.6
    - Success rate (score >= 0.8) >= 35 %
    - Mean false-positive rate <= 1.0 (at most 1 FP per episode on average)
    """
    score_ok = float(summary.get("mean_score", 0.0)) >= 0.6
    success_ok = float(summary.get("success_rate", 0.0)) >= 0.35
    fp_ok = float(summary.get("mean_false_positives", 9999.0)) <= 1.0
    return score_ok and success_ok and fp_ok


def run_episode_verifiers(
    final_observation: dict[str, Any],
    trajectory: list[dict[str, Any]] | None = None,
) -> dict[str, bool]:
    """Run all applicable verifiers and return a named result dict.

    Useful for the training dashboard: log this dict alongside reward so
    that reward hacking becomes immediately visible.
    """
    results: dict[str, bool] = {
        "done": verify_episode_done(final_observation),
        "no_invalid_action": verify_no_invalid_action(final_observation),
        "attacker_stopped": verify_attacker_stopped(final_observation),
        "no_false_positives": verify_no_false_positives(final_observation),
        "quarantine_complete": verify_quarantine_completeness(final_observation),
        "secrets_rotated": verify_secret_rotation(final_observation),
        "teams_notified": verify_notification(final_observation),
        "benchmark_score_ok": verify_benchmark_score(final_observation),
        "no_spam": verify_no_spam(final_observation),
    }
    if trajectory:
        results["investigation_before_action"] = verify_investigation_before_action(trajectory)
        results["no_premature_conclude"] = verify_no_premature_conclude(trajectory)
        results["critical_secrets_rotated"] = verify_critical_secrets_rotated(
            trajectory, final_observation
        )
    return results
