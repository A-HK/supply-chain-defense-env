from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.agentic_security_lab_environment import AgenticSecurityLabEnvironment  # noqa: E402


def make_action(command: str, **parameters):
    return type("Action", (), {"command": command, "parameters": parameters})()


def test_reset_hides_ground_truth() -> None:
    env = AgenticSecurityLabEnvironment("easy")
    observation = env.reset()
    assert observation.active_malicious_packages == []
    assert observation.exposed_secrets == []


def test_invalid_mode_falls_back_to_benchmark() -> None:
    env = AgenticSecurityLabEnvironment("easy")
    env.reset(mode="unsupported", command_fallback_enabled=True)
    assert env.state.mode == "benchmark"
    assert env.state.mode_fallback_used is True


def test_command_alias_counts_fallback_usage() -> None:
    env = AgenticSecurityLabEnvironment("easy")
    env.reset(command_fallback_enabled=True)
    observation = env.step(make_action("inspect", package="axios@1.7.4"))
    assert env.state.command_fallback_used_count == 1
    assert "axios@1.7.4" in observation.active_malicious_packages


def test_immediate_conclude_has_zero_benchmark_score() -> None:
    env = AgenticSecurityLabEnvironment("easy")
    env.reset()
    observation = env.step(make_action("conclude"))
    assert observation.reward < 0
    assert observation.data["benchmark_score"] == 0.0


def test_grader_matches_expected_easy_score() -> None:
    env = AgenticSecurityLabEnvironment("easy")
    env.reset()
    env.step(make_action("scan_logs", package="axios@1.7.4"))
    env.step(make_action("quarantine", package="axios@1.7.4"))
    env.step(make_action("rotate_secret", secret="STRIPE_SECRET_KEY"))
    observation = env.step(make_action("conclude"))
    assert observation.data["score_breakdown"]["quarantine_ratio"] == 1.0
    assert observation.data["score_breakdown"]["rotate_ratio"] == 0.5
    assert observation.data["score_breakdown"]["notify_ratio"] == 0.0
    assert observation.data["benchmark_score"] == 0.625
