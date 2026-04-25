"""Anti-reward-hacking tests.

Validates that the environment correctly penalises known exploit patterns:
1. scan_logs spam (low-value repeat)
2. False positive quarantine
3. Immediate conclude for free bonuses
4. Repeated identical actions
5. Invalid command spam
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.agentic_security_lab_environment import AgenticSecurityLabEnvironment  # noqa: E402


def make_action(command: str, **parameters):
    return type("Action", (), {"command": command, "parameters": parameters})()


# ── 1. scan_logs spam should not accumulate unbounded reward ──

def test_scan_logs_spam_is_low_value() -> None:
    """Spamming scan_logs on the same package should give diminishing/low total reward."""
    env = AgenticSecurityLabEnvironment("easy")
    env.reset(mode="training")
    total = 0.0
    for _ in range(10):
        obs = env.step(make_action("scan_logs", package="axios@1.7.4"))
        total += obs.reward
    # 10 * 0.02 = 0.20 — should be small compared to a proper containment (~0.8+)
    assert total <= 0.25, f"scan_logs spam yielded {total:.3f} — too high"


# ── 2. False positive quarantine should be penalised ──

def test_false_positive_quarantine_penalty() -> None:
    """Quarantining a clean package should give negative reward."""
    env = AgenticSecurityLabEnvironment("easy")
    env.reset(mode="benchmark")
    obs = env.step(make_action("quarantine", package="axios@1.7.3"))
    assert obs.reward == -0.05, f"Expected -0.05 for FP quarantine, got {obs.reward}"
    assert env.state.false_positive_count == 1


# ── 3. Immediate conclude should not give free bonuses ──

def test_immediate_conclude_negative_reward() -> None:
    """Concluding without doing anything should give negative reward."""
    env = AgenticSecurityLabEnvironment("easy")
    env.reset(mode="benchmark")
    obs = env.step(make_action("conclude"))
    assert obs.reward < 0, f"Immediate conclude gave reward={obs.reward}, expected negative"
    assert obs.data["benchmark_score"] == 0.0


# ── 4. Re-rotating already rotated secret is penalised ──

def test_re_rotate_penalty() -> None:
    """Rotating an already-rotated secret should be penalised."""
    env = AgenticSecurityLabEnvironment("easy")
    env.reset(mode="training")
    # First: discover via scan, then rotate
    env.step(make_action("scan_logs", package="axios@1.7.4"))
    obs1 = env.step(make_action("rotate_secret", secret="STRIPE_SECRET_KEY"))
    assert obs1.reward > 0  # first rotation is positive
    obs2 = env.step(make_action("rotate_secret", secret="STRIPE_SECRET_KEY"))
    assert obs2.reward == -0.02, f"Re-rotate gave {obs2.reward}, expected -0.02"


# ── 5. Invalid command spam is penalised ──

def test_invalid_command_penalty() -> None:
    """Unknown commands should be penalised."""
    env = AgenticSecurityLabEnvironment("easy")
    env.reset(mode="benchmark")
    obs = env.step(make_action("hack_the_planet"))
    assert obs.reward == -0.01
    assert env.state.invalid_action_count == 1
    # Spam 5 more
    for _ in range(5):
        env.step(make_action("not_a_command"))
    assert env.state.invalid_action_count == 6


# ── 6. Already-quarantined package re-quarantine is penalised ──

def test_re_quarantine_penalty() -> None:
    """Quarantining an already-quarantined package should be penalised."""
    env = AgenticSecurityLabEnvironment("easy")
    env.reset(mode="training")
    env.step(make_action("scan_logs", package="axios@1.7.4"))
    obs1 = env.step(make_action("quarantine", package="axios@1.7.4"))
    assert obs1.reward == 0.15  # first quarantine is positive
    obs2 = env.step(make_action("quarantine", package="axios@1.7.4"))
    assert obs2.reward == -0.02, f"Re-quarantine gave {obs2.reward}, expected -0.02"


# ── 7. Optimal easy episode should score high ──

def test_optimal_easy_episode() -> None:
    """An optimal sequence of actions should yield benchmark_score near 1.0."""
    env = AgenticSecurityLabEnvironment("easy")
    env.reset(mode="benchmark")
    env.step(make_action("scan_logs", package="axios@1.7.4"))
    env.step(make_action("quarantine", package="axios@1.7.4"))
    env.step(make_action("check_dependents", package="axios@1.7.4"))
    env.step(make_action("rotate_secret", secret="STRIPE_SECRET_KEY"))
    env.step(make_action("rotate_secret", secret="INTERNAL_API_TOKEN"))
    env.step(make_action("notify", team="payments-service"))
    env.step(make_action("notify", team="auth-service"))
    env.step(make_action("notify", team="api-gateway"))
    obs = env.step(make_action("conclude"))
    score = obs.data["benchmark_score"]
    assert score >= 0.95, f"Optimal episode scored {score}, expected >= 0.95"
    assert obs.reward > 0, "Optimal conclude should give positive bonus"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
