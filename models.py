"""
Supply Chain Incident Response — Typed Models
Replace the contents of:  agentic_security_lab/models.py
"""
from typing import Any, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Action  (what the agent sends each turn)
# ---------------------------------------------------------------------------

class AgenticSecurityLabAction(BaseModel):
    """
    One discrete action an agent can take during the incident.

    command — one of:
        inspect_package   examine a package's metadata / publish history
        check_dependents  list downstream consumers of a package
        rotate_secret     revoke + regenerate a named credential
        quarantine        yank a package version from the registry
        notify            send a breach notification to a downstream team
        scan_logs         pull CI/CD logs for a package to look for IOCs
        conclude          declare incident contained  (ends the episode)

    parameters — command-specific kwargs, e.g.
        {"package": "axios@1.7.4"}
        {"secret":  "STRIPE_SECRET_KEY"}
        {"team":    "payments-service"}
    """
    command: str
    parameters: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Observation  (what the agent receives after each action)
# ---------------------------------------------------------------------------

class AgenticSecurityLabObservation(BaseModel):
    """What the agent sees after each step."""

    # Required by OpenEnv spec
    success: bool
    done: bool
    reward: float

    # Human-readable result of the last action
    result: str

    # Structured output (package metadata, dep list, log lines …)
    data: dict[str, Any] = {}

    # Running incident dashboard
    incident_summary: str = ""

    # Steps before attacker completes exfiltration
    steps_remaining: int = 0

    # Credentials still exposed
    exposed_secrets: list[str] = []

    # Malicious packages still in the registry
    active_malicious_packages: list[str] = []

    # Set when the action itself was invalid
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# State  (returned by GET /state — full internal episode state)
# ---------------------------------------------------------------------------

class AgenticSecurityLabState(BaseModel):
    """Full episode state."""

    # OpenEnv base fields
    episode_id: str = ""
    step_count: int = 0

    # Task
    task_name: str = ""

    # Package registry snapshot
    packages:   dict[str, Any]       = {}   # name@version -> metadata
    dependents: dict[str, list[str]] = {}   # package      -> consumers
    secrets:    dict[str, Any]       = {}   # secret_name  -> {rotated, critical, owner}

    # Agent progress
    quarantined:    list[str] = []
    rotated_secrets: list[str] = []
    notified_teams: list[str] = []
    inspected:      list[str] = []
    scanned_logs:   list[str] = []

    # Attacker timeline
    max_steps:          int   = 20
    exfiltration_step:  int   = 15   # step at which attacker wins if crits not rotated
    attacker_progress:  float = 0.0

    # Episode outcome
    total_reward:       float = 0.0
    incident_contained: bool  = False
    attacker_succeeded: bool  = False