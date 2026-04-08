"""
Supply Chain Incident Response — Typed Models
Replace the contents of:  agentic_security_lab/models.py
"""
from typing import Any, Optional
from pydantic import BaseModel


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


class AgenticSecurityLabObservation(BaseModel):
    """What the agent sees after each step."""

    success: bool
    done: bool
    reward: float

    result: str

    data: dict[str, Any] = {}

    incident_summary: str = ""

    steps_remaining: int = 0

    exposed_secrets: list[str] = []

    active_malicious_packages: list[str] = []

    error: Optional[str] = None


class AgenticSecurityLabState(BaseModel):
    """Full episode state."""

    episode_id: str = ""
    step_count: int = 0

    task_name: str = ""

    packages:   dict[str, Any]       = {}   # name@version -> metadata
    dependents: dict[str, list[str]] = {}   # package      -> consumers
    secrets:    dict[str, Any]       = {}   # secret_name  -> {rotated, critical, owner}

    quarantined:    list[str] = []
    rotated_secrets: list[str] = []
    notified_teams: list[str] = []
    inspected:      list[str] = []
    scanned_logs:   list[str] = []

    max_steps:          int   = 20
    exfiltration_step:  int   = 15   # step at which attacker wins if crits not rotated
    attacker_progress:  float = 0.0

    total_reward:       float = 0.0
    incident_contained: bool  = False
    attacker_succeeded: bool  = False