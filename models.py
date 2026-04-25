"""Typed models for Agentic Security Lab round 2."""
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action  (what the agent sends each turn)
# ---------------------------------------------------------------------------

class AgenticSecurityLabAction(BaseModel):
    """One action emitted by the policy."""

    command: str
    parameters: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation  (what the agent receives after each action)
# ---------------------------------------------------------------------------

class AgenticSecurityLabObservation(BaseModel):
    """Observation after each action."""

    success: bool
    done: bool
    reward: float
    result: str
    data: dict[str, Any] = Field(default_factory=dict)
    incident_summary: str = ""
    steps_remaining: int = 0
    exposed_secrets: list[str] = Field(default_factory=list)
    active_malicious_packages: list[str] = Field(default_factory=list)
    visible_alerts: list[str] = Field(default_factory=list)
    uncertainty_score: float = 0.0
    plan_progress: dict[str, bool] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# State  (returned by GET /state — full internal episode state)
# ---------------------------------------------------------------------------

class AgenticSecurityLabState(BaseModel):
    """Full internal episode state."""

    episode_id: str = ""
    step_count: int = 0
    task_name: str = ""
    mode: str = "benchmark"
    mode_fallback_used: bool = False
    command_fallback_enabled: bool = False
    command_fallback_used_count: int = 0
    invalid_action_count: int = 0
    false_positive_count: int = 0
    packages: dict[str, Any] = Field(default_factory=dict)
    dependents: dict[str, list[str]] = Field(default_factory=dict)
    secrets: dict[str, Any] = Field(default_factory=dict)
    quarantined: list[str] = Field(default_factory=list)
    rotated_secrets: list[str] = Field(default_factory=list)
    notified_teams: list[str] = Field(default_factory=list)
    inspected: list[str] = Field(default_factory=list)
    scanned_logs: list[str] = Field(default_factory=list)
    discovered_packages: list[str] = Field(default_factory=list)
    discovered_secrets: list[str] = Field(default_factory=list)
    traced_packages: list[str] = Field(default_factory=list)
    pending_hidden_iocs: list[str] = Field(default_factory=list)
    discovered_iocs: list[str] = Field(default_factory=list)
    risk_events: list[str] = Field(default_factory=list)
    max_steps: int = 20
    exfiltration_step: int = 15
    attacker_progress:  float = 0.0
    total_reward: float = 0.0
    incident_contained: bool = False
    attacker_succeeded: bool = False
    plan_progress: dict[str, bool] = Field(default_factory=dict)
    trajectory_log: list[dict[str, Any]] = Field(default_factory=list)
