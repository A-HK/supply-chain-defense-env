"""Deterministic task graders for Agentic Security Lab."""

from typing import Any

try:
    from .models import AgenticSecurityLabState
except ImportError:
    from models import AgenticSecurityLabState

try:
    from server.agentic_security_lab_environment import _score_from_breakdown
except ImportError:
    # Fallback in case server package is not on the path.
    def _score_from_breakdown(breakdown: dict) -> float:  # type: ignore[misc]
        from math import inf  # noqa: F401
        return round(
            max(0.0, min(1.0,
                0.35 * breakdown["quarantine_ratio"]
                + 0.35 * breakdown["rotate_ratio"]
                + 0.20 * breakdown["notify_ratio"]
                + 0.10 * breakdown["contain_ratio"],
            )), 6,
        )

_REQUIRED: dict[str, dict[str, list[str]]] = {
    "easy": {
        "quarantine": ["axios@1.7.4"],
        "rotate_secret": ["STRIPE_SECRET_KEY", "INTERNAL_API_TOKEN"],
        "notify": ["payments-service", "auth-service", "api-gateway"],
    },
    "medium": {
        "quarantine": ["form-data@4.0.1"],
        "rotate_secret": [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "SENDGRID_API_KEY",
            "WEBHOOK_SIGNING_SECRET",
            "POSTGRES_PASSWORD",
        ],
        "notify": [
            "checkout-service",
            "inventory-api",
            "notification-service",
            "reporting-service",
            "search-service",
            "legacy-importer",
            "data-pipeline",
            "webhook-processor",
            "email-sender",
            "file-uploader",
            "crm-sync",
            "analytics-job",
        ],
    },
    "hard": {
        "quarantine": [
            "1odash@4.17.21",
            "expresss@4.18.2",
            "axios-http@1.7.4",
            "node-fetch-lite@3.3.2",
            "dotenvv@16.0.3",
        ],
        "rotate_secret": [
            "NPM_TOKEN",
            "GITHUB_ACTIONS_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "DATABASE_URL",
            "JWT_SECRET",
            "REDIS_PASSWORD",
            "INTERNAL_WEBHOOK_KEY",
        ],
        "notify": [
            "user-service",
            "cart-service",
            "admin-panel",
            "reporting-v2",
            "api-v2",
            "graphql-gateway",
            "rest-proxy",
            "auth-v2",
            "session-manager",
            "mobile-bff",
            "web-bff",
            "partner-api",
            "scraper-job",
            "feed-aggregator",
            "rss-service",
            "integration-tests",
            "deploy-scripts",
            "migration-runner",
            "seed-service",
            "config-loader",
        ],
    },
}

_EPS = 1e-6


def _ratio(done: list[str], required: list[str]) -> float:
    if not required:
        return 1.0
    return len(set(done) & set(required)) / float(len(required))


def _normalize_state(state: AgenticSecurityLabState | dict[str, Any]) -> AgenticSecurityLabState:
    if isinstance(state, AgenticSecurityLabState):
        return state
    return AgenticSecurityLabState.model_validate(state)


def _grade_task(state: AgenticSecurityLabState | dict[str, Any], task_name: str) -> float:
    s = _normalize_state(state)
    req = _REQUIRED[task_name]

    breakdown = {
        "quarantine_ratio": _ratio(s.quarantined, req["quarantine"]),
        "rotate_ratio": _ratio(s.rotated_secrets, req["rotate_secret"]),
        "notify_ratio": _ratio(s.notified_teams, req["notify"]),
        "contain_ratio": 1.0 if not s.attacker_succeeded else 0.0,
    }
    score = _score_from_breakdown(breakdown)
    return round(max(_EPS, min(1.0 - _EPS, score)), 6)


def grade_easy(state: AgenticSecurityLabState | dict[str, Any]) -> float:
    return _grade_task(state, "easy")


def grade_medium(state: AgenticSecurityLabState | dict[str, Any]) -> float:
    return _grade_task(state, "medium")


def grade_hard(state: AgenticSecurityLabState | dict[str, Any]) -> float:
    return _grade_task(state, "hard")

