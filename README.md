---
title: Agentic Security Lab
emoji: 🔒
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
license: mit
base_path: /web
---

# Agentic Security Lab — Supply Chain Incident Response

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible RL environment where an agent acts as an on-call security engineer responding to a live npm/PyPI supply chain attack.

Inspired by real 2025–2026 incidents: the **Axios npm compromise** (North Korean attribution), the **LiteLLM PyPI backdoor**, and the **PhantomRaven** slopsquatting campaign (126 malicious packages, 86,000 installs).

---

## Motivation

Supply chain attacks are the highest-impact class of incident for engineering teams right now. When a maintainer account is hijacked, the blast radius spans hundreds of downstream services within minutes. Yet no RL benchmark trains agents to *respond* operationally — not just detect statically. This environment fills that gap by modelling what a senior security engineer actually does during an active incident: trace the dependency graph, contain the malicious package, rotate every exposed credential, and notify affected teams — before the attacker exfiltrates them.

---

## Environment Description

At episode start, the agent receives an **incident alert** describing the attack scenario. It must issue a sequence of discrete actions to contain the threat. A simulated attacker runs on a countdown timer — if critical secrets are not rotated before the `exfiltration_step`, the attacker succeeds and the episode ends with a penalty.

The environment is fully deterministic and reproducible. The same task always produces the same scenario, enabling fair comparison across agents and runs.

---

## Action Space

| Command | Parameters | Description |
|---|---|---|
| `inspect_package` | `{"package": "name@version"}` | View publisher, publish date, and IOC indicators |
| `check_dependents` | `{"package": "name@version"}` | List all downstream consumer services |
| `rotate_secret` | `{"secret": "SECRET_NAME"}` | Revoke and regenerate a named credential |
| `quarantine` | `{"package": "name@version"}` | Yank the package version from the registry |
| `notify` | `{"team": "team-name"}` | Send breach alert to a downstream team |
| `scan_logs` | `{"package": "name@version"}` | Retrieve CI/CD log evidence and IOC patterns |
| `conclude` | `{}` | Declare incident contained — ends the episode |

---

## Observation Space

Each `step()` and `reset()` returns:

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether the action succeeded |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Per-step reward signal |
| `result` | `str` | Human-readable description of the action outcome |
| `data` | `dict` | Structured output (metadata, dep list, log lines) |
| `incident_summary` | `str` | Running dashboard: step count, quarantined, rotated, notified |
| `steps_remaining` | `int` | Steps before attacker completes exfiltration |
| `exposed_secrets` | `list[str]` | Secret names not yet rotated |
| `active_malicious_packages` | `list[str]` | Malicious packages not yet quarantined |
| `error` | `str or None` | Set when the action was invalid |

---

## Reward Function

Dense reward signal across the full trajectory — not sparse end-of-episode.

**Per-step rewards:**

| Action | Reward |
|---|---|
| Quarantine correct malicious package | +0.15 |
| Rotate critical secret | +0.12 |
| Rotate non-critical secret | +0.06 |
| Notify affected team | +0.04 |
| scan_logs (returns intel) | +0.02 |
| inspect_package | +0.01 |
| check_dependents | +0.01 |
| Quarantine clean package (false positive) | -0.05 |
| Rotate already-rotated secret | -0.02 |
| Invalid / unknown command | -0.01 |

**Episode bonuses (awarded on `conclude`):**

| Condition | Bonus |
|---|---|
| All required packages quarantined | +0.10 |
| All required secrets rotated | +0.10 |
| All required teams notified | +0.05 |
| Incident contained before exfiltration deadline | +0.10 |
| Attacker successfully exfiltrated credentials | -0.20 |

---

## Tasks

### easy — Single Package Compromise

A single npm package (`axios@1.7.4`) was hijacked via a compromised maintainer account. Two secrets are at risk. Three downstream consumer services are affected. The exfiltration window is 14 steps.

- Packages in scope: `axios@1.7.4` (malicious), `axios@1.7.3` (clean)
- Secrets: `STRIPE_SECRET_KEY` (critical), `INTERNAL_API_TOKEN`
- Teams to notify: `payments-service`, `auth-service`, `api-gateway`
- Expected difficulty: Solvable by a prompted frontier model
- Optimal cumulative reward: ~0.80

### medium — Transitive Dependency Attack

`form-data@4.0.1` was backdoored but is not a direct dependency — it is a dependency of `node-fetch`. The agent must trace the graph to identify the root malicious package. Five secrets exposed. Twelve consumer teams affected. Exfiltration window is 18 steps.

- Root cause: `form-data@4.0.1` (not `node-fetch@3.3.2`)
- Secrets: AWS keys, SendGrid, Postgres password, webhook signing secret
- Teams to notify: 12 services including `data-pipeline`, `email-sender`, `crm-sync`
- Expected difficulty: Requires multi-hop dependency reasoning
- Optimal cumulative reward: ~1.52

### hard — PhantomRaven Coordinated Campaign

Five slopsquatted packages published by the same actor cluster. CI/CD tokens (`NPM_TOKEN`, `GITHUB_ACTIONS_TOKEN`) are already being exfiltrated. The attacker has a 10-step head start. Eight secrets, twenty consumer teams. Frontier models will struggle to complete this within budget.

- Malicious packages: `1odash@4.17.21`, `expresss@4.18.2`, `axios-http@1.7.4`, `node-fetch-lite@3.3.2`, `dotenvv@16.0.3`
- Secrets: 6 critical including AWS keys, DB connection string, JWT secret
- Teams to notify: 20 services across 5 package dependency trees
- Expected difficulty: Genuinely challenges frontier models
- Optimal cumulative reward: ~2.74

---

## Setup and Usage

### Prerequisites

```bash
pip install openenv-core
```

### Run locally with uv

```bash
cd agentic_security_lab
uv sync
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker

```bash
docker build -t agentic-security-lab .
docker run -p 8000:8000 agentic-security-lab
```

### Smoke test

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy"}'

curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"command": "inspect_package", "parameters": {"package": "axios@1.7.4"}}'

curl http://localhost:8000/state
```

### Run inference baseline

```bash
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.3-70b-versatile
set HF_TOKEN=gsk_your_groq_key
set ENV_BASE_URL=http://localhost:8000
python inference.py
```

### Validate and deploy

```bash
openenv validate
openenv push --repo-id your-username/agentic-security-lab
```

---

## Project Structure

```
agentic_security_lab/
├── models.py                          # Typed Pydantic Action / Observation / State
├── scenarios.py                       # Deterministic task definitions
├── client.py                          # Sync HTTP client
├── inference.py                       # Baseline inference script
├── openenv.yaml                       # OpenEnv manifest (pure YAML, no frontmatter)
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── README.md                          # HF Space frontmatter lives here
└── server/
    ├── __init__.py
    ├── app.py
    ├── agentic_security_lab_environment.py
    └── requirements.txt
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Reset environment. Body: `{"task_name": "easy/medium/hard"}` |
| POST | `/step` | Take an action. Body: `{"command": "...", "parameters": {...}}` |
| GET | `/state` | Full internal episode state |
| GET | `/health` | Health check |
| GET | `/` | Environment metadata |

---

## Baseline Scores

Collected with `llama-3.3-70b-versatile` via Groq, temperature=0.2, MAX_STEPS=30:

| Task | Score (0-1) | Notes |
|---|---|---|
| easy | 0.42 | Correctly quarantines and rotates; misses some notify steps |
| medium | 0.28 | Struggles to trace transitive dep to root cause |
| hard | 0.14 | Overwhelmed by 5-package scope and 10-step head start |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | API key (Groq key or HF token) |
| `ENV_BASE_URL` | Yes in production | Your HF Space URL |
| `TASK_NAME` | No | Run single task: easy, medium, or hard |