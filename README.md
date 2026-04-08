---
title: Agentic Security Lab Environment Server
emoji: 🚨
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - security
  - supply-chain
  - incident-response
---

# Agentic Security Lab: Supply Chain Incident Response

Train and evaluate agents on a high-stakes, real-world workflow: **responding to a confirmed software supply chain compromise before attacker exfiltration succeeds**.

This environment is inspired by modern npm ecosystem incidents (compromised maintainer accounts, slopsquatted packages, CI token theft campaigns). The agent is not doing passive detection. It must **actively coordinate incident response** over multiple steps with changing state and deadline pressure.

## Recent Threat Context

This benchmark is grounded in the same incident patterns teams are dealing with now:

- **Axios package compromise reports** highlighted how a single compromised maintainer flow can rapidly expose secrets across many downstream services.
- **LiteLLM package compromise reports** reinforced that AI tooling dependencies can become a supply-chain entry point with broad blast radius in production systems.
- **Multi-package npm campaigns** (e.g., credential theft clusters) showed attackers combining slopsquatting, token theft, and CI/CD exfiltration.

Agentic Security Lab operationalizes these patterns into deterministic tasks where action ordering directly changes containment outcomes.

## Why This Environment Matters

Most current agent benchmarks reward short, stateless tasks. Real security operations are not like that.

This environment models:
- dependency-graph blast radius analysis,
- package quarantine decisions under uncertainty,
- secret rotation prioritization (critical vs non-critical),
- downstream stakeholder notification at scale,
- race-against-the-clock containment before exfiltration.

The result is an RL-friendly setting with meaningful sequential decision-making and deterministic grading.

## Scenario Design

Three deterministic tasks with increasing difficulty:

### Easy
- Single compromised package: `axios@1.7.4`
- 2 exposed secrets
- 3 affected teams
- Exfiltration deadline: 14 steps

### Medium
- Transitive compromise: `form-data@4.0.1` (via `node-fetch`)
- 5 exposed secrets
- 12 affected teams
- Exfiltration deadline: 18 steps

### Hard
- Coordinated PhantomRaven-style campaign
- 5 malicious packages (slopsquatting)
- 8 secrets including CI/CD credentials
- 20 affected teams
- Exfiltration deadline: 10 steps

## Action Space

Actions are structured commands:

- `inspect_package` with `{"package": "<name@version>"}`
- `check_dependents` with `{"package": "<name@version>"}`
- `rotate_secret` with `{"secret": "<SECRET_NAME>"}`
- `quarantine` with `{"package": "<name@version>"}`
- `notify` with `{"team": "<team-name>"}`
- `scan_logs` with `{"package": "<name@version>"}`
- `conclude` with `{}`

## Observation Space

Each step returns:

- `success`: whether the action executed correctly
- `done`: episode termination flag
- `reward`: per-step shaped reward
- `result`: human-readable action result
- `data`: structured output for machine consumption
- `incident_summary`: compact progress dashboard
- `steps_remaining`: countdown to attacker success threshold
- `exposed_secrets`: not-yet-rotated credentials
- `active_malicious_packages`: malicious packages not yet quarantined
- `error`: optional action error reason

## Reward Design

Dense reward shaping provides trajectory-level signal, not just terminal binary success.

Per-step rewards:
- quarantine correct malicious package: `+0.15`
- rotate critical secret: `+0.12`
- rotate non-critical secret: `+0.06`
- notify affected team: `+0.04`
- scan logs for intel: `+0.02`
- inspect/check dependency metadata: `+0.01`

Penalties:
- quarantine clean package (false positive): `-0.05`
- re-rotate already rotated secret: `-0.02`
- invalid/unknown command: `-0.01`

Episode-end bonuses (on `conclude`/terminal):
- all required packages quarantined: `+0.10`
- all required secrets rotated: `+0.10`
- all required teams notified: `+0.05`
- contained before exfiltration: `+0.10`
- attacker succeeds: `-0.20`

## Quick Start

### 1) Install dependencies

```bash
pip install -U openenv-core
```

### 2) Run locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3) Interact with API

```bash
# Reset to hard task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"hard"}'

# Step: inspect a suspicious package
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"command":"inspect_package","parameters":{"package":"1odash@4.17.21"}}'
```

## Baseline Inference

`inference.py` runs a model through tasks and prints structured logs in hackathon format:

- `[START]`
- `[STEP]`
- `[END]`

Required environment variables:
- `HF_TOKEN` (or compatible API key)
- `API_BASE_URL` (optional override, defaults set in script)
- `MODEL_NAME` (optional override)
- `ENV_BASE_URL` (defaults to `http://localhost:8000`)
- `TASK_NAME` (optional: `easy|medium|hard`; otherwise runs all tasks)

Run:

```bash
python inference.py
```

## OpenEnv + Submission Compliance

This project includes:
- typed action/observation/state models (`models.py`),
- `step` / `reset` / `state` endpoints (`server/app.py`),
- environment metadata (`openenv.yaml`),
- containerized runtime (`Dockerfile`),
- baseline script (`inference.py`).

Pre-submit checks:

```bash
# from repo root
docker build -t agentic-security-lab .
openenv validate
```

If using the provided validator script:

```bash
./validate-submission.sh https://your-space.hf.space .
```

## Deploy to Hugging Face Spaces

From the environment directory:

```bash
openenv push --repo-id <username>/agentic-security-lab
```

After deployment, verify:
- `POST /reset` returns `200`
- task switching works (`easy`, `medium`, `hard`)
- inference runs end-to-end within runtime budget

## Project Structure

```text
agentic_security_lab/
├── README.md
├── openenv.yaml
├── inference.py
├── models.py
├── scenarios.py
├── client.py
├── pyproject.toml
├── Dockerfile
└── server/
    ├── app.py
    └── agentic_security_lab_environment.py
```

## What Makes It Novel

- Security operations instead of toy game dynamics.
- Realistic dependency-graph containment workflow.
- Mixed objectives (containment, credential safety, communications) with competing priorities.
- Deterministic scoring with partial credit over full trajectories.

This makes Agentic Security Lab useful for evaluating planning quality, action ordering, and risk-aware behavior in agentic systems.
