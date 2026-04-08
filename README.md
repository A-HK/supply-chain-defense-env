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

When a dependency gets compromised, teams do not fail because they cannot detect one IOC. They fail because containment is a sequence problem: what to quarantine first, which secrets to rotate immediately, who to notify, and how to do it before the attacker cashes out.

This environment models that exact pressure. The agent is not doing passive detection. It must **actively coordinate incident response** over multiple steps with changing state, incomplete information, and deadline pressure.

## Recent Threat Context

This benchmark is grounded in publicly reported incident patterns teams are dealing with now:

- **Axios npm compromise**: public postmortems and threat-intel writeups describe malicious Axios versions published from a compromised maintainer account, with downstream credential risk at scale. See [axios maintainer postmortem](http://github.com/axios/axios/issues/10636), [Microsoft analysis](https://www.microsoft.com/en-us/security/blog/2026/04/01/mitigating-the-axios-npm-supply-chain-compromise/), and [Endor Labs report](https://www.endorlabs.com/learn/npm-axios-compromise).
- **LiteLLM PyPI compromise**: maintainers and advisories documented malicious LiteLLM releases tied to credential theft behavior and urgent secret rotation guidance. See [LiteLLM incident thread](https://github.com/BerriAI/litellm/issues/24518), [LiteLLM security update](https://docs.litellm.ai/blog/security-update-march-2026), and [GitLab advisory](https://advisories.gitlab.com/pkg/pypi/litellm/GHSA-5mg7-485q-xm76/).
- **PhantomRaven-style npm campaigns**: large multi-package credential theft operations emphasize how attackers blend slopsquatting, token theft, and CI/CD secret exfiltration. See [Sonatype coverage](https://www.sonatype.com/blog/phantomraven-npm-malware) and [OSSF malicious-packages tracking](https://github.com/ossf/malicious-packages/issues/1166).

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

The `data` field includes evaluator-facing telemetry:
- `reward_type`: always `training_step_reward`
- `benchmark_score`: deterministic task score computed from current state
- `score_breakdown`: component ratios (`quarantine`, `rotate`, `notify`, `contain`)
- `evaluator_metrics`: audit-friendly counts (invalid actions, fallback use, false positives, deadline status)

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

## Mode and Fallback Architecture

The environment supports explicit execution modes and safe fallbacks:

- `mode=benchmark` (default): deterministic evaluation behavior for leaderboard runs.
- `mode=training`: same state transitions with evaluator telemetry preserved for training analysis.
- Unsupported mode values safely fallback to `benchmark`, and `mode_fallback_used=true` is exposed in `state` and `data.evaluator_metrics`.

Command fallback can be enabled for robustness:
- aliases like `inspect`, `deps`, `dependents`, `rotate` map to canonical commands.
- fallback usage is counted in `command_fallback_used_count`.

You can set defaults via environment variables:
- `AGENTIC_SECURITY_LAB_MODE=benchmark|training`
- `AGENTIC_SECURITY_LAB_COMMAND_FALLBACK=true|false`

Or override at runtime via `POST /reset` body:
- `mode`
- `command_fallback_enabled`

## Training Reward vs Benchmark Score

The environment now separates learning signal from benchmark evaluation:

- `reward` (per-step): **training reward** used for RL trajectories.
- `benchmark_score` (in `observation.data`): deterministic score for evaluator reporting.

This makes it clear that reward shaping drives behavior during episodes, while benchmark score reflects objective completion quality.

## Grader Definition (Deterministic)

Task-level grading is deterministic and normalized to `[0, 1]` using explicit component completion ratios from final `state`.

For each task (`easy`, `medium`, `hard`), required sets are fixed:
- required malicious packages to quarantine,
- required secrets to rotate,
- required teams to notify.

Component ratios:
- `quarantine_ratio = |quarantined ∩ required_quarantine| / |required_quarantine|`
- `rotate_ratio = |rotated_secrets ∩ required_rotate_secret| / |required_rotate_secret|`
- `notify_ratio = |notified_teams ∩ required_notify| / |required_notify|`
- `contain_ratio = 1.0 if attacker_succeeded == false else 0.0`

Final score:

`score = 0.35 * quarantine_ratio + 0.35 * rotate_ratio + 0.20 * notify_ratio + 0.10 * contain_ratio`

Implementation details:
- score is clamped to `[0, 1]`,
- score is rounded to 6 decimals,
- same state always yields the same score (no randomness).

Worked example (`medium`):
- required: `1` quarantine, `5` secret rotations, `12` notifications
- achieved: `1` quarantine, `4` secret rotations, `9` notifications
- attacker did **not** succeed

Then:
- `quarantine_ratio = 1/1 = 1.00`
- `rotate_ratio = 4/5 = 0.80`
- `notify_ratio = 9/12 = 0.75`
- `contain_ratio = 1.00`

`score = 0.35*(1.00) + 0.35*(0.80) + 0.20*(0.75) + 0.10*(1.00) = 0.88`

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
