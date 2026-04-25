---
title: Agentic Security Lab Environment Server
emoji: "🚨"
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

Agentic Security Lab is an OpenEnv environment for training and evaluating agents on a high-stakes workflow: responding to a software supply-chain compromise before attacker exfiltration succeeds.

The benchmark is designed around three Round 2 themes:

- Theme #2: long-horizon planning under deadline pressure
- Theme #3: world modeling in a partially observable professional workflow
- Theme #4: self-improvement through rollout collection, policy training, and world-model refresh

## Problem Framing

When a dependency is compromised, success is not just "spot one IOC." The hard part is sequencing the response:

- inspect suspicious packages,
- trace blast radius through dependency graphs,
- quarantine the right package versions,
- rotate the right secrets in the right order,
- notify downstream teams before the attacker cashes out.

That is the behavior this environment scores.

## Threat Context

This benchmark is inspired by recent public incident patterns:

- Axios npm compromise and downstream credential risk
- LiteLLM PyPI compromise and urgent secret rotation guidance
- PhantomRaven-style npm campaigns using slopsquatting and CI/CD token theft

The environment itself is deterministic in `benchmark` mode and self-contained; the threat references motivate the task design, but scoring does not depend on external services.

## Scenario Design

Three fixed tasks are included:

- `easy`: one malicious package, two secrets, three downstream teams
- `medium`: transitive compromise via `form-data@4.0.1`, five secrets, twelve teams
- `hard`: five malicious packages, eight secrets, twenty teams, tight exfiltration window

Each task uses the same action interface but increases horizon length, blast radius, and coordination load.

## API Contract

### Actions

- `inspect_package` with `{"package": "<name@version>"}`
- `check_dependents` with `{"package": "<name@version>"}`
- `rotate_secret` with `{"secret": "<SECRET_NAME>"}`
- `quarantine` with `{"package": "<name@version>"}`
- `notify` with `{"team": "<team-name>"}`
- `scan_logs` with `{"package": "<name@version>"}`
- `conclude` with `{}`

### Reset Options

`POST /reset` supports:

- `task_name`
- `mode=benchmark|training`
- `command_fallback_enabled=true|false`

Invalid mode values fall back to `benchmark` and the fallback is reported in state and evaluator telemetry.

### Observation Fields

Each step returns:

- `reward`: dense training reward for the current action
- `data.benchmark_score`: deterministic evaluator score in `[0, 1]`
- `data.score_breakdown`: `quarantine_ratio`, `rotate_ratio`, `notify_ratio`, `contain_ratio`
- `data.evaluator_metrics`: invalid actions, false positives, mode fallback, command fallback count, deadline status
- `active_malicious_packages`: confirmed malicious packages not yet quarantined
- `exposed_secrets`: discovered secrets not yet rotated

The environment no longer leaks all malicious packages or secrets at reset. Discovery must come from package inspection, log scans, and dependency tracing.

## Deterministic Grading

The benchmark score is computed from the current state:

`score = 0.35 * quarantine_ratio + 0.35 * rotate_ratio + 0.20 * notify_ratio + 0.10 * contain_ratio`

Implementation details:

- `quarantine_ratio` = required malicious packages quarantined / total required malicious packages
- `rotate_ratio` = required secrets rotated / total required secrets
- `notify_ratio` = required teams notified / total required teams
- `contain_ratio` = `1.0` only when all required malicious packages are contained and the attacker has not succeeded
- score is clamped to `[0, 1]` and rounded to 6 decimals

This score is separate from the dense per-step training reward.

## Reward Design

Dense reward shaping is preserved for rollout learning:

- correct malicious package quarantine: `+0.15`
- rotate critical secret: `+0.12`
- rotate non-critical secret: `+0.06`
- notify affected team: `+0.04`
- scan logs: `+0.02`
- inspect/check dependency metadata: `+0.01`

Penalties:

- false-positive quarantine: `-0.05`
- re-rotate an already rotated secret: `-0.02`
- invalid target or unknown command: `-0.01`
- immediate or low-progress conclude: negative reward instead of a free bonus

## Planner, World Model, and Training

The repo now includes:

- `planning/`: plan memory, goal progression, and replanning logic
- `world_model/`: transition-conditioned action ranking model trained from collected rollouts
- `training/train_grpo.py`: live-environment rollout collection plus TRL-backed LoRA policy training
- `training/self_improve.py`: repeated collect -> train -> rebuild world model loop
- `notebooks/round2_training_colab.ipynb`: Colab notebook that installs deps, starts the server, trains, evaluates, and exports plots

## Quick Start

### Install

```bash
pip install -U openenv-core
pip install -e .
```

For training:

```bash
pip install -e .[train]
```

### Run the Environment

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Smoke Test

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"hard","mode":"benchmark"}'
```

## Baseline Inference

`inference.py` emits only the required hackathon log lines:

- `[START]`
- `[STEP]`
- `[END]`

It uses `benchmark_score` for final scoring instead of summing shaped rewards.

Required environment variables:

- `HF_TOKEN` or compatible API key
- `API_BASE_URL` and `MODEL_NAME` for model routing
- `ENV_BASE_URL` for the environment server

Run:

```bash
python inference.py
```

## Training and Evaluation

Collect trajectories and train a LoRA policy:

```bash
python training/train_grpo.py \
  --env-base-url http://localhost:8000 \
  --task medium \
  --episodes 12 \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir artifacts/checkpoints/policy
```

Train the world model from collected transitions:

```bash
python world_model/train_world_model.py \
  --transitions artifacts/transitions.jsonl \
  --output artifacts/world_model.json
```

Summarize and plot metrics:

```bash
python training/evaluate.py
python training/plot_metrics.py
```

## Colab

The notebook lives at [notebooks/round2_training_colab.ipynb](notebooks/round2_training_colab.ipynb) and keeps the existing file path so it can be linked directly from the submission.

## Validation

Run the local checks before submission:

```bash
openenv validate
docker build -t agentic-security-lab .
```

For the provided validator:

```bash
./validate-submission.sh https://your-space.hf.space .
```

## Artifacts

The repo expects the following generated artifacts:

- [artifacts/reward_curve.png](artifacts/reward_curve.png)
- [artifacts/loss_curve.png](artifacts/loss_curve.png)
- [artifacts/baseline_vs_trained.png](artifacts/baseline_vs_trained.png)
- `artifacts/metrics.jsonl`
- `artifacts/transitions.jsonl`
- `artifacts/world_model.json`

### Inline Plots

![Reward Curve](artifacts/reward_curve.png)

Reward collected across rollout episodes.

![Loss Curve](artifacts/loss_curve.png)

Training loss proxy logged during trajectory collection and policy tuning.

![Baseline vs Trained](artifacts/baseline_vs_trained.png)

Visual comparison plot used in the submission package.

## Submission Links

Fill these in before submitting:

- Hugging Face Space: `<ADD_PUBLIC_SPACE_URL>`
- Colab Notebook: [notebooks/round2_training_colab.ipynb](notebooks/round2_training_colab.ipynb)
- Blog / video / slides: `<ADD_PUBLIC_WRITEUP_URL>`

## Project Structure

```text
agentic_security_lab/
|-- README.md
|-- openenv.yaml
|-- inference.py
|-- models.py
|-- scenarios.py
|-- client.py
|-- pyproject.toml
|-- Dockerfile
|-- planning/
|-- training/
|-- world_model/
|-- notebooks/
|-- artifacts/
`-- server/
```

## Current Status

Implemented in code:

- validator-clean packaging
- benchmark/training modes
- fallback accounting
- deterministic benchmark grading
- hidden-state removal at reset
- hackathon-format inference logging
- TRL-backed policy-training entrypoint
- Colab notebook path retained and updated

Still user-supplied before final submission:

- public HF Space URL
- public writeup / video / slides link
