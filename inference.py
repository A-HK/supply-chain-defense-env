"""
inference.py — agentic_security_lab/inference.py  (project root)
=================================================================
Mandatory stdout format:
    [START] task=<n> env=agentic-security-lab model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Env vars (set in HF Space secrets):
    API_BASE_URL    LLM endpoint  — default: HF inference router
    MODEL_NAME      model id      — default: Qwen/Qwen2.5-7B-Instruct
    HF_TOKEN        API key       — injected automatically by the validator
    ENV_BASE_URL    Space URL     — default: localhost (validator sets this)
    TASK_NAME       easy|medium|hard  (omit to run all three)
"""
import json
import os
import textwrap
from typing import Optional

import httpx
from openai import OpenAI
from graders import grade_easy, grade_hard, grade_medium

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

ALL_TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 30
BENCHMARK = "agentic-security-lab"
SCORE_EPS = 1e-6

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert security engineer responding to a supply-chain incident.
    Issue ONE action per turn as a JSON object with keys "command" and "parameters".

    Commands:
      inspect_package   {"package": "<name@version>"}
      check_dependents  {"package": "<name@version>"}
      rotate_secret     {"secret": "<SECRET_NAME>"}
      quarantine        {"package": "<name@version>"}
      notify            {"team": "<team-name>"}
      scan_logs         {"package": "<name@version>"}
      conclude          {}

    Strategy:
      1. scan_logs / inspect_package on all packages to identify IOCs.
      2. quarantine every malicious package immediately.
      3. check_dependents to find all affected downstream teams.
      4. rotate_secret for every exposed credential — critical ones first.
      5. notify every affected team.
      6. conclude when done.

    Reply with ONLY the JSON object. No markdown, no explanation, no prose.
    Example: {"command": "quarantine", "parameters": {"package": "axios@1.7.4"}}
""").strip()


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.6f} rewards={rewards_str}",
        flush=True,
    )


def env_reset(http: httpx.Client, task_name: str) -> dict:
    r = http.post("/reset", json={"task_name": task_name})
    r.raise_for_status()
    return r.json()


def env_step(http: httpx.Client, command: str, parameters: dict) -> dict:
    r = http.post("/step", json={"command": command, "parameters": parameters})
    r.raise_for_status()
    return r.json()


def env_state(http: httpx.Client) -> dict:
    r = http.get("/state")
    r.raise_for_status()
    return r.json()


def parse_action(raw: str) -> tuple:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(l for l in lines if not l.startswith("```")).strip()
    try:
        obj = json.loads(text)
        return obj.get("command", "conclude"), obj.get("parameters", {})
    except (json.JSONDecodeError, AttributeError):
        return "conclude", {}


def run_task(llm: OpenAI, http: httpx.Client, task_name: str) -> float:
    rewards = []
    step_n  = 0
    success = False
    score   = 0.0

    log_start(task=task_name, model=MODEL_NAME)

    try:
        obs = env_reset(http, task_name)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": obs["result"]},
        ]

        for step_n in range(1, MAX_STEPS + 1):

            try:
                resp = llm.chat.completions.create(
                    model       = MODEL_NAME,
                    messages    = messages,
                    max_tokens  = 150,
                    temperature = 0.2,
                )
                raw = resp.choices[0].message.content or ""
            except Exception as llm_err:
                raw = '{"command": "conclude", "parameters": {}}'
                _ = llm_err

            command, params = parse_action(raw)
            action_str      = json.dumps({"command": command, "parameters": params})

            obs    = env_step(http, command, params)
            reward = float(obs.get("reward", 0.0))
            done   = bool(obs.get("done", False))
            error  = obs.get("error") or None

            rewards.append(reward)
            log_step(step=step_n, action=action_str, reward=reward,
                     done=done, error=error)

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",      "content": obs["result"]})

            if done:
                break

        final_state = env_state(http)
        graders = {
            "easy": grade_easy,
            "medium": grade_medium,
            "hard": grade_hard,
        }
        score = graders[task_name](final_state)
        score = max(SCORE_EPS, min(1.0 - SCORE_EPS, score))
        success = (
            score >= 1.0 - SCORE_EPS
            and not bool(final_state.get("attacker_succeeded", False))
        )

    except Exception:
        log_end(success=False, steps=step_n, score=0.0, rewards=rewards)
        return 0.0

    log_end(success=success, steps=step_n, score=score, rewards=rewards)
    return score


def main() -> None:
    llm  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    http = httpx.Client(base_url=ENV_BASE_URL, timeout=60)

    task_env = os.getenv("TASK_NAME", "")
    tasks    = [task_env] if task_env in ALL_TASKS else ALL_TASKS

    try:
        for task in tasks:
            run_task(llm, http, task)
    finally:
        http.close()


if __name__ == "__main__":
    main()