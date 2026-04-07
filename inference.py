"""
inference.py
Place at: agentic_security_lab/inference.py  (project root)
============================================================
Mandatory stdout format — do NOT change field names or order:

    [START] task=<n> env=agentic-security-lab model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Required env vars:
    API_BASE_URL    LLM endpoint   (set to Groq or HF router)
    MODEL_NAME      model id
    HF_TOKEN        API key for the LLM provider
    ENV_BASE_URL    Your deployed HF Space URL, e.g. https://a-hk-agentic-security-lab.hf.space
                    (defaults to localhost only for local dev)
    TASK_NAME       easy | medium | hard  (omit to run all three)
"""
import json
import os
import textwrap
from typing import Optional

import httpx
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
# IMPORTANT: ENV_BASE_URL must point to your HF Space URL when running the
# official validation. Set it as a Space secret or pass it as an env var.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
API_KEY      = (
    os.getenv("GROQ_API_KEY")
    or os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or ""
)
ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "http://localhost:8000"   # override with your HF Space URL in production
)

ALL_TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 30
BENCHMARK = "agentic-security-lab"

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

# ── Logging helpers (exact format required by validator) ───────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ── Environment HTTP helpers ───────────────────────────────────────────────────

def make_http_client() -> httpx.Client:
    return httpx.Client(base_url=ENV_BASE_URL, timeout=60)


def env_reset(http: httpx.Client, task_name: str) -> dict:
    r = http.post("/reset", json={"task_name": task_name})
    r.raise_for_status()
    return r.json()


def env_step(http: httpx.Client, command: str, parameters: dict) -> dict:
    r = http.post("/step", json={"command": command, "parameters": parameters})
    r.raise_for_status()
    return r.json()


# ── Action parsing ─────────────────────────────────────────────────────────────

def parse_action(raw: str) -> tuple[str, dict]:
    """Parse LLM output into (command, parameters). Handles markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(l for l in lines if not l.startswith("```")).strip()
    try:
        obj = json.loads(text)
        return obj.get("command", "conclude"), obj.get("parameters", {})
    except json.JSONDecodeError:
        return "conclude", {}


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_task(llm: OpenAI, http: httpx.Client, task_name: str) -> float:
    rewards: list[float] = []
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
            # Get action from LLM
            resp = llm.chat.completions.create(
                model       = MODEL_NAME,
                messages    = messages,
                max_tokens  = 150,
                temperature = 0.2,
            )
            raw             = resp.choices[0].message.content or ""
            command, params = parse_action(raw)
            action_str      = json.dumps({"command": command, "parameters": params})

            # Step environment
            obs    = env_step(http, command, params)
            reward = float(obs.get("reward", 0.0))
            done   = bool(obs.get("done", False))
            error  = obs.get("error") or None

            rewards.append(reward)

            log_step(step=step_n, action=action_str, reward=reward,
                     done=done, error=error)

            # Extend conversation
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",      "content": obs["result"]})

            if done:
                break

        # Score: clamp cumulative reward to [0, 1] as required by spec
        raw_score = sum(rewards)
        score     = min(1.0, max(0.0, raw_score))
        success   = score > 0.0

    except Exception:
        # [END] must always be emitted — even on exception
        log_end(success=False, steps=step_n, score=0.0, rewards=rewards)
        raise

    log_end(success=success, steps=step_n, score=score, rewards=rewards)
    return score


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    llm  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    http = make_http_client()

    task_env = os.getenv("TASK_NAME", "")
    tasks    = [task_env] if task_env in ALL_TASKS else ALL_TASKS

    scores: dict[str, float] = {}
    try:
        for task in tasks:
            scores[task] = run_task(llm, http, task)
    finally:
        http.close()

    print("\n=== Baseline Results ===", flush=True)
    for task, s in scores.items():
        print(f"  {task:8s}  score={s:.2f}", flush=True)


if __name__ == "__main__":
    main()