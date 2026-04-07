"""
inference.py — place this at the ROOT of agentic_security_lab/
=============================================================
Mandatory stdout format (do not change field names or order):
    [START] task=<name> env=agentic-security-lab model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Required env vars:
    HF_TOKEN        your Hugging Face / API key
    API_BASE_URL    LLM endpoint   (default: HF router)
    MODEL_NAME      model id       (default: Qwen/Qwen2.5-72B-Instruct)
    ENV_BASE_URL    server URL     (default: http://localhost:8000)
    TASK_NAME       easy|medium|hard  (default: runs all three)
"""
import json
import os
import sys
import textwrap

import httpx
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY = (
    os.getenv("GROQ_API_KEY")
    or os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or ""
)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")


ALL_TASKS  = ["easy", "medium", "hard"]
MAX_STEPS  = 30

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
      1. scan_logs / inspect_package on all packages to find IOCs.
      2. quarantine every malicious package.
      3. check_dependents to find all affected teams.
      4. rotate_secret for every exposed credential — critical ones first.
      5. notify every affected team.
      6. conclude when all actions are done.

    Reply with ONLY the JSON object. No markdown, no explanation.
    Example: {"command": "quarantine", "parameters": {"package": "axios@1.7.4"}}
""").strip()

llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
http        = httpx.Client(base_url=ENV_BASE_URL, timeout=60)


def env_reset(task_name: str) -> dict:
    r = http.post("/reset", json={"task_name": task_name})
    r.raise_for_status()
    return r.json()


def env_step(command: str, parameters: dict) -> dict:
    r = http.post("/step", json={"command": command, "parameters": parameters})
    r.raise_for_status()
    return r.json()


def parse_action(raw: str) -> tuple[str, dict]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(l for l in lines if not l.startswith("```")).strip()
    try:
        obj = json.loads(text)
        return obj.get("command", "conclude"), obj.get("parameters", {})
    except json.JSONDecodeError:
        return "conclude", {}


def run_task(task_name: str) -> float:
    obs = env_reset(task_name)
    print(f"[START] task={task_name} env=agentic-security-lab model={MODEL_NAME}",
          flush=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": obs["result"]},
    ]

    rewards: list[float] = []
    step_n  = 0
    success = False

    for step_n in range(1, MAX_STEPS + 1):
        resp = llm_client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = messages,
            max_tokens  = 150,
            temperature = 0.2,
        )
        raw             = resp.choices[0].message.content or ""
        command, params = parse_action(raw)
        action_str      = json.dumps({"command": command, "parameters": params})

        obs    = env_step(command, params)
        reward = obs.get("reward", 0.0)
        done   = obs.get("done", False)
        error  = obs.get("error") or "null"
        rewards.append(reward)

        print(
            f"[STEP] step={step_n} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error={error}",
            flush=True,
        )

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",      "content": obs["result"]})

        if done:
            success = sum(rewards) > 0
            break

    final_score  = min(1.0, max(0.0, sum(rewards)))
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step_n} "
        f"score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )
    return final_score


def main():
    print(API_BASE_URL)
    print(MODEL_NAME)
    print(API_KEY)
    print(ENV_BASE_URL)
    task_env = os.getenv("TASK_NAME", "")
    tasks    = [task_env] if task_env in ALL_TASKS else ALL_TASKS

    scores = {}
    for task in tasks:
        scores[task] = run_task(task)

    print("\n=== Baseline Results ===", flush=True)
    for task, score in scores.items():
        print(f"  {task:8s}  score={score:.2f}", flush=True)


if __name__ == "__main__":
    main()