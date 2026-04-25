"""Hackathon-format baseline inference for Agentic Security Lab."""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any

import httpx
from openai import OpenAI

from planning import LongHorizonPlanner, PlanMemory, Replanner
from world_model.model import LightweightWorldModel
from world_model.rollout import choose_best_action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
TASK_NAME = os.getenv("TASK_NAME", "")
USE_PLANNER = os.getenv("USE_PLANNER", "1") == "1"
WORLD_MODEL_PATH = os.getenv("WORLD_MODEL_PATH", "artifacts/world_model.json")

ALL_TASKS = ["easy", "medium", "hard"]
BENCHMARK = "agentic-security-lab"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are responding to a live software supply-chain compromise.
    Return exactly one JSON object with keys "command" and "parameters".

    Valid commands:
    - inspect_package {"package": "<name@version>"}
    - check_dependents {"package": "<name@version>"}
    - rotate_secret {"secret": "<SECRET_NAME>"}
    - quarantine {"package": "<name@version>"}
    - notify {"team": "<team-name>"}
    - scan_logs {"package": "<name@version>"}
    - conclude {}

    Prioritize: investigate, trace root cause, contain, rotate secrets, notify teams, conclude.
    Reply with JSON only.
    """
).strip()


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_value = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_value}",
        flush=True,
    )


def env_reset(http: httpx.Client, task_name: str) -> dict[str, Any]:
    response = http.post("/reset", json={"task_name": task_name, "mode": "benchmark"})
    response.raise_for_status()
    return response.json()


def env_step(http: httpx.Client, command: str, parameters: dict[str, Any]) -> dict[str, Any]:
    response = http.post("/step", json={"command": command, "parameters": parameters})
    response.raise_for_status()
    return response.json()


def parse_action(raw: str) -> tuple[str, dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        return "conclude", {}
    return action.get("command", "conclude"), action.get("parameters", {})


def build_messages(observation: dict[str, Any], history: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": observation["result"]},
    ]


def candidate_actions(observation: dict[str, Any], memory: PlanMemory) -> list[dict[str, Any]]:
    planner = LongHorizonPlanner()
    plan_actions = planner.build_plan(observation, memory)
    return plan_actions or [{"command": "conclude", "parameters": {}}]


def load_world_model() -> LightweightWorldModel | None:
    if not os.path.exists(WORLD_MODEL_PATH):
        return None
    try:
        return LightweightWorldModel.load(WORLD_MODEL_PATH)
    except Exception:
        return None


def run_task(llm: OpenAI, http: httpx.Client, task_name: str, world_model: LightweightWorldModel | None) -> float:
    rewards: list[float] = []
    score = 0.0
    step_count = 0
    success = False
    planner_memory = PlanMemory()
    replanner = Replanner()
    last_reward = 0.0
    history: list[dict[str, str]] = []

    log_start(task_name, MODEL_NAME)
    try:
        observation = env_reset(http, task_name)
        max_steps = int(observation.get("data", {}).get("max_steps", 20))
        for step_count in range(1, max_steps + 1):
            planned_action = None
            if USE_PLANNER:
                if replanner.should_replan(last_reward, observation, planner_memory):
                    planner_memory.action_queue = []
                if not planner_memory.action_queue:
                    planner_memory.action_queue = candidate_actions(observation, planner_memory)
                if world_model and planner_memory.action_queue:
                    imagined = choose_best_action(world_model, planner_memory.action_queue, observation)
                    if imagined:
                        planned_action = imagined
                        planner_memory.action_queue = [
                            action for action in planner_memory.action_queue if action != planned_action
                        ]
                elif planner_memory.action_queue:
                    planned_action = planner_memory.action_queue.pop(0)

            if planned_action is None:
                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=build_messages(observation, history),
                    max_tokens=150,
                    temperature=0.2,
                )
                raw = response.choices[0].message.content or ""
                command, parameters = parse_action(raw)
            else:
                command = planned_action["command"]
                parameters = planned_action["parameters"]

            action_payload = json.dumps({"command": command, "parameters": parameters})
            observation = env_step(http, command, parameters)
            reward = float(observation.get("reward", 0.0))
            done = bool(observation.get("done", False))
            error = observation.get("error")

            rewards.append(reward)
            last_reward = reward
            if reward > 0 and command in {"scan_logs", "inspect_package"}:
                planner_memory.mark_completed("investigate")
            if reward > 0 and command == "check_dependents":
                planner_memory.mark_completed("trace_root_cause")
            if reward > 0 and command == "quarantine":
                planner_memory.mark_completed("contain")
            if reward > 0 and command == "rotate_secret":
                planner_memory.mark_completed("recover")
            if reward > 0 and command == "notify":
                planner_memory.mark_completed("notify")
            if done:
                planner_memory.mark_completed("conclude")

            history.extend(
                [
                    {"role": "assistant", "content": action_payload},
                    {"role": "user", "content": observation["result"]},
                ]
            )
            log_step(step_count, action_payload, reward, done, error)

            score = float(observation.get("data", {}).get("benchmark_score", 0.0))
            if done:
                break

        success = score >= 0.8
        log_end(success, step_count, score, rewards)
        return score
    except Exception:
        log_end(False, step_count, 0.0, rewards)
        raise


def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    world_model = load_world_model()
    task_names = [TASK_NAME] if TASK_NAME in ALL_TASKS else ALL_TASKS

    with httpx.Client(base_url=ENV_BASE_URL, timeout=60) as http:
        for task_name in task_names:
            run_task(llm, http, task_name, world_model)


if __name__ == "__main__":
    main()
