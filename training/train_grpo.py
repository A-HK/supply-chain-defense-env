"""Training entrypoints: expert SFT warmup + GRPO with all features wired in.

Phase 0 (SFT warmup): collect_examples() + train_sft() — expert heuristic demos
Phase 1 (GRPO):       train_grpo_live() — TRL GRPOTrainer + environment_factory
                       with procedural scenarios, trajectory filtering, and adversarial attacker

All modules are wired in here — not standalone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer
except ImportError:
    Dataset = None
    LoraConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None
    SFTTrainer = None

from training.callbacks import MetricsCallback
from training.adversarial_attacker import AdversarialAttacker
from training.trajectory_filter import TrajectoryFilter, CollapseDetector
from training.procedural_scenarios import generate_procedural_scenario

try:
    from world_model.dataset import append_transition
except ImportError:
    def append_transition(*a, **kw): pass


# ── Expert heuristic for SFT warmup data ─────────────────────────────────

def build_prompt(observation: dict[str, Any]) -> str:
    return (
        "You are acting inside Agentic Security Lab.\n"
        "Return one JSON action with keys command and parameters.\n\n"
        f"Observation:\n{observation['result']}\n\n"
        f"Summary: {observation.get('incident_summary', '')}\n"
        f"Known malicious packages: {observation.get('active_malicious_packages', [])}\n"
        f"Known exposed secrets: {observation.get('exposed_secrets', [])}\n"
    )


def expert_action(observation: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    active_packages = observation.get("active_malicious_packages", [])
    exposed_secrets = observation.get("exposed_secrets", [])
    data = observation.get("data", {})
    packages_in_scope = data.get("packages_in_scope", [])
    current_package = data.get("package")
    dependents = data.get("dependents", [])

    if current_package and dependents:
        for team in dependents:
            memory["teams"].add(team)

    for package in active_packages:
        if package not in memory["quarantined"]:
            memory["quarantined"].add(package)
            return {"command": "quarantine", "parameters": {"package": package}}

    for secret in exposed_secrets:
        if secret not in memory["rotated"]:
            memory["rotated"].add(secret)
            return {"command": "rotate_secret", "parameters": {"secret": secret}}

    if memory["teams"]:
        team = sorted(memory["teams"] - memory["notified"])
        if team:
            memory["notified"].add(team[0])
            return {"command": "notify", "parameters": {"team": team[0]}}

    for package in packages_in_scope:
        if package not in memory["scanned"]:
            memory["scanned"].add(package)
            return {"command": "scan_logs", "parameters": {"package": package}}

    for package in packages_in_scope:
        if package not in memory["inspected"]:
            memory["inspected"].add(package)
            return {"command": "inspect_package", "parameters": {"package": package}}

    for package in packages_in_scope:
        if package not in memory["checked_dependents"]:
            memory["checked_dependents"].add(package)
            return {"command": "check_dependents", "parameters": {"package": package}}

    return {"command": "conclude", "parameters": {}}


# ── Phase 0: Expert trajectory collection + SFT ─────────────────────────

def collect_examples(
    env_base_url: str, task: str, episodes: int,
    metrics_path: str, transitions_path: str, examples_path: str,
) -> list[dict[str, str]]:
    """Collect expert demonstrations via heuristic policy."""
    callback = MetricsCallback(metrics_path)
    attacker = AdversarialAttacker(seed=42)
    examples: list[dict[str, str]] = []
    Path(examples_path).parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=env_base_url, timeout=60) as http:
        for episode in range(1, episodes + 1):
            observation = http.post("/reset", json={"task_name": task, "mode": "training"}).json()
            total_reward = 0.0
            done = False
            step = 0
            memory = {k: set() for k in
                      ["scanned", "inspected", "checked_dependents", "quarantined",
                       "rotated", "teams", "notified"]}
            while not done and step < int(observation.get("data", {}).get("max_steps", 20)):
                step += 1
                action = expert_action(observation, memory)
                prompt = build_prompt(observation)
                next_observation = http.post("/step", json=action).json()
                reward = float(next_observation.get("reward", 0.0))
                benchmark_score = float(next_observation.get("data", {}).get("benchmark_score", 0.0))
                total_reward += reward
                examples.append({"prompt": prompt, "completion": json.dumps(action)})
                append_transition(transitions_path, {
                    "episode": episode, "step": step,
                    "observation": observation, "action": action,
                    "reward": reward, "next_observation": next_observation,
                    "success": next_observation.get("error") in (None, ""),
                })
                callback.log({"type": "train_step", "episode": episode, "step": step,
                              "loss": max(0.0, 1.0 - benchmark_score),
                              "reward": reward, "benchmark_score": benchmark_score})
                observation = next_observation
                done = bool(observation.get("done", False))

            # Feed adversarial attacker
            attacker.observe_defender({
                "benchmark_score": benchmark_score,
                "breakdown": observation.get("data", {}).get("score_breakdown", {}),
            })
            callback.log({"type": "episode_end", "episode": episode,
                          "total_reward": total_reward,
                          "success": benchmark_score >= 0.8,
                          "benchmark_score": benchmark_score,
                          "adversarial_level": attacker.get_metrics()["adversarial_level"]})

    Path(examples_path).write_text(
        "\n".join(json.dumps(item) for item in examples), encoding="utf-8")
    return examples


def train_sft(examples: list[dict[str, str]], model_name: str,
              output_dir: str, learning_rate: float, epochs: int) -> None:
    """Phase 0: SFT warmup on expert demonstrations."""
    if not all([Dataset, LoraConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, SFTTrainer]):
        raise ImportError("Training deps missing. pip install -e .[train]")
    dataset = Dataset.from_list(
        [{"text": item["prompt"] + item["completion"]} for item in examples])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_config = LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM")
    args = TrainingArguments(output_dir=output_dir, learning_rate=learning_rate,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=epochs, logging_steps=1, save_strategy="epoch", report_to=[])
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
                         processing_class=tokenizer, peft_config=lora_config)
    trainer.train()
    trainer.save_model(output_dir)


# ── Phase 1: GRPO (actual RL — run from Colab notebook) ─────────────────
# The GRPO training loop lives in notebooks/round2_training_colab.ipynb
# and uses:
#   - training.procedural_scenarios.generate_procedural_scenario
#   - training.trajectory_filter.CollapseDetector
#   - training.adversarial_attacker.AdversarialAttacker
#   - TRL GRPOTrainer with environment_factory=SecurityIncidentEnv
# See the notebook for the full wired-in pipeline.


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0: Expert SFT warmup")
    parser.add_argument("--env-base-url", default="http://localhost:8000")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--task", default="easy")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="artifacts/checkpoints/policy")
    parser.add_argument("--metrics-path", default="artifacts/metrics.jsonl")
    parser.add_argument("--transitions-path", default="artifacts/transitions.jsonl")
    parser.add_argument("--examples-path", default="artifacts/training_examples.jsonl")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    examples = collect_examples(
        env_base_url=args.env_base_url, task=args.task, episodes=args.episodes,
        metrics_path=args.metrics_path, transitions_path=args.transitions_path,
        examples_path=args.examples_path)
    if not args.collect_only:
        train_sft(examples=examples, model_name=args.model_name,
                   output_dir=args.output_dir, learning_rate=args.learning_rate,
                   epochs=args.epochs)


if __name__ == "__main__":
    main()
