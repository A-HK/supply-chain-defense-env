"""
HuggingFace Jobs training script.

Run with:
    hf jobs uv run \
        --flavor t4-medium \
        --timeout 20m \
        --secrets HF_TOKEN \
        --with unsloth --with "trl>=0.16" --with "transformers>=4.44" \
        --with "datasets>=2.21" --with "accelerate>=0.34" --with "peft>=0.12" \
        --with bitsandbytes --with jmespath --with fastapi --with uvicorn \
        --with pydantic --with numpy --with requests \
        train_job.py

Override mode with --env TRAIN_MODE=FAST_TRAIN or --env TRAIN_MODE=FULL.
Requires HF_TOKEN secret (write access) to push the adapter to the Hub.
"""

import os, sys, gc, math, json, pathlib, subprocess
from collections import defaultdict
import numpy as np

# ── clone repo (HF Jobs only uploads this script, not the full repo) ─────────
SPACE_REPO = "https://huggingface.co/spaces/A-HK/agentic-security-lab"
REPO_ROOT  = pathlib.Path("/repo")
if not (REPO_ROOT / "models.py").exists():
    print(f"Cloning {SPACE_REPO} …")
    subprocess.run(["git", "clone", "--depth", "1", SPACE_REPO, str(REPO_ROOT)], check=True)
    print("Clone done.")
else:
    print("Repo already present.")

sys.path.insert(0, str(REPO_ROOT))
ARTIFACT_DIR = REPO_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# ── HuggingFace login ────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)

# ── hyperparameters ──────────────────────────────────────────────────────────
MODEL_NAME     = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
HUB_MODEL_ID   = "A-HK/security-incident-responder-grpo"
OUTPUT_DIR     = str(ARTIFACT_DIR / "grpo_checkpoint")

# ── mode selector ────────────────────────────────────────────────────────────
# FAST_TEST : smoke-test the Jobs pipeline, ~10 min on T4-medium ($0.07)
# FAST_TRAIN: real learning in <1 h on L4 (~$0.55)
# FULL      : best quality, ~2.5 h on L4 (~$2.00)
MODE = os.environ.get("TRAIN_MODE", "FAST_TEST")   # override with --env TRAIN_MODE=FULL

MAX_SEQ_LENGTH       = 4096
LORA_RANK            = 32
LORA_ALPHA           = 32
NUM_GENERATIONS      = 4
LEARNING_RATE        = 5e-6
KL_COEFF             = 0.04
MAX_COMPLETION_LENGTH = 192
GRAD_ACCUM           = 4
TEMPERATURE          = 0.8
NUM_TRAIN_EPOCHS     = 3
EPISODES_PER_DIFFICULTY = 40

if MODE == "FAST_TEST":
    MAX_SEQ_LENGTH = 768; LORA_RANK = 16; LORA_ALPHA = 16
    NUM_GENERATIONS = 2; MAX_COMPLETION_LENGTH = 64
    GRAD_ACCUM = 1; NUM_TRAIN_EPOCHS = 1; EPISODES_PER_DIFFICULTY = 4
elif MODE == "FAST_TRAIN":
    MAX_SEQ_LENGTH = 1536; LORA_RANK = 16; LORA_ALPHA = 16
    NUM_GENERATIONS = 2; MAX_COMPLETION_LENGTH = 96
    GRAD_ACCUM = 2; NUM_TRAIN_EPOCHS = 1; EPISODES_PER_DIFFICULTY = 25

EASY_ADVANCE         = 0.55
MEDIUM_ADVANCE       = 0.45
ROLLING_WINDOW       = 5
SAVE_STEPS           = 20 if MODE != "FAST_TEST" else 999999
LOGGING_STEPS        = 1 if MODE == "FULL" else 2
PUSH_TO_HUB          = bool(HF_TOKEN) and MODE != "FAST_TEST"

print(f"Model: {MODEL_NAME} | Hub: {HUB_MODEL_ID} | Mode: {MODE}")
print(f"epochs={NUM_TRAIN_EPOCHS} | eps/diff={EPISODES_PER_DIFFICULTY} | gens={NUM_GENERATIONS} "
      f"| max_comp={MAX_COMPLETION_LENGTH} | seq={MAX_SEQ_LENGTH} | lora_r={LORA_RANK}")

# ── imports ──────────────────────────────────────────────────────────────────
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback

from server.agentic_security_lab_environment import AgenticSecurityLabEnvironment
from models import AgenticSecurityLabAction
from training.adversarial_attacker import AdversarialAttacker
from training.callbacks import MetricsCallback
from training.trajectory_filter import CollapseDetector

# ── global curriculum state ───────────────────────────────────────────────────
CURRENT_DIFFICULTY = "easy"
ATTACKER = AdversarialAttacker(seed=42)

# ── SecurityIncidentEnv (TRL environment_factory wrapper) ────────────────────
# Some TRL/Unsloth versions call reward_funcs without the `environments` kwarg.
# The class-level registry lets reward functions recover env state in that case.
class SecurityIncidentEnv:
    _registry: list = []          # populated in __init__, drained by reward fns

    def __init__(self):
        self._env = AgenticSecurityLabEnvironment(CURRENT_DIFFICULTY)
        self.cumulative_reward = 0.0
        self.step_rewards = []; self.actions_taken = []
        self.done = False; self.steps_used = 0
        self.task_name = CURRENT_DIFFICULTY; self._last_sig = None
        SecurityIncidentEnv._registry.append(self)  # register for reward fn fallback

    def reset(self, **kwargs) -> str:
        self.cumulative_reward = 0.0; self.step_rewards = []; self.actions_taken = []
        self.done = False; self.steps_used = 0; self._last_sig = None
        task = kwargs.get("task_name", CURRENT_DIFFICULTY); self.task_name = task
        self._env = AgenticSecurityLabEnvironment(task)
        obs = self._env.reset(task_name=task, mode="training")
        return obs.result

    def _do_step(self, command, parameters):
        if self.done: raise ValueError("Episode ended.")
        obs = self._env.step(AgenticSecurityLabAction(command=command, parameters=parameters))
        reward = obs.reward
        sig = json.dumps({"c": command, "p": parameters}, sort_keys=True)
        if sig == self._last_sig: reward -= 0.1
        self._last_sig = sig
        self.cumulative_reward += reward; self.step_rewards.append(reward)
        self.actions_taken.append(command); self.steps_used += 1; self.done = obs.done
        lines = [obs.result, "", f"Status: {obs.incident_summary}",
                 f"Steps remaining: {obs.steps_remaining}"]
        if obs.exposed_secrets: lines.append(f"Secrets to rotate: {obs.exposed_secrets}")
        if obs.active_malicious_packages: lines.append(f"Active malicious: {obs.active_malicious_packages}")
        if obs.done: lines.append(f"Benchmark: {obs.data.get('benchmark_score', 0):.3f}")
        return "\n".join(lines)

    def inspect_package(self, package: str) -> str:
        """Inspect package metadata and IOC indicators.\n\nArgs:\n    package: Package name@version\nReturns:\n    Publisher, publish date, IOCs found."""
        return self._do_step("inspect_package", {"package": package})

    def check_dependents(self, package: str) -> str:
        """List downstream services depending on this package.\n\nArgs:\n    package: Package name@version\nReturns:\n    Affected downstream teams."""
        return self._do_step("check_dependents", {"package": package})

    def rotate_secret(self, secret: str) -> str:
        """Rotate a compromised credential.\n\nArgs:\n    secret: Secret name\nReturns:\n    Rotation confirmation."""
        return self._do_step("rotate_secret", {"secret": secret})

    def quarantine_package(self, package: str) -> str:
        """Quarantine a malicious package.\n\nArgs:\n    package: Package name@version\nReturns:\n    Quarantine result."""
        return self._do_step("quarantine", {"package": package})

    def notify_team(self, team: str) -> str:
        """Notify affected team of breach.\n\nArgs:\n    team: Team name\nReturns:\n    Notification confirmation."""
        return self._do_step("notify", {"team": team})

    def scan_logs(self, package: str) -> str:
        """Scan CI/CD logs for IOCs.\n\nArgs:\n    package: Package name@version\nReturns:\n    Suspicious log entries."""
        return self._do_step("scan_logs", {"package": package})

    def conclude_incident(self) -> str:
        """End episode, declare incident contained."""
        return self._do_step("conclude", {})


# ── reward functions ──────────────────────────────────────────────────────────
# Accept both calling conventions:
#   new TRL/Unsloth: reward_func(completions=..., prompts=...)  → environments=None
#   old TRL with environment_factory: reward_func(environments=envs, ...)
def _resolve_envs(completions, environments):
    """Return env list from kwarg if given, else drain the class registry."""
    if environments is not None:
        return environments
    n = len(completions) if completions is not None else len(SecurityIncidentEnv._registry)
    envs = SecurityIncidentEnv._registry[:n]
    del SecurityIncidentEnv._registry[:n]
    return envs

def environment_reward(completions=None, environments=None, **kwargs):
    envs = _resolve_envs(completions, environments)
    return [env.cumulative_reward * 2.0 for env in envs]

def efficiency_reward(completions=None, environments=None, **kwargs):
    envs = _resolve_envs(completions, environments)
    budgets = {"easy": 14, "medium": 18, "hard": 10}
    out = []
    for env in envs:
        budget = budgets.get(env.task_name, 14)
        step_ratio = env.steps_used / max(1, budget)
        shaped = 0.18 * (1.0 - step_ratio)
        if env.done and env.cumulative_reward > 0 and step_ratio <= 0.9:
            shaped += 0.05
        out.append(float(max(-0.2, min(0.2, shaped))))
    return out

def diversity_reward(completions=None, environments=None, **kwargs):
    envs = _resolve_envs(completions, environments)
    rewards = []
    for env in envs:
        actions = list(env.actions_taken)
        n = len(actions)
        if n < 3:
            rewards.append(0.0); continue
        counts = defaultdict(int)
        for a in actions: counts[a] += 1
        probs = [c / n for c in counts.values()]
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        max_entropy = math.log(max(2, len(counts)))
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
        max_frac = max(counts.values()) / n
        loop_penalty = min(0.2, (max_frac - 0.6) * 0.5) if max_frac > 0.6 else 0.0
        rewards.append(float(max(-0.2, min(0.2, 0.12 * entropy_norm - loop_penalty))))
    return rewards


# ── load model ───────────────────────────────────────────────────────────────
print("Loading model …")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,        # auto: bf16 on A100/L4, fp16 on T4
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)
print(f"Loaded {MODEL_NAME} with LoRA r={LORA_RANK}")

# ── dataset (all easy — curriculum advances the env during training) ──────────
SYSTEM_PROMPT = (
    "You are an expert security incident responder. Contain the supply-chain compromise "
    "before the attacker exfiltrates. Strategy: 1) scan_logs/inspect to find IOCs, "
    "2) quarantine malicious packages, 3) check_dependents for affected teams, "
    "4) rotate secrets (critical first), 5) notify all affected teams, 6) conclude when done."
)
USER_PROMPT = (
    "A supply-chain incident detected. Use available tools to investigate, contain, and "
    "remediate. The attacker is actively exfiltrating credentials."
)
train_dataset = Dataset.from_list([
    {"prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT}],
     "task_name": "easy"}
] * EPISODES_PER_DIFFICULTY)
print(f"Dataset: {len(train_dataset)} episodes")

# ── GRPOConfig ────────────────────────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
use_bf16 = any(x in gpu_name for x in ["a100", "a10", "h100", "l4", "l40"])
print(f"GPU: {gpu_name} | bf16={use_bf16}")

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LENGTH,
    temperature=TEMPERATURE,
    beta=KL_COEFF,
    scale_rewards=False,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    bf16=use_bf16,
    fp16=not use_bf16,
    logging_steps=LOGGING_STEPS,

    logging_first_step=True,
    log_completions=True,
    disable_tqdm=False,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    push_to_hub=PUSH_TO_HUB,
    hub_model_id=HUB_MODEL_ID,
    hub_strategy="every_save",
    warmup_ratio=0.1,
    reward_weights=[1.0, 0.3, 0.2],
    report_to=[],
)

# ── curriculum callback ───────────────────────────────────────────────────────
collapse_detector = CollapseDetector(window=15)

class CurriculumGRPOCallback(TrainerCallback):
    def __init__(self):
        self.metrics_cb = MetricsCallback(str(ARTIFACT_DIR / "grpo_metrics.jsonl"))
        self.episode_rewards = []; self.losses = []; self.step_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        global CURRENT_DIFFICULTY
        if not logs: return
        loss = logs.get("loss") or logs.get("train_loss")
        if loss is not None: self.losses.append(loss)
        reward = logs.get("reward")
        if reward is not None:
            self.step_count += 1
            self.episode_rewards.append(reward)
            self.metrics_cb.log({"type": "grpo_step", "step": self.step_count,
                                  "reward": reward, "loss": loss, "difficulty": CURRENT_DIFFICULTY})
            alert = collapse_detector.update(reward=reward)
            if alert: print(f"  WARNING: {alert}")
            ATTACKER.observe_defender({"benchmark_score": reward / 2.0,
                                       "breakdown": {"notify_ratio": 0.5, "rotate_ratio": 0.5}})
            if len(self.episode_rewards) >= ROLLING_WINDOW:
                avg = np.mean(self.episode_rewards[-ROLLING_WINDOW:])
                if CURRENT_DIFFICULTY == "easy" and avg > EASY_ADVANCE * 2:
                    CURRENT_DIFFICULTY = "medium"
                    print(f"\nCURRICULUM -> medium (avg={avg:.3f})")
                elif CURRENT_DIFFICULTY == "medium" and avg > MEDIUM_ADVANCE * 2:
                    CURRENT_DIFFICULTY = "hard"
                    print(f"\nCURRICULUM -> hard (avg={avg:.3f})")
            if self.step_count % 5 == 0:
                recent = self.episode_rewards[-5:]
                adv = ATTACKER.get_metrics()
                print(f"  step {self.step_count}: reward={reward:.3f} "
                      f"avg5={np.mean(recent):.3f} diff={CURRENT_DIFFICULTY} "
                      f"adv={adv['adversarial_level']:.2f}")


# ── train ─────────────────────────────────────────────────────────────────────
gc.collect(); torch.cuda.empty_cache()
FastLanguageModel.for_training(model)

curriculum_cb = CurriculumGRPOCallback()
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[environment_reward, efficiency_reward, diversity_reward],
    train_dataset=train_dataset,
    args=grpo_config,
    environment_factory=SecurityIncidentEnv,
    callbacks=[curriculum_cb],
)

print("=" * 60 + f"\nGRPO Training | {MODEL_NAME} | {HUB_MODEL_ID}\n" + "=" * 60)
train_result = trainer.train()
print(f"Done: {train_result.global_step} steps, loss={train_result.training_loss:.4f}")

# ── save ──────────────────────────────────────────────────────────────────────
if PUSH_TO_HUB:
    print(f"Pushing adapter to {HUB_MODEL_ID} …")
    model.push_to_hub_merged(HUB_MODEL_ID, tokenizer, save_method="lora", token=HF_TOKEN)
    print("Done — adapter uploaded.")
else:
    model.save_pretrained(str(ARTIFACT_DIR / "lora_adapter"))
    tokenizer.save_pretrained(str(ARTIFACT_DIR / "lora_adapter"))
    print(f"Saved locally to {ARTIFACT_DIR / 'lora_adapter'}")
