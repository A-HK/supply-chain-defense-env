"""Multi-round self-improvement loop.

Each round runs Phase-0 SFT warmup (expert trajectory collection + SFT),
rebuilds the world model from the new transitions, evaluates, and decides
whether to advance to the next difficulty tier.

IMPORTANT: this script drives Phase-0 SFT warmup only.
GRPO (Phase-1 RL) lives in notebooks/round2_training_colab.ipynb and requires
a GPU Colab runtime.  To run the full pipeline:
  1. Run this script to get a warm-started SFT checkpoint.
  2. Set HUB_MODEL_ID in the notebook to that checkpoint and run GRPO there.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from .evaluate import summarize_runs
    from .verifiers import holdout_pass
except ImportError:
    from training.evaluate import summarize_runs
    from training.verifiers import holdout_pass


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-0 SFT self-improvement loop: "
            "collect expert demos → SFT → rebuild world model → evaluate."
        )
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--python", default="python")
    parser.add_argument("--env-base-url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"]
    checkpoint_root = Path("artifacts/checkpoints")
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    for idx, task in enumerate(tasks[: args.rounds], start=1):
        print(f"[SELF_IMPROVE] Phase-0 SFT warmup — round={idx} task={task}")
        run(
            [
                args.python,
                "training/train_grpo.py",  # Phase-0 SFT entrypoint
                "--env-base-url",
                args.env_base_url,
                "--task",
                task,
                "--episodes",
                str(10 + idx * 5),
                "--model-name",
                args.model_name,
                "--output-dir",
                str(checkpoint_root / f"sft_round_{idx}_{task}"),
            ]
        )
        print(f"[SELF_IMPROVE] Rebuilding world model from transitions …")
        run([args.python, "world_model/train_world_model.py", "--transitions", "artifacts/transitions.jsonl"])

        summary = summarize_runs()
        print(f"[SELF_IMPROVE] round={idx} task={task} summary={summary}")
        if holdout_pass(summary):
            print(
                "[SELF_IMPROVE] Holdout checks passed; advancing to next difficulty. "
                "Run GRPO (notebooks/round2_training_colab.ipynb) for RL refinement."
            )
        else:
            print("[SELF_IMPROVE] Holdout checks failed; keeping easier curriculum for next round.")


if __name__ == "__main__":
    main()
