"""Self-improvement loop: collect failures, retrain, rebuild world model, re-evaluate."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--python", default="python")
    parser.add_argument("--env-base-url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"]
    checkpoint_root = Path("artifacts/checkpoints")
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    for idx, task in enumerate(tasks[: args.rounds], start=1):
        run(
            [
                args.python,
                "training/train_grpo.py",
                "--env-base-url",
                args.env_base_url,
                "--task",
                task,
                "--episodes",
                str(10 + idx * 5),
                "--model-name",
                args.model_name,
                "--output-dir",
                str(checkpoint_root / f"round_{idx}_{task}"),
            ]
        )
        run([args.python, "world_model/train_world_model.py", "--transitions", "artifacts/transitions.jsonl"])
        summary = summarize_runs()
        print(f"[SELF_IMPROVE] round={idx} summary={summary}")
        if holdout_pass(summary):
            print("[SELF_IMPROVE] holdout checks passed; increasing challenge next round.")
        else:
            print("[SELF_IMPROVE] holdout checks failed; keep easier curriculum.")


if __name__ == "__main__":
    main()
