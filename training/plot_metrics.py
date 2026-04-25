"""Generate training plots from JSONL metrics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt


def load_rewards(path: str = "artifacts/metrics.jsonl") -> tuple[list[float], list[float]]:
    rewards: list[float] = []
    losses: list[float] = []
    file_path = Path(path)
    if not file_path.exists():
        return rewards, losses
    for line in file_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("type") == "episode_end":
            rewards.append(float(item.get("total_reward", 0.0)))
        elif item.get("type") == "train_step":
            losses.append(float(item.get("loss", 0.0)))
    return rewards, losses


def main() -> None:
    rewards, losses = load_rewards()
    out = Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(range(1, len(rewards) + 1), rewards, label="Episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "reward_curve.png")
    plt.close()

    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, label="Training loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "loss_curve.png")
    plt.close()

    plt.figure()
    baseline = [0.0] * len(rewards)
    trained = rewards
    x = range(1, len(rewards) + 1)
    plt.plot(x, baseline, label="Baseline reward")
    plt.plot(x, trained, label="Collected/trained reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Baseline vs Trained")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "baseline_vs_trained.png")
    plt.close()


if __name__ == "__main__":
    main()
