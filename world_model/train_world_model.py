"""Train a lightweight world model from collected transitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .dataset import load_transitions
    from .model import LightweightWorldModel
except ImportError:
    from world_model.dataset import load_transitions
    from world_model.model import LightweightWorldModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", default="artifacts/transitions.jsonl")
    parser.add_argument("--output", default="artifacts/world_model.json")
    args = parser.parse_args()

    transitions = load_transitions(args.transitions)
    model = LightweightWorldModel.fit(transitions)
    model.save(args.output)
    print(f"Saved world model to {args.output} with {model.stats.transition_count} transitions.")


if __name__ == "__main__":
    main()
