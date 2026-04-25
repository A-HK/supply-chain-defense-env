"""Imagined rollouts backed by the world model."""

from __future__ import annotations

from typing import Any

from .model import LightweightWorldModel


def imagine_rollout(
    world_model: LightweightWorldModel,
    observation: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    imagined: list[dict[str, Any]] = []
    for action in candidate_actions:
        pred = world_model.predict(observation, action)
        imagined.append({"action": action, **pred})
    return imagined


def choose_best_action(
    world_model: LightweightWorldModel,
    candidate_actions: list[dict[str, Any]],
    observation: dict[str, Any],
) -> dict[str, Any] | None:
    imagined = imagine_rollout(world_model, observation, candidate_actions)
    if not imagined:
        return None
    imagined.sort(
        key=lambda item: (
            item.get("predicted_value", 0.0),
            item.get("confidence", 0.0),
        ),
        reverse=True,
    )
    return imagined[0]["action"]
