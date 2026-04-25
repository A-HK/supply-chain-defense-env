"""Replanning trigger logic."""

from .plan_memory import PlanMemory


class Replanner:
    def should_replan(self, last_reward: float, observation: dict, memory: PlanMemory) -> bool:
        if last_reward <= 0:
            memory.stalled_steps += 1
        else:
            memory.stalled_steps = 0

        if observation.get("uncertainty_score", 0.0) > 0.7 and memory.stalled_steps >= 2:
            memory.replans += 1
            return True
        if memory.stalled_steps >= 4:
            memory.replans += 1
            return True
        return False
