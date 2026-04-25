"""Planner metrics helpers."""

from .plan_memory import PlanMemory


def compute_plan_metrics(memory: PlanMemory) -> dict:
    total_goals = max(1, len(memory.goals))
    return {
        "plan_goal_completion": len(memory.completed) / total_goals,
        "plan_replans": memory.replans,
        "plan_stalled_steps": memory.stalled_steps,
        "plan_pending_actions": len(memory.action_queue),
    }
