"""Plan memory for long-horizon episodes."""

from dataclasses import dataclass, field


@dataclass
class PlanMemory:
    goals: list[str] = field(
        default_factory=lambda: [
            "investigate",
            "trace_root_cause",
            "contain",
            "recover",
            "notify",
            "conclude",
        ]
    )
    completed: set[str] = field(default_factory=set)
    action_queue: list[dict] = field(default_factory=list)
    replans: int = 0
    stalled_steps: int = 0

    def mark_completed(self, goal: str) -> None:
        if goal in self.goals:
            self.completed.add(goal)

    def next_goal(self) -> str:
        for goal in self.goals:
            if goal not in self.completed:
                return goal
        return "conclude"
