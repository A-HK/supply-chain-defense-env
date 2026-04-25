"""Simple hierarchical planner."""

from .plan_memory import PlanMemory


class LongHorizonPlanner:
    def build_plan(self, observation: dict, memory: PlanMemory) -> list[dict]:
        if memory.action_queue:
            return memory.action_queue

        next_goal = memory.next_goal()
        active_pkgs = observation.get("active_malicious_packages", [])
        exposed = observation.get("exposed_secrets", [])
        data = observation.get("data", {})
        packages_in_scope = data.get("packages_in_scope", [])
        traced_package = data.get("package")
        evaluator_metrics = data.get("evaluator_metrics", {})
        known_fallbacks = evaluator_metrics.get("command_fallback_used_count", 0)

        if next_goal == "investigate":
            targets = active_pkgs or packages_in_scope[:3]
            actions = [{"command": "scan_logs", "parameters": {"package": pkg}} for pkg in targets]
            if not actions:
                memory.mark_completed("investigate")
                return self.build_plan(observation, memory)
            return actions

        if next_goal == "trace_root_cause":
            targets = []
            if traced_package:
                targets.append({"command": "check_dependents", "parameters": {"package": traced_package}})
            targets.extend(
                {"command": "inspect_package", "parameters": {"package": pkg}}
                for pkg in packages_in_scope
                if pkg not in active_pkgs
            )
            if not targets:
                memory.mark_completed("trace_root_cause")
                return self.build_plan(observation, memory)
            return targets[:3]

        if next_goal == "contain":
            actions = [{"command": "quarantine", "parameters": {"package": pkg}} for pkg in active_pkgs]
            if not actions:
                memory.mark_completed("contain")
                return self.build_plan(observation, memory)
            return actions

        if next_goal == "recover":
            actions = [{"command": "rotate_secret", "parameters": {"secret": secret}} for secret in exposed[:4]]
            if not actions:
                memory.mark_completed("recover")
                return self.build_plan(observation, memory)
            return actions

        if next_goal == "notify":
            score_breakdown = data.get("score_breakdown", {})
            if score_breakdown.get("notify_ratio", 0.0) >= 1.0:
                memory.mark_completed("notify")
                return self.build_plan(observation, memory)
            packages = active_pkgs or packages_in_scope
            notify_targets = []
            for package in packages:
                dependents = data.get("dependents", []) if package == traced_package else []
                for team in dependents:
                    notify_targets.append({"command": "notify", "parameters": {"team": team}})
            if not notify_targets and known_fallbacks == 0 and traced_package:
                return [{"command": "check_dependents", "parameters": {"package": traced_package}}]
            if not notify_targets:
                memory.mark_completed("notify")
                return self.build_plan(observation, memory)
            return notify_targets[:5]

        return [{"command": "conclude", "parameters": {}}]
