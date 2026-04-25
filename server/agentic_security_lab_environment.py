"""Environment logic for long-horizon, partially observable incident response."""

from __future__ import annotations

import random
import uuid
from typing import Any

from models import (
    AgenticSecurityLabAction,
    AgenticSecurityLabObservation,
    AgenticSecurityLabState,
)
from scenarios import generate_scenario, get_scenario

# Attempt to use the procedural generator for richer training-mode variation.
try:
    from training.procedural_scenarios import generate_procedural_scenario as _gen_proc
    _HAS_PROCEDURAL = True
except ImportError:
    _HAS_PROCEDURAL = False

# Inherit from OpenEnv's base Environment class if available; fall back to
# plain object so the file works in environments where openenv-core is absent.
try:
    from openenv.core import Environment as _OpenEnvBase  # type: ignore[import]
except ImportError:
    _OpenEnvBase = object  # type: ignore[misc,assignment]


def _score_from_breakdown(breakdown: dict[str, float]) -> float:
    """Shared formula used by both in-episode scoring and graders."""
    return round(
        max(
            0.0,
            min(
                1.0,
                0.35 * breakdown["quarantine_ratio"]
                + 0.35 * breakdown["rotate_ratio"]
                + 0.20 * breakdown["notify_ratio"]
                + 0.10 * breakdown["contain_ratio"],
            ),
        ),
        6,
    )

VALID_COMMANDS = {
    "inspect_package",
    "check_dependents",
    "rotate_secret",
    "quarantine",
    "notify",
    "scan_logs",
    "conclude",
}

COMMAND_ALIASES = {
    "inspect": "inspect_package",
    "deps": "check_dependents",
    "dependents": "check_dependents",
    "rotate": "rotate_secret",
}

VALID_MODES = {"benchmark", "training"}


class AgenticSecurityLabEnvironment(_OpenEnvBase):
    """Supply-chain incident response environment.

    Inherits from ``openenv.core.Environment`` when openenv-core is installed,
    giving the framework standard lifecycle hooks.  Falls back to plain
    ``object`` so the class works in lightweight / testing contexts too.
    """

    def __init__(self, task_name: str = "easy") -> None:
        self._task_name = task_name
        self._state = AgenticSecurityLabState()
        self._scenario: dict[str, Any] = {}
        self._rng = random.Random(0)

    def reset(
        self,
        task_name: str | None = None,
        mode: str | None = None,
        command_fallback_enabled: bool | None = None,
    ) -> AgenticSecurityLabObservation:
        if task_name:
            self._task_name = task_name

        requested_mode = (mode or "benchmark").strip().lower()
        resolved_mode = requested_mode if requested_mode in VALID_MODES else "benchmark"
        mode_fallback_used = resolved_mode != requested_mode

        if command_fallback_enabled is None:
            command_fallback_enabled = False

        if resolved_mode == "benchmark":
            scenario = get_scenario(self._task_name)
            self._rng = random.Random(self._task_name)
        else:
            # Prefer fully procedural generation (infinite unique incidents) when
            # available; fall back to jitter-only variation for minimal installs.
            if _HAS_PROCEDURAL:
                scenario = _gen_proc(difficulty=self._task_name)
            else:
                scenario = generate_scenario(self._task_name, difficulty_scale=1.0)
            self._rng = random.Random()
        self._scenario = scenario

        self._state = AgenticSecurityLabState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=self._task_name,
            mode=resolved_mode,
            mode_fallback_used=mode_fallback_used,
            command_fallback_enabled=bool(command_fallback_enabled),
            packages=scenario["packages"],
            dependents=scenario["dependents"],
            secrets=scenario["secrets"],
            max_steps=scenario["max_steps"],
            exfiltration_step=scenario["exfiltration_step"],
            pending_hidden_iocs=list(scenario.get("hidden_iocs", [])),
            plan_progress={
                "investigate": False,
                "trace_root_cause": False,
                "contain": False,
                "recover": False,
                "notify": False,
                "conclude": False,
            },
        )

        return self._build_obs(
            reward=0.0,
            success=True,
            result=(
                f"[INCIDENT ALERT] Task: {self._task_name.upper()}\n"
                f"{scenario['description']}\n\n"
                f"Packages in scope: {list(scenario['packages'].keys())}\n"
                f"Exfiltration window: {scenario['exfiltration_step']} steps "
                f"(budget: {scenario['max_steps']} total)\n"
                "Commands: inspect_package, check_dependents, rotate_secret, "
                "quarantine, notify, scan_logs, conclude"
            ),
            steps_left=scenario["exfiltration_step"],
            data={
                "packages_in_scope": list(scenario["packages"].keys()),
                "mode": resolved_mode,
                "command_fallback_enabled": bool(command_fallback_enabled),
                "max_steps": scenario["max_steps"],
            },
        )

    def step(self, action: AgenticSecurityLabAction) -> AgenticSecurityLabObservation:
        state = self._state
        if state.incident_contained or state.attacker_succeeded or state.step_count >= state.max_steps:
            return self._terminal_obs(0.0, result="Episode already ended.")

        state.step_count += 1
        steps_left = max(0, state.exfiltration_step - state.step_count)
        self._advance_attacker()

        command, fallback_used = self._canonicalize_command(action.command)
        params = action.parameters

        if fallback_used:
            state.command_fallback_used_count += 1

        if command not in VALID_COMMANDS:
            state.invalid_action_count += 1
            reward = -0.01
            state.total_reward += reward
            return self._build_obs(
                reward=reward,
                success=False,
                result=f"Unknown command '{action.command}'. Valid: {sorted(VALID_COMMANDS)}",
                steps_left=steps_left,
                error=f"Invalid command: {action.command}",
            )

        dispatch = {
            "inspect_package": self._cmd_inspect,
            "check_dependents": self._cmd_check_dependents,
            "rotate_secret": self._cmd_rotate_secret,
            "quarantine": self._cmd_quarantine,
            "notify": self._cmd_notify,
            "scan_logs": self._cmd_scan_logs,
            "conclude": self._cmd_conclude,
        }
        observation = dispatch[command](params, steps_left)
        self._log_transition(command, params, observation)
        return observation

    @property
    def state(self) -> AgenticSecurityLabState:
        return self._state

    def _canonicalize_command(self, raw_command: str) -> tuple[str, bool]:
        command = raw_command.strip().lower()
        if command in VALID_COMMANDS:
            return command, False
        if self._state.command_fallback_enabled and command in COMMAND_ALIASES:
            return COMMAND_ALIASES[command], True
        return command, False

    def _cmd_inspect(self, params: dict[str, Any], steps_left: int) -> AgenticSecurityLabObservation:
        package = params.get("package", "")
        if package not in self._state.packages:
            return self._invalid_target("Package", package, steps_left)

        metadata = self._state.packages[package]
        self._state.inspected.append(package)
        self._state.plan_progress["investigate"] = True
        if metadata.get("malicious"):
            self._remember_package(package)

        discovered_deps = metadata.get("deps", [])
        if discovered_deps:
            self._state.plan_progress["trace_root_cause"] = True

        reward = 0.01
        self._state.total_reward += reward
        return self._build_obs(
            reward=reward,
            success=True,
            result=(
                f"Inspection: {package}\n"
                f"Publisher: {metadata.get('publisher', 'unknown')}\n"
                f"Published: {metadata.get('publish_date', 'unknown')}\n"
                f"Versions: {metadata.get('versions', [])}\n"
                f"Dependencies: {discovered_deps or 'None'}\n"
                f"IOC summary: {metadata.get('iocs') or 'No direct IOC found.'}"
            ),
            steps_left=steps_left,
            data={"package": package, "metadata": metadata},
        )

    def _cmd_check_dependents(self, params: dict[str, Any], steps_left: int) -> AgenticSecurityLabObservation:
        package = params.get("package", "")
        dependents = self._state.dependents.get(package)
        if dependents is None:
            return self._invalid_target("Package", package, steps_left)

        self._state.traced_packages.append(package)
        self._state.plan_progress["trace_root_cause"] = True
        if self._state.packages.get(package, {}).get("malicious"):
            self._remember_package(package)

        reward = 0.01
        self._state.total_reward += reward
        return self._build_obs(
            reward=reward,
            success=True,
            result=f"Dependents of {package} ({len(dependents)} total): {dependents}",
            steps_left=steps_left,
            data={"package": package, "dependents": dependents},
        )

    def _cmd_rotate_secret(self, params: dict[str, Any], steps_left: int) -> AgenticSecurityLabObservation:
        secret = params.get("secret", "")
        if secret not in self._state.secrets:
            return self._invalid_target("Secret", secret, steps_left)

        secret_meta = self._state.secrets[secret]
        if secret_meta["rotated"]:
            reward = -0.02
            self._state.total_reward += reward
            return self._build_obs(
                reward=reward,
                success=False,
                result=f"Secret '{secret}' was already rotated.",
                steps_left=steps_left,
            )

        secret_meta["rotated"] = True
        self._state.rotated_secrets.append(secret)
        self._state.plan_progress["recover"] = True
        reward = 0.12 if secret_meta["critical"] else 0.06
        self._state.total_reward += reward
        return self._build_obs(
            reward=reward,
            success=True,
            result=(
                f"Rotated '{secret}' "
                f"({'CRITICAL' if secret_meta['critical'] else 'standard'}).\n"
                f"Owner: {secret_meta['owner']}. Old value invalidated."
            ),
            steps_left=steps_left,
            data={"secret": secret, "critical": secret_meta["critical"]},
        )

    def _cmd_quarantine(self, params: dict[str, Any], steps_left: int) -> AgenticSecurityLabObservation:
        package = params.get("package", "")
        if package not in self._state.packages:
            return self._invalid_target("Package", package, steps_left)
        if package in self._state.quarantined:
            return self._build_obs(
                reward=-0.02,
                success=False,
                result=f"'{package}' already quarantined.",
                steps_left=steps_left,
            )

        package_meta = self._state.packages[package]
        if not package_meta["malicious"]:
            reward = -0.05
            self._state.false_positive_count += 1
            self._state.total_reward += reward
            return self._build_obs(
                reward=reward,
                success=False,
                result=f"False positive: '{package}' is not malicious.",
                steps_left=steps_left,
            )

        self._state.quarantined.append(package)
        self._state.plan_progress["contain"] = True
        self._remember_package(package)
        reward = 0.15
        self._state.total_reward += reward
        return self._build_obs(
            reward=reward,
            success=True,
            result=f"Quarantined '{package}'. Registry blocks future installs.",
            steps_left=steps_left,
            data={"package": package},
        )

    def _cmd_notify(self, params: dict[str, Any], steps_left: int) -> AgenticSecurityLabObservation:
        team = params.get("team", "")
        valid_teams = set()
        for teams in self._state.dependents.values():
            valid_teams.update(teams)
        if team not in valid_teams:
            return self._invalid_target("Team", team, steps_left)
        if team in self._state.notified_teams:
            return self._build_obs(
                reward=0.0,
                success=True,
                result=f"Team '{team}' already notified.",
                steps_left=steps_left,
            )

        self._state.notified_teams.append(team)
        self._state.plan_progress["notify"] = True
        reward = 0.04
        self._state.total_reward += reward
        return self._build_obs(
            reward=reward,
            success=True,
            result=f"Notified '{team}' with incident guidance.",
            steps_left=steps_left,
            data={"team": team},
        )

    def _cmd_scan_logs(self, params: dict[str, Any], steps_left: int) -> AgenticSecurityLabObservation:
        package = params.get("package", "")
        if package not in self._state.packages:
            return self._invalid_target("Package", package, steps_left)

        self._state.scanned_logs.append(package)
        hints = self._scenario.get("scan_logs_hints", {})
        log_result = hints.get(package, f"No suspicious entries found for {package}.")
        if self._state.packages[package]["malicious"]:
            self._remember_package(package)
            self._discover_secrets_for_package(package)

        if (
            self._state.mode == "training"
            and self._state.pending_hidden_iocs
            and self._rng.random() < self._scenario.get("stochastic", {}).get("alert_reveal_chance", 0.35)
        ):
            hidden = self._state.pending_hidden_iocs.pop(0)
            self._state.discovered_iocs.append(hidden)
            log_result = f"{log_result}\nAdditional hidden signal: {hidden}"

        self._state.plan_progress["investigate"] = True
        reward = 0.02
        self._state.total_reward += reward
        return self._build_obs(
            reward=reward,
            success=True,
            result=f"CI/CD log scan - {package}:\n  {log_result}",
            steps_left=steps_left,
            data={"package": package, "log_excerpt": log_result},
        )

    def _cmd_conclude(self, _params: dict[str, Any], steps_left: int) -> AgenticSecurityLabObservation:
        ratios = self._score_breakdown()
        q_done = ratios["quarantine_ratio"] == 1.0
        r_done = ratios["rotate_ratio"] == 1.0
        n_done = ratios["notify_ratio"] == 1.0
        contained = ratios["contain_ratio"] == 1.0 and q_done and r_done

        bonus = 0.0
        if q_done:
            bonus += 0.10
        if r_done:
            bonus += 0.10
        if n_done:
            bonus += 0.05
        if contained:
            bonus += 0.10
        if self._state.attacker_succeeded:
            bonus -= 0.20
        if not (q_done or r_done or n_done):
            bonus -= 0.05

        self._state.plan_progress["conclude"] = True
        self._state.total_reward += bonus
        self._state.incident_contained = True

        req = self._scenario["required_actions"]
        missing_packages = sorted(set(req["quarantine"]) - set(self._state.quarantined))
        missing_secrets = sorted(set(req["rotate_secret"]) - set(self._state.rotated_secrets))
        missing_teams = sorted(set(req["notify"]) - set(self._state.notified_teams))

        result = (
            f"{'OK' if q_done else 'MISS'} Packages quarantined: "
            f"{len(self._state.quarantined)}/{len(req['quarantine'])}"
            + (f" - missing: {missing_packages}" if missing_packages else "")
            + "\n"
            + f"{'OK' if r_done else 'MISS'} Secrets rotated: "
            + f"{len(self._state.rotated_secrets)}/{len(req['rotate_secret'])}"
            + (f" - missing: {missing_secrets}" if missing_secrets else "")
            + "\n"
            + f"{'OK' if n_done else 'MISS'} Teams notified: "
            + f"{len(self._state.notified_teams)}/{len(req['notify'])}"
            + (f" - missing: {missing_teams}" if missing_teams else "")
            + "\n"
            + (
                "Contained before exfiltration"
                if contained
                else "Incident closed before full containment"
            )
        )
        return self._terminal_obs(bonus, result=result)

    def _invalid_target(self, label: str, value: str, steps_left: int) -> AgenticSecurityLabObservation:
        self._state.invalid_action_count += 1
        reward = -0.01
        self._state.total_reward += reward
        return self._build_obs(
            reward=reward,
            success=False,
            result=f"{label} '{value}' not found.",
            steps_left=steps_left,
            error=f"{label} not found",
        )

    def _remember_package(self, package: str) -> None:
        if package not in self._state.discovered_packages:
            self._state.discovered_packages.append(package)

    def _discover_secrets_for_package(self, package: str) -> None:
        affected_owners = set(self._state.dependents.get(package, []))
        for secret_name, secret_meta in self._state.secrets.items():
            if secret_meta["owner"] in affected_owners and secret_name not in self._state.discovered_secrets:
                self._state.discovered_secrets.append(secret_name)

    def _score_breakdown(self) -> dict[str, float]:
        required = self._scenario["required_actions"]
        quarantine_ratio = self._ratio(self._state.quarantined, required["quarantine"])
        rotate_ratio = self._ratio(self._state.rotated_secrets, required["rotate_secret"])
        notify_ratio = self._ratio(self._state.notified_teams, required["notify"])
        contain_ratio = 1.0 if (not self._state.attacker_succeeded and quarantine_ratio == 1.0) else 0.0
        return {
            "quarantine_ratio": quarantine_ratio,
            "rotate_ratio": rotate_ratio,
            "notify_ratio": notify_ratio,
            "contain_ratio": contain_ratio,
        }

    def _benchmark_score(self) -> float:
        return _score_from_breakdown(self._score_breakdown())

    @staticmethod
    def _ratio(actual: list[str], required: list[str]) -> float:
        if not required:
            return 1.0
        return len(set(actual) & set(required)) / len(set(required))

    def _build_obs(
        self,
        reward: float,
        success: bool,
        result: str,
        steps_left: int,
        data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> AgenticSecurityLabObservation:
        benchmark_score = self._benchmark_score()
        score_breakdown = self._score_breakdown()
        evaluator_metrics = {
            "invalid_actions": self._state.invalid_action_count,
            "false_positives": self._state.false_positive_count,
            "mode_fallback_used": self._state.mode_fallback_used,
            "mode": self._state.mode,
            "command_fallback_enabled": self._state.command_fallback_enabled,
            "command_fallback_used_count": self._state.command_fallback_used_count,
            "deadline_reached": self._state.step_count >= self._state.exfiltration_step,
            "attacker_succeeded": self._state.attacker_succeeded,
        }
        observation_data = {
            "reward_type": "training_step_reward",
            "benchmark_score": benchmark_score,
            "score_breakdown": score_breakdown,
            "evaluator_metrics": evaluator_metrics,
            "max_steps": self._state.max_steps,
            "packages_in_scope": list(self._state.packages.keys()),
        }
        if data:
            observation_data.update(data)

        done = (
            self._state.incident_contained
            or self._state.attacker_succeeded
            or self._state.step_count >= self._state.max_steps
        )
        uncertainty = self._uncertainty_score()

        return AgenticSecurityLabObservation(
            success=success,
            done=done,
            reward=reward,
            result=result,
            data=observation_data,
            incident_summary=(
                f"Step {self._state.step_count}/{self._state.max_steps} | "
                f"KnownPackages:{len(self._state.discovered_packages)} "
                f"KnownSecrets:{len(self._state.discovered_secrets)} "
                f"Q:{len(self._state.quarantined)} "
                f"R:{len(self._state.rotated_secrets)} "
                f"N:{len(self._state.notified_teams)} "
                f"Attacker:{self._state.attacker_progress:.2f}"
            ),
            steps_remaining=steps_left,
            exposed_secrets=[
                secret
                for secret in self._state.discovered_secrets
                if not self._state.secrets[secret]["rotated"]
            ],
            active_malicious_packages=[
                package
                for package in self._state.discovered_packages
                if package not in self._state.quarantined
            ],
            visible_alerts=self._state.discovered_iocs[-5:],
            uncertainty_score=uncertainty,
            plan_progress=dict(self._state.plan_progress),
            info={
                "attacker_progress": self._state.attacker_progress,
                "risk_events": self._state.risk_events[-3:],
                "discovered_packages": len(self._state.discovered_packages),
                "discovered_secrets": len(self._state.discovered_secrets),
            },
            error=error,
        )

    def _terminal_obs(self, reward: float, result: str = "") -> AgenticSecurityLabObservation:
        self._state.incident_contained = True
        return self._build_obs(
            reward=reward,
            success=True,
            result=result or "Episode ended.",
            steps_left=0,
        )

    def _uncertainty_score(self) -> float:
        total_malicious = sum(1 for meta in self._state.packages.values() if meta["malicious"])
        total_known = len(self._state.discovered_packages)
        unresolved = max(0, total_malicious - total_known)
        base = unresolved / max(1, total_malicious)
        hidden = len(self._state.pending_hidden_iocs) / max(1, len(self._scenario.get("hidden_iocs", [])))
        return round(max(0.0, min(1.0, 0.15 + 0.5 * base + 0.35 * hidden)), 4)

    def _advance_attacker(self) -> None:
        if self._state.attacker_succeeded:
            return

        if self._state.mode == "benchmark":
            delta = 1.0 / max(1, self._state.exfiltration_step)
        else:
            jitter = self._scenario.get("stochastic", {}).get("progress_jitter", 0.3)
            delta = max(
                0.05,
                (1.0 / max(1, self._state.exfiltration_step)) + self._rng.uniform(-jitter, jitter) * 0.05,
            )

        self._state.attacker_progress = min(1.0, self._state.attacker_progress + delta)
        if self._state.attacker_progress > 0.85 and not self._state.risk_events:
            self._state.risk_events.append("Attacker foothold appears to be deepening.")

        if self._state.step_count >= self._state.exfiltration_step:
            critical_remaining = [
                secret_name
                for secret_name, secret_meta in self._state.secrets.items()
                if secret_meta["critical"] and not secret_meta["rotated"]
            ]
            if critical_remaining:
                self._state.attacker_succeeded = True
                self._state.risk_events.append("Critical secrets exfiltrated.")

    def _log_transition(self, command: str, params: dict[str, Any], obs: AgenticSecurityLabObservation) -> None:
        self._state.trajectory_log.append(
            {
                "step": self._state.step_count,
                "mode": self._state.mode,
                "command": command,
                "params": params,
                "reward": obs.reward,
                "done": obs.done,
                "success": obs.success,
                "benchmark_score": obs.data.get("benchmark_score", 0.0),
                "attacker_progress": self._state.attacker_progress,
                "plan_progress": dict(self._state.plan_progress),
            }
        )
