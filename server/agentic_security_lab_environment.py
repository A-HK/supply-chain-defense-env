"""
Supply Chain Incident Response — Environment Logic
Replace the contents of:
    agentic_security_lab/server/agentic_security_lab_environment.py

Reward design (dense — signal every step, not just at episode end)
───────────────────────────────────────────────────────────────────
  quarantine correct package          +0.15
  rotate critical secret              +0.12
  rotate non-critical secret          +0.06
  notify affected team                +0.04
  scan_logs  (returns intel)          +0.02
  inspect_package                     +0.01
  check_dependents                    +0.01
  quarantine clean package (FP)       −0.05
  rotate already-rotated secret       −0.02
  invalid / unknown command           −0.01

Episode-end bonuses (awarded on conclude or max_steps):
  all required packages quarantined   +0.10
  all required secrets rotated        +0.10
  all required teams notified         +0.05
  concluded before exfil deadline     +0.10
  attacker succeeded                  −0.20
"""
import uuid
from typing import Any

from models import (
    AgenticSecurityLabAction,
    AgenticSecurityLabObservation,
    AgenticSecurityLabState,
)
from scenarios import get_scenario

VALID_COMMANDS = {
    "inspect_package",
    "check_dependents",
    "rotate_secret",
    "quarantine",
    "notify",
    "scan_logs",
    "conclude",
}


class AgenticSecurityLabEnvironment:
    """Supply chain incident response RL environment."""

    def __init__(self, task_name: str = "easy"):
        self._task_name = task_name
        self._state     = AgenticSecurityLabState()
        self._scenario: dict[str, Any] = {}

    def reset(self, task_name: str | None = None) -> AgenticSecurityLabObservation:
        if task_name:
            self._task_name = task_name

        sc = get_scenario(self._task_name)
        self._scenario = sc

        self._state = AgenticSecurityLabState(
            episode_id        = str(uuid.uuid4()),
            step_count        = 0,
            task_name         = self._task_name,
            packages          = sc["packages"],
            dependents        = sc["dependents"],
            secrets           = sc["secrets"],
            max_steps         = sc["max_steps"],
            exfiltration_step = sc["exfiltration_step"],
        )

        malicious = [p for p, d in sc["packages"].items() if d["malicious"]]
        exposed   = list(sc["secrets"].keys())

        return AgenticSecurityLabObservation(
            success = True,
            done    = False,
            reward  = 0.0,
            result  = (
                f"[INCIDENT ALERT]  Task: {self._task_name.upper()}\n"
                f"{sc['description']}\n\n"
                f"Packages in scope : {list(sc['packages'].keys())}\n"
                f"Exfiltration in   : {sc['exfiltration_step']} steps "
                f"(budget: {sc['max_steps']} steps total)\n\n"
                f"Commands available: inspect_package, check_dependents, "
                f"rotate_secret, quarantine, notify, scan_logs, conclude"
            ),
            data                     = {"packages_in_scope": list(sc["packages"].keys())},
            incident_summary         = f"Incident open. {len(malicious)} malicious package(s) suspected.",
            steps_remaining          = sc["exfiltration_step"],
            exposed_secrets          = exposed,
            active_malicious_packages = malicious,
        )

    def step(self, action: AgenticSecurityLabAction) -> AgenticSecurityLabObservation:
        s = self._state

        if s.incident_contained or s.attacker_succeeded or s.step_count >= s.max_steps:
            return self._terminal_obs(0.0)

        s.step_count += 1
        steps_left = max(0, s.exfiltration_step - s.step_count)

        if s.step_count >= s.exfiltration_step and not s.attacker_succeeded:
            crit_exposed = [
                k for k, v in s.secrets.items()
                if not v["rotated"] and v["critical"]
            ]
            if crit_exposed:
                s.attacker_succeeded = True

        cmd    = action.command.strip().lower()
        params = action.parameters

        if cmd not in VALID_COMMANDS:
            reward = -0.01
            s.total_reward += reward
            return self._build_obs(
                reward  = reward,
                success = False,
                result  = f"Unknown command '{cmd}'. Valid: {sorted(VALID_COMMANDS)}",
                steps_left = steps_left,
                error   = f"Invalid command: {cmd}",
            )

        dispatch = {
            "inspect_package":  self._cmd_inspect,
            "check_dependents": self._cmd_check_dependents,
            "rotate_secret":    self._cmd_rotate_secret,
            "quarantine":       self._cmd_quarantine,
            "notify":           self._cmd_notify,
            "scan_logs":        self._cmd_scan_logs,
            "conclude":         self._cmd_conclude,
        }
        return dispatch[cmd](params, steps_left)

    @property
    def state(self) -> AgenticSecurityLabState:
        return self._state

    def _cmd_inspect(self, params: dict, steps_left: int) -> AgenticSecurityLabObservation:
        pkg = params.get("package", "")
        if not pkg or pkg not in self._state.packages:
            return self._build_obs(-0.01, False,
                f"Package '{pkg}' not found. Known: {list(self._state.packages.keys())}",
                steps_left, error="Package not found")

        meta = self._state.packages[pkg]
        self._state.inspected.append(pkg)
        reward = 0.01
        self._state.total_reward += reward

        ioc_str = (
            f"⚠  IOCs detected: {meta['iocs']}"
            if meta.get("iocs")
            else "✓  No IOCs found."
        )
        return self._build_obs(
            reward  = reward,
            success = True,
            result  = (
                f"Inspection — {pkg}\n"
                f"  Publisher : {meta.get('publisher', 'unknown')}\n"
                f"  Published : {meta.get('publish_date', 'unknown')}\n"
                f"  Versions  : {meta.get('versions', [])}\n"
                f"  {ioc_str}"
            ),
            data       = {"package": pkg, "metadata": meta},
            steps_left = steps_left,
        )

    def _cmd_check_dependents(self, params: dict, steps_left: int) -> AgenticSecurityLabObservation:
        pkg  = params.get("package", "")
        deps = self._state.dependents.get(pkg)
        if deps is None:
            return self._build_obs(-0.01, False,
                f"No dependency data for '{pkg}'.", steps_left,
                error="Package not in dep graph")

        reward = 0.01
        self._state.total_reward += reward
        return self._build_obs(
            reward  = reward,
            success = True,
            result  = f"Dependents of {pkg}  ({len(deps)} total): {deps}",
            data    = {"package": pkg, "dependents": deps},
            steps_left = steps_left,
        )

    def _cmd_rotate_secret(self, params: dict, steps_left: int) -> AgenticSecurityLabObservation:
        secret = params.get("secret", "")
        s = self._state

        if secret not in s.secrets:
            return self._build_obs(-0.01, False,
                f"Secret '{secret}' unknown. Known: {list(s.secrets.keys())}",
                steps_left, error="Secret not found")

        if s.secrets[secret]["rotated"]:
            reward = -0.02
            s.total_reward += reward
            return self._build_obs(reward, False,
                f"Secret '{secret}' was already rotated.", steps_left)

        s.secrets[secret]["rotated"] = True
        s.rotated_secrets.append(secret)
        is_critical = s.secrets[secret]["critical"]
        reward = 0.12 if is_critical else 0.06
        s.total_reward += reward

        return self._build_obs(
            reward  = reward,
            success = True,
            result  = (
                f"✓  Rotated '{secret}' "
                f"({'CRITICAL' if is_critical else 'standard'}).\n"
                f"   Owner: {s.secrets[secret]['owner']}. "
                f"   Old value invalidated, new value issued."
            ),
            data       = {"secret": secret, "critical": is_critical},
            steps_left = steps_left,
        )

    def _cmd_quarantine(self, params: dict, steps_left: int) -> AgenticSecurityLabObservation:
        pkg = params.get("package", "")
        s   = self._state

        if pkg not in s.packages:
            return self._build_obs(-0.01, False,
                f"Package '{pkg}' not in scope.", steps_left,
                error="Package not found")

        if pkg in s.quarantined:
            return self._build_obs(-0.02, False,
                f"'{pkg}' already quarantined.", steps_left)

        if not s.packages[pkg]["malicious"]:
            reward = -0.05
            s.total_reward += reward
            return self._build_obs(reward, False,
                f"⚠  False positive: '{pkg}' is NOT malicious. "
                f"Unnecessary quarantine harms users.", steps_left)

        s.quarantined.append(pkg)
        reward = 0.15
        s.total_reward += reward
        return self._build_obs(
            reward  = reward,
            success = True,
            result  = (
                f"✓  Yanked '{pkg}' from registry. "
                f"Download counter zeroed. Consumers will see 404 on next install."
            ),
            data       = {"package": pkg},
            steps_left = steps_left,
        )

    def _cmd_notify(self, params: dict, steps_left: int) -> AgenticSecurityLabObservation:
        team = params.get("team", "")
        s    = self._state

        all_teams: set[str] = set()
        for teams in s.dependents.values():
            all_teams.update(teams)

        if team not in all_teams:
            return self._build_obs(-0.01, False,
                f"Team '{team}' not in affected list.", steps_left,
                error="Unknown team")

        if team in s.notified_teams:
            return self._build_obs(0.0, True,
                f"Team '{team}' already notified.", steps_left)

        s.notified_teams.append(team)
        reward = 0.04
        s.total_reward += reward
        return self._build_obs(
            reward  = reward,
            success = True,
            result  = f"✓  Notified '{team}': breach alert sent with remediation steps.",
            data    = {"team": team},
            steps_left = steps_left,
        )

    def _cmd_scan_logs(self, params: dict, steps_left: int) -> AgenticSecurityLabObservation:
        pkg   = params.get("package", "")
        hints = self._scenario.get("scan_logs_hints", {})

        if pkg not in self._state.packages:
            return self._build_obs(-0.01, False,
                f"Package '{pkg}' not in scope.", steps_left,
                error="Package not found")

        self._state.scanned_logs.append(pkg)
        log_result = hints.get(pkg, f"No suspicious entries found for {pkg}.")
        reward = 0.02
        self._state.total_reward += reward
        return self._build_obs(
            reward  = reward,
            success = True,
            result  = f"CI/CD log scan — {pkg}:\n  {log_result}",
            data    = {"package": pkg, "log_excerpt": log_result},
            steps_left = steps_left,
        )

    def _cmd_conclude(self, _params: dict, steps_left: int) -> AgenticSecurityLabObservation:
        s   = self._state
        req = self._scenario["required_actions"]

        q_done = set(s.quarantined)    >= set(req["quarantine"])
        r_done = set(s.rotated_secrets) >= set(req["rotate_secret"])
        n_done = set(s.notified_teams) >= set(req["notify"])

        bonus = 0.0
        if q_done: bonus += 0.10
        if r_done: bonus += 0.10
        if n_done: bonus += 0.05
        if not s.attacker_succeeded: bonus += 0.10
        if     s.attacker_succeeded: bonus -= 0.20

        s.total_reward      += bonus
        s.incident_contained = True

        missing_q = set(req["quarantine"])    - set(s.quarantined)
        missing_r = set(req["rotate_secret"]) - set(s.rotated_secrets)
        missing_n = set(req["notify"])        - set(s.notified_teams)

        result = (
            f"{'✓' if q_done else '✗'}  Packages quarantined : "
            f"{len(s.quarantined)}/{len(req['quarantine'])}"
            + (f"  — missing: {missing_q}" if missing_q else "") + "\n"
            f"{'✓' if r_done else '✗'}  Secrets rotated      : "
            f"{len(s.rotated_secrets)}/{len(req['rotate_secret'])}"
            + (f"  — missing: {missing_r}" if missing_r else "") + "\n"
            f"{'✓' if n_done else '✗'}  Teams notified       : "
            f"{len(s.notified_teams)}/{len(req['notify'])}"
            + (f"  — missing: {missing_n}" if missing_n else "") + "\n"
            f"{'✓  Contained before exfiltration' if not s.attacker_succeeded else '✗  Attacker exfiltrated credentials'}\n"
            f"Total episode reward : {s.total_reward:.3f}"
        )
        return self._terminal_obs(bonus, result=result)

    def _build_obs(
        self,
        reward:     float,
        success:    bool,
        result:     str,
        steps_left: int,
        data:       dict | None = None,
        error:      str  | None = None,
    ) -> AgenticSecurityLabObservation:
        s = self._state
        malicious_left = [
            p for p, d in s.packages.items()
            if d["malicious"] and p not in s.quarantined
        ]
        exposed_left = [k for k, v in s.secrets.items() if not v["rotated"]]
        done = s.incident_contained or s.attacker_succeeded or s.step_count >= s.max_steps

        return AgenticSecurityLabObservation(
            success  = success,
            done     = done,
            reward   = reward,
            result   = result,
            data     = data or {},
            incident_summary = (
                f"Step {s.step_count}/{s.max_steps}  |  "
                f"Quarantined: {len(s.quarantined)}  |  "
                f"Rotated: {len(s.rotated_secrets)}  |  "
                f"Notified: {len(s.notified_teams)}  |  "
                f"Attacker: {'⚠ ACTIVE' if s.attacker_succeeded else 'not yet'}"
            ),
            steps_remaining           = steps_left,
            exposed_secrets           = exposed_left,
            active_malicious_packages = malicious_left,
            error = error,
        )

    def _terminal_obs(self, reward: float, result: str = "") -> AgenticSecurityLabObservation:
        s = self._state
        s.incident_contained = True
        return AgenticSecurityLabObservation(
            success  = True,
            done     = True,
            reward   = reward,
            result   = result or "Episode ended.",
            incident_summary = (
                f"CLOSED  |  Steps: {s.step_count}  |  "
                f"Total reward: {s.total_reward:.3f}"
            ),
            steps_remaining           = 0,
            exposed_secrets           = [k for k, v in s.secrets.items() if not v["rotated"]],
            active_malicious_packages = [],
        )