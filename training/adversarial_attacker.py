"""Adversarial attacker role for self-play training.

Implements a co-evolving attacker that adapts package obfuscation,
exfiltration timing, and decoy placement based on defender behavior.

Reference: SPIRAL (arXiv 2506.24119) - Role-Conditioned Advantage Estimation
"""

from __future__ import annotations

import copy
import random
from typing import Any


class AdversarialAttacker:
    """Generates adaptive scenario modifications to challenge the defender.

    Observes defender success patterns and generates harder scenarios by:
    1. Adding decoy packages (increase false positive pressure)
    2. Hiding IOCs deeper (require more investigation steps)
    3. Tightening exfiltration deadlines
    4. Creating dependency chains that obscure root cause
    5. Exploiting defender's measured blind spots
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._defender_history: list[dict[str, Any]] = []
        self._difficulty_level: float = 1.0
        self._adaptation_rate: float = 0.1

    def observe_defender(self, episode_result: dict[str, Any]) -> None:
        """Record defender episode for adaptation."""
        self._defender_history.append(episode_result)
        if len(self._defender_history) > 50:
            self._defender_history = self._defender_history[-50:]
        recent = self._defender_history[-10:]
        if len(recent) >= 5:
            avg = sum(r.get("benchmark_score", 0) for r in recent) / len(recent)
            if avg > 0.7:
                self._difficulty_level = min(2.0, self._difficulty_level + self._adaptation_rate)
            elif avg < 0.3:
                self._difficulty_level = max(0.5, self._difficulty_level - self._adaptation_rate)

    def modify_scenario(self, base_scenario: dict[str, Any]) -> dict[str, Any]:
        """Apply adversarial modifications to make scenario harder."""
        sc = copy.deepcopy(base_scenario)
        level = self._difficulty_level

        if level > 1.0 and self._rng.random() < 0.6:
            sc = self._add_decoys(sc)
        if level > 0.8:
            sc = self._obscure_iocs(sc, level)
        if level > 1.2:
            sc["exfiltration_step"] = max(4, sc["exfiltration_step"] - max(1, int((level-1.0)*3)))
        if level > 1.1 and self._rng.random() < 0.5:
            sc = self._add_dependency_chain(sc)
        if self._defender_history:
            sc = self._exploit_blind_spots(sc)

        sc["adversarial_level"] = round(level, 2)
        return sc

    def _add_decoys(self, sc):
        pkgs = sc["packages"]
        decoys = ["debug-helper@1.0.0", "test-utils@2.3.1", "build-tools@0.9.5",
                  "lint-config@3.1.0", "dev-server@1.5.2"]
        for name in self._rng.sample(decoys, min(2, len(decoys))):
            if name not in pkgs:
                pkgs[name] = {"compromised":False,"malicious":False,
                    "publish_date":f"2026-0{self._rng.randint(1,3)}-{self._rng.randint(10,28)}",
                    "publisher":f"suspicious-user-{self._rng.randint(1,99)}",
                    "iocs":[],"versions":[name.split("@")[1]],"is_decoy":True}
                sc.setdefault("scan_logs_hints",{})[name] = (
                    f"Minor anomaly: {name} has unusual postinstall. Likely benign telemetry.")
        return sc

    def _obscure_iocs(self, sc, level):
        hints = sc.get("scan_logs_hints", {})
        for pid, hint in hints.items():
            meta = sc["packages"].get(pid, {})
            if meta.get("malicious") and self._rng.random() < (level-0.8)*0.5:
                hints[pid] = f"Unusual network activity from {pid} build step. Needs deeper analysis."
        hidden = sc.get("hidden_iocs", [])
        for pid, meta in sc["packages"].items():
            if meta.get("malicious") and meta.get("iocs") and self._rng.random() < (level-0.8)*0.3:
                moved = meta["iocs"].pop(0)
                hidden.append(f"Deep scan of {pid}: {moved}")
                if not meta["iocs"]:
                    meta["iocs"] = ["Behavioral anomaly detected (pending analysis)"]
        sc["hidden_iocs"] = hidden
        return sc

    def _add_dependency_chain(self, sc):
        mal = [p for p, m in sc["packages"].items() if m.get("malicious")]
        leg = [p for p, m in sc["packages"].items() if not m.get("malicious")]
        if mal and leg:
            target = self._rng.choice(leg)
            dep = self._rng.choice(mal)
            sc["packages"][target].setdefault("deps", []).append(dep)
            sc.setdefault("scan_logs_hints",{})[target] = f"No IOCs in {target}. Check deps: {sc['packages'][target].get('deps',[])}"
        return sc

    def _exploit_blind_spots(self, sc):
        if len(self._defender_history) < 5:
            return sc
        recent = self._defender_history[-10:]
        avg_notify = sum(r.get("breakdown",{}).get("notify_ratio",0) for r in recent) / len(recent)
        if avg_notify < 0.3:
            deps = sc.get("dependents", {})
            extras = ["data-warehouse", "log-aggregator", "cdn-purger"]
            for pid in deps:
                deps[pid].extend(self._rng.sample(extras, min(2, len(extras))))
            all_teams = set()
            for ts in deps.values(): all_teams.update(ts)
            sc["required_actions"]["notify"] = list(all_teams)
        avg_rotate = sum(r.get("breakdown",{}).get("rotate_ratio",0) for r in recent) / len(recent)
        if avg_rotate < 0.4:
            secrets = sc.get("secrets", {})
            for sn, meta in [("DATADOG_API_KEY",{"rotated":False,"critical":True,"owner":"monitoring","description":"Critical: monitoring key"}),
                             ("VAULT_TOKEN",{"rotated":False,"critical":True,"owner":"secrets-mgr","description":"Critical: Vault root token"})]:
                if sn not in secrets:
                    secrets[sn] = meta
                    sc["required_actions"]["rotate_secret"].append(sn)
        return sc

    def get_metrics(self) -> dict[str, Any]:
        return {"adversarial_level": round(self._difficulty_level, 3),
                "episodes_observed": len(self._defender_history),
                "avg_defender_score": sum(r.get("benchmark_score",0) for r in self._defender_history[-10:])
                    / max(1, len(self._defender_history[-10:])) if self._defender_history else 0.0}
