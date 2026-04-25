"""Real-world security API integrations for the environment.

Provides live lookups against free, no-auth-required security databases.
These become additional tools the agent can call during incidents,
creating a genuine information-gathering vs. time-pressure tradeoff.

APIs: OSV.dev, deps.dev, GitHub Advisory Database, npm Registry.
All free, no API key required.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)
_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json"})
_TIMEOUT = 8


def query_osv(package: str, ecosystem: str = "npm", version: str | None = None) -> dict[str, Any]:
    """Query OSV.dev for known vulnerabilities and malware entries."""
    payload: dict[str, Any] = {"package": {"name": package, "ecosystem": ecosystem}}
    if version:
        payload["version"] = version
    try:
        r = _SESSION.post("https://api.osv.dev/v1/query", json=payload, timeout=_TIMEOUT)
        r.raise_for_status()
        vulns = r.json().get("vulns", [])
    except Exception as exc:
        logger.warning("OSV query failed for %s: %s", package, exc)
        return {"total": 0, "malicious": [], "cves": [], "ghsa": [], "error": str(exc)}
    return {
        "total": len(vulns),
        "malicious": [v["id"] for v in vulns if v["id"].startswith("MAL-")],
        "cves": [v["id"] for v in vulns if v["id"].startswith("CVE-")],
        "ghsa": [v["id"] for v in vulns if v["id"].startswith("GHSA-")],
        "summaries": {v["id"]: v.get("summary", "")[:200] for v in vulns[:5]},
    }


def query_deps_dev(package: str, ecosystem: str = "npm", version: str | None = None) -> dict[str, Any]:
    """Query deps.dev for package metadata, deprecation, and advisory keys."""
    eco = ecosystem.lower()
    url = f"https://api.deps.dev/v3alpha/systems/{eco}/packages/{package}/versions/{version}" if version else f"https://api.deps.dev/v3alpha/systems/{eco}/packages/{package}"
    try:
        r = _SESSION.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        d = r.json()
    except Exception as exc:
        return {"error": str(exc)}
    if version:
        return {"published": d.get("publishedAt", "unknown"), "deprecated": d.get("isDeprecated", False),
                "advisories": [a.get("id", "") for a in d.get("advisoryKeys", [])],
                "licenses": d.get("licenses", [])}
    versions = d.get("versions", [])
    return {"version_count": len(versions),
            "latest": versions[-1].get("versionKey", {}).get("version") if versions else None}


def query_github_advisory(ghsa_id: str) -> dict[str, Any]:
    """Fetch full advisory details from GitHub Advisory Database."""
    try:
        r = _SESSION.get(f"https://api.github.com/advisories/{ghsa_id}",
                         headers={"Accept": "application/vnd.github+json"}, timeout=_TIMEOUT)
        r.raise_for_status()
        d = r.json()
    except Exception as exc:
        return {"error": str(exc)}
    return {"id": d.get("ghsa_id"), "severity": d.get("severity"),
            "cvss_score": d.get("cvss", {}).get("score"),
            "summary": d.get("summary", "")[:300], "type": d.get("type"),
            "packages": [{"name": v.get("package", {}).get("name"),
                          "ecosystem": v.get("package", {}).get("ecosystem"),
                          "vulnerable_range": v.get("vulnerable_version_range")}
                         for v in d.get("vulnerabilities", [])]}


def query_npm_integrity(package: str, version: str) -> dict[str, Any]:
    """Check npm package integrity: hash, SLSA signatures, deprecation."""
    try:
        r = _SESSION.get(f"https://registry.npmjs.org/{package}/{version}", timeout=_TIMEOUT)
        r.raise_for_status()
        d = r.json()
    except Exception as exc:
        return {"error": str(exc)}
    dist = d.get("dist", {})
    return {"integrity": dist.get("integrity"), "shasum": dist.get("shasum"),
            "has_slsa_signature": "signatures" in dist, "deprecated": d.get("deprecated")}


def comprehensive_package_check(package: str, ecosystem: str = "npm", version: str | None = None) -> dict[str, Any]:
    """Run all checks and compute risk score."""
    result: dict[str, Any] = {"package": package, "ecosystem": ecosystem, "version": version}
    osv = query_osv(package, ecosystem, version)
    result["osv"] = osv
    result["deps_dev"] = query_deps_dev(package, ecosystem, version)
    result["npm_integrity"] = query_npm_integrity(package, version) if ecosystem.lower() == "npm" and version else None
    ghsa_ids = osv.get("ghsa", [])
    result["ghsa_detail"] = query_github_advisory(ghsa_ids[0]) if ghsa_ids else None
    risk = 0.0
    if osv.get("malicious"): risk = 1.0
    elif osv.get("total", 0) > 0: risk = min(0.8, 0.3 + 0.1 * osv["total"])
    if result.get("deps_dev", {}).get("deprecated"): risk = max(risk, 0.5)
    ghsa = result.get("ghsa_detail") or {}
    if ghsa.get("type") == "malware": risk = 1.0
    elif ghsa.get("cvss_score") and ghsa["cvss_score"] >= 9.0: risk = max(risk, 0.9)
    result["risk_score"] = round(risk, 2)
    result["verdict"] = "MALICIOUS" if risk >= 0.9 else "HIGH_RISK" if risk >= 0.6 else "SUSPICIOUS" if risk >= 0.3 else "LIKELY_CLEAN"
    return result
