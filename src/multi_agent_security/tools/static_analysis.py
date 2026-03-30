import asyncio
import json
import logging
import os
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_SEMGREP_TIMEOUT = 30
_AUDIT_TIMEOUT = 60


class StaticAnalysisFinding(BaseModel):
    tool: str  # "semgrep" | "pip-audit" | "npm-audit"
    rule_id: str
    file_path: str
    line_start: int
    line_end: int
    message: str
    severity: str
    cwe: Optional[str] = None


async def run_semgrep(repo_path: str, language: str) -> list[StaticAnalysisFinding]:
    """Run semgrep with auto config on the repo and return parsed findings.

    Returns an empty list if semgrep is not installed, times out, or finds nothing.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "semgrep",
            "--config", "auto",
            "--json",
            "--lang", language,
            repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_SEMGREP_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.warning("semgrep timed out after %ds — skipping", _SEMGREP_TIMEOUT)
            return []
    except FileNotFoundError:
        logger.warning("semgrep not installed — skipping static analysis")
        return []
    except Exception as exc:
        logger.warning("semgrep failed: %s — skipping", exc)
        return []

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning("semgrep returned non-JSON output — skipping")
        return []

    findings: list[StaticAnalysisFinding] = []
    for result in data.get("results", []):
        extra = result.get("extra", {})
        metadata = extra.get("metadata", {})
        cwe_raw = metadata.get("cwe")
        if isinstance(cwe_raw, list):
            cwe = cwe_raw[0] if cwe_raw else None
        else:
            cwe = cwe_raw

        findings.append(StaticAnalysisFinding(
            tool="semgrep",
            rule_id=result.get("check_id", "unknown"),
            file_path=result.get("path", ""),
            line_start=result.get("start", {}).get("line", 0),
            line_end=result.get("end", {}).get("line", 0),
            message=extra.get("message", ""),
            severity=extra.get("severity", "WARNING"),
            cwe=cwe,
        ))

    return findings


async def run_dependency_audit(repo_path: str, language: str) -> list[StaticAnalysisFinding]:
    """Run language-specific dependency audit and return parsed findings.

    Supports Python (pip-audit) and JavaScript (npm audit).
    Returns an empty list for unsupported languages or if the tool is unavailable.
    """
    if language == "python":
        return await _run_pip_audit(repo_path)
    elif language == "javascript":
        return await _run_npm_audit(repo_path)
    else:
        # TODO: add govulncheck (Go) and Maven/Gradle audit (Java)
        return []


async def _run_pip_audit(repo_path: str) -> list[StaticAnalysisFinding]:
    req_file = os.path.join(repo_path, "requirements.txt")
    cmd = ["pip-audit", "--format=json"]
    if os.path.exists(req_file):
        cmd += ["-r", req_file]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_AUDIT_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.warning("pip-audit timed out — skipping")
            return []
    except FileNotFoundError:
        logger.warning("pip-audit not installed — skipping dependency audit")
        return []
    except Exception as exc:
        logger.warning("pip-audit failed: %s — skipping", exc)
        return []

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return []

    findings: list[StaticAnalysisFinding] = []
    for dep in data.get("dependencies", []):
        for vuln in dep.get("vulns", []):
            findings.append(StaticAnalysisFinding(
                tool="pip-audit",
                rule_id=vuln.get("id", "unknown"),
                file_path="requirements.txt",
                line_start=0,
                line_end=0,
                message=f"{dep.get('name', '')} {dep.get('version', '')}: {vuln.get('description', '')}",
                severity="HIGH",
                cwe=None,
            ))

    return findings


async def _run_npm_audit(repo_path: str) -> list[StaticAnalysisFinding]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "npm", "audit", "--json",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_AUDIT_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.warning("npm audit timed out — skipping")
            return []
    except FileNotFoundError:
        logger.warning("npm not installed — skipping dependency audit")
        return []
    except Exception as exc:
        logger.warning("npm audit failed: %s — skipping", exc)
        return []

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return []

    findings: list[StaticAnalysisFinding] = []
    # npm audit v7+ returns {"vulnerabilities": {...}}
    for name, info in data.get("vulnerabilities", {}).items():
        via = info.get("via", [])
        for source in via:
            if not isinstance(source, dict):
                continue
            findings.append(StaticAnalysisFinding(
                tool="npm-audit",
                rule_id=str(source.get("source", "unknown")),
                file_path="package.json",
                line_start=0,
                line_end=0,
                message=f"{name}: {source.get('title', '')}",
                severity=source.get("severity", "moderate").upper(),
                cwe=source.get("cwe", [None])[0] if source.get("cwe") else None,
            ))

    return findings
