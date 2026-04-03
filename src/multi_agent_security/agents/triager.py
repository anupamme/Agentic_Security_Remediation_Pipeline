import logging
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from multi_agent_security.agents.base import BaseAgent
from multi_agent_security.types import (
    AgentMessage,
    FixStrategy,
    TriageResult,
    Vulnerability,
    VulnSeverity,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a security vulnerability triager. Your job is to assess vulnerabilities found by a scanner and prioritize them for remediation.

For each vulnerability, determine:
1. **Severity** (critical / high / medium / low / info) — based on potential impact if exploited
2. **Exploitability score** (0.0 to 1.0) — how easy is it to exploit in practice? Consider: is user input involved? Is the code reachable from an external interface? Are there existing protections?
3. **Fix strategy** — one of: one_liner, refactor, dependency_bump, config_change, multi_file
4. **Estimated complexity** — low / medium / high
5. **Priority order** — which vulnerabilities should be fixed first?

Consider the repo context: framework used, whether tests exist, the overall codebase size.

If a vulnerability from the scanner seems like a false positive based on the broader context, set its severity to "info" and note this in your reasoning.

Respond with JSON only, matching the provided schema.
"""


# ---------------------------------------------------------------------------
# Input / output models
# ---------------------------------------------------------------------------

class RepoMetadata(BaseModel):
    repo_name: str
    language: str
    framework: Optional[str] = None  # "django", "flask", "express", "spring", etc.
    has_tests: bool
    file_count: int
    dependency_files: list[str]  # e.g., ["requirements.txt", "package.json"]


class TriagerInput(BaseModel):
    vulnerabilities: list[Vulnerability]
    repo_metadata: RepoMetadata


class TriagerOutput(BaseModel):
    triage_results: list[TriageResult]  # one per input vulnerability
    priority_order: list[str]           # vuln IDs in recommended fix order
    summary: str


# ---------------------------------------------------------------------------
# Intermediate LLM response models
# ---------------------------------------------------------------------------

class _LLMTriageItem(BaseModel):
    vuln_id: str
    severity: VulnSeverity
    exploitability_score: float = Field(ge=0.0, le=1.0)
    fix_strategy: FixStrategy
    estimated_complexity: str  # "low" | "medium" | "high"
    triage_reasoning: str


class _LLMTriageResult(BaseModel):
    triage_results: list[_LLMTriageItem]
    priority_order: list[str]
    summary: str


# ---------------------------------------------------------------------------
# Default values for missing/invalid triage entries
# ---------------------------------------------------------------------------

_DEFAULT_SEVERITY = VulnSeverity.MEDIUM
_DEFAULT_EXPLOITABILITY = 0.5
_DEFAULT_STRATEGY = FixStrategy.REFACTOR
_DEFAULT_COMPLEXITY = "medium"
_DEFAULT_REASONING = "Default: LLM did not return a result for this vulnerability."


def _clamp_exploitability(score: float, vuln_id: str) -> float:
    if score < 0.0 or score > 1.0:
        clamped = max(0.0, min(1.0, score))
        logger.warning(
            "Exploitability score %.3f for %s is out of [0, 1]; clamping to %.3f",
            score, vuln_id, clamped,
        )
        return clamped
    return score


# ---------------------------------------------------------------------------
# Triager Agent
# ---------------------------------------------------------------------------

class TriagerAgent(BaseAgent):
    name = "triager"

    async def run(
        self, input_data: BaseModel, context: list[AgentMessage]
    ) -> tuple[TriagerOutput, AgentMessage]:
        assert isinstance(input_data, TriagerInput)
        inp: TriagerInput = input_data

        system_prompt = _SYSTEM_PROMPT
        user_prompt = self._build_user_prompt(inp)

        response = await self.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=_LLMTriageResult,
            agent_name=type(self).name,
        )

        metrics = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
        }

        try:
            llm_result = _LLMTriageResult.model_validate_json(response.content)
        except Exception as exc:
            logger.warning("Failed to parse LLM triage result: %s", exc)
            llm_result = _LLMTriageResult(
                triage_results=[],
                priority_order=[],
                summary="Triage parse error; all vulnerabilities assigned defaults.",
            )

        # Build dict of LLM results by vuln_id
        llm_by_id: dict[str, _LLMTriageItem] = {
            item.vuln_id: item for item in llm_result.triage_results
        }

        expected_ids = [v.id for v in inp.vulnerabilities]
        expected_id_set = set(expected_ids)

        # Build final triage results, filling defaults for any missing IDs
        triage_results: list[TriageResult] = []
        for vuln_id in expected_ids:
            if vuln_id in llm_by_id:
                item = llm_by_id[vuln_id]
                score = _clamp_exploitability(item.exploitability_score, vuln_id)
                triage_results.append(TriageResult(
                    vuln_id=vuln_id,
                    severity=item.severity,
                    exploitability_score=score,
                    fix_strategy=item.fix_strategy,
                    estimated_complexity=item.estimated_complexity,
                    triage_reasoning=item.triage_reasoning,
                ))
            else:
                logger.warning(
                    "LLM did not return a triage result for %s; assigning defaults.", vuln_id
                )
                triage_results.append(TriageResult(
                    vuln_id=vuln_id,
                    severity=_DEFAULT_SEVERITY,
                    exploitability_score=_DEFAULT_EXPLOITABILITY,
                    fix_strategy=_DEFAULT_STRATEGY,
                    estimated_complexity=_DEFAULT_COMPLEXITY,
                    triage_reasoning=_DEFAULT_REASONING,
                ))

        # Validate and fix priority_order
        priority_order = self._fix_priority_order(
            llm_result.priority_order, expected_ids, expected_id_set
        )

        output = TriagerOutput(
            triage_results=triage_results,
            priority_order=priority_order,
            summary=llm_result.summary,
        )

        message = AgentMessage(
            agent_name=type(self).name,
            timestamp=datetime.now(timezone.utc),
            content=output.model_dump_json(),
            token_count_input=metrics["input_tokens"],
            token_count_output=metrics["output_tokens"],
            latency_ms=metrics["latency_ms"],
            cost_usd=metrics["cost_usd"],
        )

        return output, message

    def _build_user_prompt(self, inp: TriagerInput) -> str:
        meta = inp.repo_metadata
        parts: list[str] = [
            f"## Repository: {meta.repo_name}",
            f"## Language: {meta.language}",
            f"## Framework: {meta.framework or 'unknown'}",
            f"## Has Tests: {meta.has_tests}",
            f"## File Count: {meta.file_count}",
            "",
            "## Vulnerabilities Found by Scanner:",
        ]

        for vuln in inp.vulnerabilities:
            parts.append(
                f"\n### {vuln.id}: {vuln.vuln_type.value}\n"
                f"- File: {vuln.file_path}, Lines {vuln.line_start}-{vuln.line_end}\n"
                f"- Confidence: {vuln.confidence}\n"
                f"- Description: {vuln.description}\n"
                f"- Code:\n```\n{vuln.code_snippet}\n```\n"
                f"- Scanner reasoning: {vuln.scanner_reasoning}"
            )

        return "\n".join(parts)

    @staticmethod
    def _fix_priority_order(
        llm_order: list[str],
        expected_ids: list[str],
        expected_id_set: set[str],
    ) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for vid in llm_order:
            if vid not in expected_id_set:
                logger.warning(
                    "priority_order contains unknown vuln ID %r; removing.", vid
                )
                continue
            if vid in seen:
                continue
            seen.add(vid)
            cleaned.append(vid)

        # Append any missing IDs in original order
        for vid in expected_ids:
            if vid not in seen:
                logger.warning(
                    "priority_order missing %r; appending at end.", vid
                )
                cleaned.append(vid)

        return cleaned
