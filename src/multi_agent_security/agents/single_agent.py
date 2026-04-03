"""
Single-agent baseline: one LLM call performs all four pipeline roles.

This is the simplest possible approach — the baseline all multi-agent
configurations must beat to justify their complexity.
"""
import glob as _glob
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from multi_agent_security.agents.base import BaseAgent
from multi_agent_security.types import (
    AgentMessage,
    FixStrategy,
    Patch,
    ReviewResult,
    TriageResult,
    Vulnerability,
    VulnSeverity,
    VulnType,
)

logger = logging.getLogger(__name__)

# Same char budget as the scanner for consistency (~3 000 tokens)
_SMALL_BATCH_MAX_CHARS = 12_000

_LANG_EXTENSIONS: dict[str, list[str]] = {
    "python": ["*.py"],
    "javascript": ["*.js", "*.ts", "*.jsx", "*.tsx"],
    "go": ["*.go"],
    "java": ["*.java"],
}

_SYSTEM_PROMPT = """\
You are a security engineer performing a complete security audit. \
For the given code file(s), you must:

1. DETECT all security vulnerabilities (with CWE classification, location, and confidence)
2. TRIAGE each vulnerability (severity, exploitability, fix strategy)
3. GENERATE a patch for each vulnerability (unified diff format)
4. REVIEW your own patches (correctness, security, style scores)

Be thorough but precise. Minimize false positives.

Respond with JSON only, matching the provided schema.
"""


# ---------------------------------------------------------------------------
# Public input / output models
# ---------------------------------------------------------------------------

class SingleAgentInput(BaseModel):
    repo_path: str
    target_files: list[str] = Field(default_factory=list)  # empty = scan all
    language: str


class SingleAgentOutput(BaseModel):
    vulnerabilities: list[Vulnerability]
    triage_results: list[TriageResult]
    patches: list[Patch]
    reviews: list[ReviewResult]
    summary: str


# ---------------------------------------------------------------------------
# Intermediate LLM response models (no IDs — assigned post-parse)
# ---------------------------------------------------------------------------

class _RawVuln(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    vuln_type: VulnType
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    code_snippet: str
    scanner_reasoning: str

    @field_validator("vuln_type", mode="before")
    @classmethod
    def coerce_unknown_vuln_type(cls, v: object) -> object:
        known = {m.value for m in VulnType}
        if isinstance(v, str) and v not in known:
            logger.warning("Unrecognised vuln_type %r — mapping to OTHER", v)
            return VulnType.OTHER
        return v


class _RawTriage(BaseModel):
    vuln_index: int = Field(description="0-based index into the vulnerabilities list")
    severity: VulnSeverity
    exploitability_score: float = Field(ge=0.0, le=1.0)
    fix_strategy: FixStrategy
    estimated_complexity: str
    triage_reasoning: str


class _RawPatch(BaseModel):
    vuln_index: int = Field(description="0-based index into the vulnerabilities list")
    file_path: str
    original_code: str
    patched_code: str
    unified_diff: str
    patch_reasoning: str


class _RawReview(BaseModel):
    vuln_index: int = Field(description="0-based index into the vulnerabilities list")
    patch_accepted: bool
    correctness_score: float = Field(ge=0.0, le=1.0)
    security_score: float = Field(ge=0.0, le=1.0)
    style_score: float = Field(ge=0.0, le=1.0)
    review_reasoning: str
    revision_request: Optional[str] = None


class _SingleAgentLLMResponse(BaseModel):
    vulnerabilities: list[_RawVuln]
    triage_results: list[_RawTriage]
    patches: list[_RawPatch]
    reviews: list[_RawReview]
    summary: str


# ---------------------------------------------------------------------------
# SingleAgent
# ---------------------------------------------------------------------------

class SingleAgent(BaseAgent):
    """
    A single LLM call that performs all four roles at once.
    This is the simplest possible approach — the baseline all
    multi-agent configurations must beat to justify their complexity.
    """

    name = "single_agent"

    async def run(
        self, input_data: BaseModel, context: list[AgentMessage]
    ) -> tuple[SingleAgentOutput, AgentMessage]:
        assert isinstance(input_data, SingleAgentInput)
        inp: SingleAgentInput = input_data

        file_contexts = self._load_files(inp)
        batches = self._batch_files(file_contexts)

        all_vulns: list[Vulnerability] = []
        all_triage: list[TriageResult] = []
        all_patches: list[Patch] = []
        all_reviews: list[ReviewResult] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0
        total_cost_usd = 0.0
        summaries: list[str] = []

        for batch in batches:
            user_prompt = self._build_user_prompt(batch, inp.language)
            response = await self.llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_format=_SingleAgentLLMResponse,
                agent_name=type(self).name,
            )
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            total_latency_ms += response.latency_ms
            total_cost_usd += response.cost_usd

            try:
                parsed = _SingleAgentLLMResponse.model_validate_json(response.content)
            except Exception as exc:
                logger.warning("SingleAgent failed to parse batch response: %s", exc)
                continue

            # Assign sequential IDs offset by previously accumulated vulns
            id_offset = len(all_vulns)
            batch_vulns = [
                Vulnerability(
                    id=f"VULN-{id_offset + i + 1:03d}",
                    file_path=rv.file_path,
                    line_start=rv.line_start,
                    line_end=rv.line_end,
                    vuln_type=rv.vuln_type,
                    description=rv.description,
                    confidence=rv.confidence,
                    code_snippet=rv.code_snippet,
                    scanner_reasoning=rv.scanner_reasoning,
                )
                for i, rv in enumerate(parsed.vulnerabilities)
                if rv.confidence >= 0.5
            ]
            all_vulns.extend(batch_vulns)

            for rt in parsed.triage_results:
                idx = rt.vuln_index
                if 0 <= idx < len(batch_vulns):
                    all_triage.append(TriageResult(
                        vuln_id=batch_vulns[idx].id,
                        severity=rt.severity,
                        exploitability_score=rt.exploitability_score,
                        fix_strategy=rt.fix_strategy,
                        estimated_complexity=rt.estimated_complexity,
                        triage_reasoning=rt.triage_reasoning,
                    ))
                else:
                    logger.warning("SingleAgent: triage vuln_index %d out of range", idx)

            for rp in parsed.patches:
                idx = rp.vuln_index
                if 0 <= idx < len(batch_vulns):
                    all_patches.append(Patch(
                        vuln_id=batch_vulns[idx].id,
                        file_path=rp.file_path,
                        original_code=rp.original_code,
                        patched_code=rp.patched_code,
                        unified_diff=rp.unified_diff,
                        patch_reasoning=rp.patch_reasoning,
                    ))
                else:
                    logger.warning("SingleAgent: patch vuln_index %d out of range", idx)

            for rr in parsed.reviews:
                idx = rr.vuln_index
                if 0 <= idx < len(batch_vulns):
                    all_reviews.append(ReviewResult(
                        vuln_id=batch_vulns[idx].id,
                        patch_accepted=rr.patch_accepted,
                        correctness_score=rr.correctness_score,
                        security_score=rr.security_score,
                        style_score=rr.style_score,
                        review_reasoning=rr.review_reasoning,
                        revision_request=rr.revision_request,
                    ))
                else:
                    logger.warning("SingleAgent: review vuln_index %d out of range", idx)

            summaries.append(parsed.summary)

        output = SingleAgentOutput(
            vulnerabilities=all_vulns,
            triage_results=all_triage,
            patches=all_patches,
            reviews=all_reviews,
            summary=" | ".join(summaries) if summaries else "No vulnerabilities found.",
        )
        message = AgentMessage(
            agent_name=type(self).name,
            timestamp=datetime.now(timezone.utc),
            content=output.model_dump_json(),
            token_count_input=total_input_tokens,
            token_count_output=total_output_tokens,
            latency_ms=total_latency_ms,
            cost_usd=total_cost_usd,
        )
        return output, message

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_files(self, inp: SingleAgentInput) -> list[tuple[str, str]]:
        """Return list of (relative_path, content) pairs."""
        if inp.target_files:
            paths = [os.path.join(inp.repo_path, f) for f in inp.target_files]
        else:
            patterns = _LANG_EXTENSIONS.get(inp.language, ["*.*"])
            paths = []
            for pattern in patterns:
                paths.extend(
                    _glob.glob(os.path.join(inp.repo_path, "**", pattern), recursive=True)
                )

        contexts: list[tuple[str, str]] = []
        for path in paths:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                contexts.append((os.path.relpath(path, inp.repo_path), content))
            except OSError as exc:
                logger.warning("Could not read %s: %s", path, exc)
        return contexts

    def _batch_files(
        self, file_contexts: list[tuple[str, str]]
    ) -> list[list[tuple[str, str]]]:
        """Group (path, content) pairs into char-budget batches."""
        batches: list[list[tuple[str, str]]] = []
        current: list[tuple[str, str]] = []
        current_chars = 0
        for path, content in file_contexts:
            if current_chars + len(content) > _SMALL_BATCH_MAX_CHARS and current:
                batches.append(current)
                current = []
                current_chars = 0
            current.append((path, content))
            current_chars += len(content)
        if current:
            batches.append(current)
        return batches

    def _build_user_prompt(
        self, batch: list[tuple[str, str]], language: str
    ) -> str:
        parts = []
        for path, content in batch:
            parts.append(
                f"### File: {path}\n"
                f"```{language}\n{content}\n```"
            )
        return "Perform a complete security audit of this code.\n\n" + "\n\n".join(parts)
