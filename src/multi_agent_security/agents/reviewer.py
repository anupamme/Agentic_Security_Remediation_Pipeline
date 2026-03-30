import logging
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from multi_agent_security.agents.base import BaseAgent
from multi_agent_security.tools.test_runner import TestResult
from multi_agent_security.types import (
    AgentMessage,
    Patch,
    ReviewResult,
    TriageResult,
    Vulnerability,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are a senior security engineer reviewing a code patch that was generated to fix a security vulnerability. Your job is to determine if the patch is correct, secure, and maintainable.

Evaluate the patch on three dimensions:
1. **Correctness** (correctness_score, 0.0-1.0): Does the patch actually fix the vulnerability? Does it break any existing functionality? Is the logic sound?
2. **Security** (security_score, 0.0-1.0): Does the fix fully close the attack vector? Does it introduce any new vulnerabilities? Is it the right approach for this class of vulnerability?
3. **Style** (style_score, 0.0-1.0): Does the patch follow the existing code style? Is it minimal (no unnecessary changes)? Would an open-source maintainer accept this PR?

Decision rules:
- ACCEPT (patch_accepted = true) if: correctness_score >= 0.7 AND security_score >= 0.8 AND style_score >= 0.5
- REJECT (patch_accepted = false) if any score is below those thresholds

If rejecting, provide a specific, actionable revision_request that tells the patcher exactly what to change.

Respond with JSON only, matching the provided schema.\
"""

_REVISION_SECTION = """

NOTE: This is revision attempt {revision_number}. The patcher has already tried and failed. Be especially precise in your feedback — point to specific lines and be concrete about what needs to change.\
"""

_TEST_RESULT_SECTION = """\

### Test Results:
- Status: {test_status}
- Tests run: {tests_run}
- Tests failed: {tests_failed}
- Output (truncated):
```
{test_output}
```
A patch that causes test failures should almost always be REJECTED unless the test itself was testing insecure behavior.\
"""

# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------

_CORRECTNESS_THRESHOLD = 0.7
_SECURITY_THRESHOLD = 0.8
_STYLE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Input / output models
# ---------------------------------------------------------------------------


class ReviewerInput(BaseModel):
    vulnerability: Vulnerability
    triage: TriageResult
    patch: Patch
    original_file_content: str
    patched_file_content: str
    language: str
    test_result: Optional[TestResult] = None
    revision_number: int = 1


class ReviewerOutput(BaseModel):
    review: ReviewResult
    should_retry: bool


# ---------------------------------------------------------------------------
# Internal LLM response model
# ---------------------------------------------------------------------------


class _LLMReviewResponse(BaseModel):
    patch_accepted: bool
    correctness_score: float
    security_score: float
    style_score: float
    review_reasoning: str
    revision_request: Optional[str] = None


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _enforce_decision_rules(review: ReviewResult) -> ReviewResult:
    """Override the LLM's accept/reject decision if scores contradict it.

    - If LLM said accept but security_score < 0.8 (or other threshold not met) → force reject
    - If LLM said reject but all scores meet thresholds → force accept (LLM was too conservative)
    Logs any overrides.
    """
    thresholds_met = (
        review.correctness_score >= _CORRECTNESS_THRESHOLD
        and review.security_score >= _SECURITY_THRESHOLD
        and review.style_score >= _STYLE_THRESHOLD
    )
    if review.patch_accepted and not thresholds_met:
        logger.warning(
            "Overriding LLM accept decision for %s: scores do not meet thresholds "
            "(correctness=%.2f, security=%.2f, style=%.2f)",
            review.vuln_id,
            review.correctness_score,
            review.security_score,
            review.style_score,
        )
        return review.model_copy(update={"patch_accepted": False})
    if not review.patch_accepted and thresholds_met:
        logger.warning(
            "Overriding LLM reject decision for %s: all scores meet thresholds "
            "(correctness=%.2f, security=%.2f, style=%.2f)",
            review.vuln_id,
            review.correctness_score,
            review.security_score,
            review.style_score,
        )
        return review.model_copy(update={"patch_accepted": True})
    return review


def should_retry(review: ReviewResult, revision_number: int, max_revisions: int) -> bool:
    """Return True if the patcher should attempt another revision.

    Conditions (all must hold):
    - Patch was rejected
    - revision_number < max_revisions
    - The review has a non-empty revision_request (actionable feedback)
    - At least one score is >= 0.3 (the patch isn't completely wrong — worth retrying)
    """
    if review.patch_accepted:
        return False
    if revision_number >= max_revisions:
        return False
    if not review.revision_request:
        return False
    has_potential = (
        review.correctness_score >= 0.3
        or review.security_score >= 0.3
        or review.style_score >= 0.3
    )
    return has_potential


# ---------------------------------------------------------------------------
# ReviewerAgent
# ---------------------------------------------------------------------------


class ReviewerAgent(BaseAgent):
    name = "reviewer"

    async def run(
        self, input_data: BaseModel, context: list[AgentMessage]
    ) -> tuple[ReviewerOutput, AgentMessage]:
        assert isinstance(input_data, ReviewerInput)
        inp: ReviewerInput = input_data

        system_prompt = self._build_system_prompt(inp)
        user_prompt = self._build_user_prompt(inp)

        response = await self.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=_LLMReviewResponse,
        )

        try:
            llm_result = _LLMReviewResponse.model_validate_json(response.content)
        except Exception as exc:
            logger.warning(
                "Failed to parse LLM review response for %s: %s",
                inp.vulnerability.id,
                exc,
            )
            llm_result = _LLMReviewResponse(
                patch_accepted=False,
                correctness_score=0.0,
                security_score=0.0,
                style_score=0.0,
                review_reasoning="LLM response parse error — patch rejected by default.",
                revision_request="Could not parse reviewer response; please resubmit the patch.",
            )

        review = ReviewResult(
            vuln_id=inp.vulnerability.id,
            patch_accepted=llm_result.patch_accepted,
            correctness_score=max(0.0, min(1.0, llm_result.correctness_score)),
            security_score=max(0.0, min(1.0, llm_result.security_score)),
            style_score=max(0.0, min(1.0, llm_result.style_score)),
            review_reasoning=llm_result.review_reasoning,
            revision_request=llm_result.revision_request,
        )
        review = _enforce_decision_rules(review)

        max_revisions = self.config.agents.patcher.max_revision_loops
        retry = should_retry(review, inp.revision_number, max_revisions)

        output = ReviewerOutput(review=review, should_retry=retry)

        message = AgentMessage(
            agent_name=type(self).name,
            timestamp=datetime.now(timezone.utc),
            content=output.model_dump_json(),
            token_count_input=response.input_tokens,
            token_count_output=response.output_tokens,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
        )

        return output, message

    def _build_system_prompt(self, inp: ReviewerInput) -> str:
        prompt = _SYSTEM_PROMPT_BASE
        if inp.revision_number > 1:
            prompt += _REVISION_SECTION.format(revision_number=inp.revision_number)
        return prompt

    def _build_user_prompt(self, inp: ReviewerInput) -> str:
        vuln = inp.vulnerability
        triage = inp.triage
        patch = inp.patch

        parts = [
            f"## Vulnerability: {vuln.id} — {vuln.vuln_type.value}",
            f"## File: {vuln.file_path}",
            f"## Severity: {triage.severity.value}",
            f"## Fix Strategy Used: {triage.fix_strategy.value}",
            "",
            "### Vulnerability Description:",
            vuln.description,
            "",
            "### Original Code:",
            f"```{inp.language}",
            inp.original_file_content,
            "```",
            "",
            "### Patched Code:",
            f"```{inp.language}",
            inp.patched_file_content,
            "```",
            "",
            "### Unified Diff:",
            "```diff",
            patch.unified_diff,
            "```",
            "",
            "### Patcher's Reasoning:",
            patch.patch_reasoning,
        ]

        if inp.test_result is not None:
            tr = inp.test_result
            status = "PASSED" if tr.passed else "FAILED"
            parts += [
                "",
                _TEST_RESULT_SECTION.format(
                    test_status=status,
                    tests_run=tr.tests_run,
                    tests_failed=tr.tests_failed,
                    test_output=tr.output[:1000],
                ),
            ]

        parts += [
            "",
            "Review this patch and decide: ACCEPT or REJECT.",
        ]

        return "\n".join(parts)
