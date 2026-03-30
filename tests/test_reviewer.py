"""Tests for ReviewerAgent, _enforce_decision_rules, and should_retry."""
import json
import pathlib
from unittest.mock import AsyncMock, patch

import pytest

from multi_agent_security.agents.reviewer import (
    ReviewerAgent,
    ReviewerInput,
    ReviewerOutput,
    _enforce_decision_rules,
    should_retry,
)
from multi_agent_security.llm_client import LLMResponse
from multi_agent_security.tools.test_runner import TestResult
from multi_agent_security.types import (
    FixStrategy,
    Patch,
    ReviewResult,
    TriageResult,
    Vulnerability,
    VulnSeverity,
    VulnType,
)

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "vulnerable_repo"
_SQL_ORIGINAL = FIXTURES_DIR.joinpath("app.py").read_text()

_SQL_PATCHED = """\
import sqlite3


def get_user(username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    return cursor.fetchone()
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vulnerability(
    id: str = "VULN-001",
    file_path: str = "app.py",
    line_start: int = 7,
    line_end: int = 7,
    vuln_type: VulnType = VulnType.SQL_INJECTION,
    description: str = "SQL injection via f-string interpolation",
    code_snippet: str = "cursor.execute(f\"SELECT * FROM users WHERE username = '{username}'\")",
    scanner_reasoning: str = "Unsanitised user input interpolated into SQL query",
) -> Vulnerability:
    return Vulnerability(
        id=id,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        vuln_type=vuln_type,
        description=description,
        confidence=0.95,
        code_snippet=code_snippet,
        scanner_reasoning=scanner_reasoning,
    )


def _make_triage(
    vuln_id: str = "VULN-001",
    severity: VulnSeverity = VulnSeverity.HIGH,
    fix_strategy: FixStrategy = FixStrategy.ONE_LINER,
    complexity: str = "low",
) -> TriageResult:
    return TriageResult(
        vuln_id=vuln_id,
        severity=severity,
        exploitability_score=0.9,
        fix_strategy=fix_strategy,
        estimated_complexity=complexity,
        triage_reasoning="Direct SQL injection; easily exploitable from external input.",
    )


def _make_patch(
    vuln_id: str = "VULN-001",
    original_code: str = _SQL_ORIGINAL,
    patched_code: str = _SQL_PATCHED,
    unified_diff: str = "",
    patch_reasoning: str = "Fixed by using parameterised query.",
) -> Patch:
    return Patch(
        vuln_id=vuln_id,
        file_path="app.py",
        original_code=original_code,
        patched_code=patched_code,
        unified_diff=unified_diff,
        patch_reasoning=patch_reasoning,
    )


def _make_reviewer_input(
    vulnerability: Vulnerability | None = None,
    triage: TriageResult | None = None,
    patch: Patch | None = None,
    original_file_content: str = _SQL_ORIGINAL,
    patched_file_content: str = _SQL_PATCHED,
    language: str = "python",
    test_result: TestResult | None = None,
    revision_number: int = 1,
) -> ReviewerInput:
    return ReviewerInput(
        vulnerability=vulnerability or _make_vulnerability(),
        triage=triage or _make_triage(),
        patch=patch or _make_patch(),
        original_file_content=original_file_content,
        patched_file_content=patched_file_content,
        language=language,
        test_result=test_result,
        revision_number=revision_number,
    )


def _make_review_result(
    vuln_id: str = "VULN-001",
    patch_accepted: bool = True,
    correctness_score: float = 0.9,
    security_score: float = 0.9,
    style_score: float = 0.8,
    review_reasoning: str = "Patch correctly uses parameterised queries.",
    revision_request: str | None = None,
) -> ReviewResult:
    return ReviewResult(
        vuln_id=vuln_id,
        patch_accepted=patch_accepted,
        correctness_score=correctness_score,
        security_score=security_score,
        style_score=style_score,
        review_reasoning=review_reasoning,
        revision_request=revision_request,
    )


def _llm_review_response(
    patch_accepted: bool,
    correctness_score: float,
    security_score: float,
    style_score: float,
    review_reasoning: str,
    revision_request: str | None = None,
    input_tokens: int = 300,
    output_tokens: int = 80,
) -> LLMResponse:
    payload = json.dumps({
        "patch_accepted": patch_accepted,
        "correctness_score": correctness_score,
        "security_score": security_score,
        "style_score": style_score,
        "review_reasoning": review_reasoning,
        "revision_request": revision_request,
    })
    return LLMResponse(
        content=payload,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=75.0,
        cost_usd=0.002,
        model="claude-sonnet-4-20250514",
    )


# ---------------------------------------------------------------------------
# TestEnforceDecisionRules
# ---------------------------------------------------------------------------


class TestEnforceDecisionRules:
    def test_accepted_with_all_thresholds_met_unchanged(self):
        review = _make_review_result(
            patch_accepted=True,
            correctness_score=0.9,
            security_score=0.9,
            style_score=0.8,
        )
        result = _enforce_decision_rules(review)
        assert result.patch_accepted is True

    def test_accepted_with_low_security_score_forced_reject(self):
        review = _make_review_result(
            patch_accepted=True,
            correctness_score=0.9,
            security_score=0.3,  # below 0.8 threshold
            style_score=0.8,
        )
        result = _enforce_decision_rules(review)
        assert result.patch_accepted is False

    def test_accepted_with_low_correctness_score_forced_reject(self):
        review = _make_review_result(
            patch_accepted=True,
            correctness_score=0.5,  # below 0.7 threshold
            security_score=0.9,
            style_score=0.8,
        )
        result = _enforce_decision_rules(review)
        assert result.patch_accepted is False

    def test_rejected_with_all_thresholds_met_forced_accept(self):
        review = _make_review_result(
            patch_accepted=False,
            correctness_score=0.8,
            security_score=0.9,
            style_score=0.7,
            revision_request="Some nitpick",
        )
        result = _enforce_decision_rules(review)
        assert result.patch_accepted is True

    def test_rejected_below_thresholds_unchanged(self):
        review = _make_review_result(
            patch_accepted=False,
            correctness_score=0.4,
            security_score=0.4,
            style_score=0.3,
            revision_request="Fix the vulnerability properly.",
        )
        result = _enforce_decision_rules(review)
        assert result.patch_accepted is False

    def test_other_fields_preserved_after_override(self):
        review = _make_review_result(
            patch_accepted=True,
            correctness_score=0.9,
            security_score=0.3,
            style_score=0.8,
            review_reasoning="Looks mostly fine.",
        )
        result = _enforce_decision_rules(review)
        assert result.review_reasoning == "Looks mostly fine."
        assert result.correctness_score == 0.9


# ---------------------------------------------------------------------------
# TestShouldRetry
# ---------------------------------------------------------------------------


class TestShouldRetry:
    def test_accepted_patch_never_retries(self):
        review = _make_review_result(patch_accepted=True)
        assert should_retry(review, revision_number=1, max_revisions=3) is False

    def test_rejected_with_revisions_remaining_and_request_retries(self):
        review = _make_review_result(
            patch_accepted=False,
            correctness_score=0.5,
            security_score=0.5,
            style_score=0.5,
            revision_request="Use parameterised queries instead of string interpolation.",
        )
        assert should_retry(review, revision_number=1, max_revisions=3) is True

    def test_rejected_at_max_revisions_no_retry(self):
        review = _make_review_result(
            patch_accepted=False,
            correctness_score=0.5,
            security_score=0.5,
            style_score=0.5,
            revision_request="Still needs fixing.",
        )
        assert should_retry(review, revision_number=3, max_revisions=3) is False

    def test_rejected_beyond_max_revisions_no_retry(self):
        review = _make_review_result(
            patch_accepted=False,
            correctness_score=0.5,
            revision_request="Still needs fixing.",
        )
        assert should_retry(review, revision_number=4, max_revisions=3) is False

    def test_rejected_with_empty_revision_request_no_retry(self):
        review = _make_review_result(
            patch_accepted=False,
            correctness_score=0.5,
            security_score=0.5,
            style_score=0.5,
            revision_request=None,
        )
        assert should_retry(review, revision_number=1, max_revisions=3) is False

    def test_rejected_with_all_scores_below_0_3_no_retry(self):
        review = _make_review_result(
            patch_accepted=False,
            correctness_score=0.1,
            security_score=0.1,
            style_score=0.1,
            revision_request="This patch is completely wrong.",
        )
        assert should_retry(review, revision_number=1, max_revisions=3) is False


# ---------------------------------------------------------------------------
# TestReviewerAgent
# ---------------------------------------------------------------------------


class TestReviewerAgent:
    async def test_mock_llm_accepts_good_patch(self, app_config, dry_run_llm_client):
        agent = ReviewerAgent(config=app_config, llm_client=dry_run_llm_client)
        inp = _make_reviewer_input()
        mock_resp = _llm_review_response(
            patch_accepted=True,
            correctness_score=0.9,
            security_score=0.9,
            style_score=0.8,
            review_reasoning="Patch correctly uses parameterised queries, closing the injection vector.",
        )

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            output, message = await agent.run(inp, context=[])

        assert isinstance(output, ReviewerOutput)
        assert output.review.patch_accepted is True
        assert output.review.correctness_score >= 0.7
        assert output.review.security_score >= 0.8
        assert output.should_retry is False

    async def test_mock_llm_rejects_bad_patch(self, app_config, dry_run_llm_client):
        agent = ReviewerAgent(config=app_config, llm_client=dry_run_llm_client)
        inp = _make_reviewer_input(revision_number=1)
        mock_resp = _llm_review_response(
            patch_accepted=False,
            correctness_score=0.4,
            security_score=0.3,
            style_score=0.6,
            review_reasoning="The patch does not fix the SQL injection.",
            revision_request="Use parameterised queries (cursor.execute with ? placeholder) instead of f-string interpolation.",
        )

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            output, message = await agent.run(inp, context=[])

        assert output.review.patch_accepted is False
        assert output.review.revision_request
        assert len(output.review.revision_request) > 0
        assert output.should_retry is True

    async def test_decision_override_llm_accepts_low_security(
        self, app_config, dry_run_llm_client
    ):
        agent = ReviewerAgent(config=app_config, llm_client=dry_run_llm_client)
        inp = _make_reviewer_input()
        mock_resp = _llm_review_response(
            patch_accepted=True,
            correctness_score=0.9,
            security_score=0.3,  # below 0.8 threshold
            style_score=0.8,
            review_reasoning="Mostly fine.",
        )

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            output, _ = await agent.run(inp, context=[])

        assert output.review.patch_accepted is False

    async def test_retry_exhausted_at_max_revisions(self, app_config, dry_run_llm_client):
        agent = ReviewerAgent(config=app_config, llm_client=dry_run_llm_client)
        max_revisions = app_config.agents.patcher.max_revision_loops
        inp = _make_reviewer_input(revision_number=max_revisions)
        mock_resp = _llm_review_response(
            patch_accepted=False,
            correctness_score=0.5,
            security_score=0.5,
            style_score=0.5,
            review_reasoning="Still not fixed.",
            revision_request="Try a different approach.",
        )

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            output, _ = await agent.run(inp, context=[])

        assert output.review.patch_accepted is False
        assert output.should_retry is False

    async def test_failed_tests_influence_rejection_reasoning(
        self, app_config, dry_run_llm_client
    ):
        agent = ReviewerAgent(config=app_config, llm_client=dry_run_llm_client)
        test_result = TestResult(
            passed=False,
            output="FAILED tests/test_app.py::test_get_user - AssertionError",
            tests_run=3,
            tests_failed=1,
        )
        inp = _make_reviewer_input(test_result=test_result)
        mock_resp = _llm_review_response(
            patch_accepted=False,
            correctness_score=0.4,
            security_score=0.6,
            style_score=0.7,
            review_reasoning="The patch causes test failures, indicating broken functionality.",
            revision_request="The patch breaks existing tests. Fix the logic so tests pass.",
        )

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            output, _ = await agent.run(inp, context=[])

        assert output.review.patch_accepted is False
        assert output.review.revision_request
        assert output.should_retry is True

    async def test_agent_message_fields_populated(self, app_config, dry_run_llm_client):
        agent = ReviewerAgent(config=app_config, llm_client=dry_run_llm_client)
        inp = _make_reviewer_input()
        mock_resp = _llm_review_response(
            patch_accepted=True,
            correctness_score=0.9,
            security_score=0.9,
            style_score=0.8,
            review_reasoning="LGTM.",
            input_tokens=400,
            output_tokens=90,
        )

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            _, message = await agent.run(inp, context=[])

        assert message.agent_name == "reviewer"
        assert message.token_count_input == 400
        assert message.token_count_output == 90
        assert message.cost_usd > 0
        assert message.latency_ms >= 0
        assert message.timestamp.tzinfo is not None

    async def test_parse_error_fallback_rejects(self, app_config, dry_run_llm_client):
        agent = ReviewerAgent(config=app_config, llm_client=dry_run_llm_client)
        inp = _make_reviewer_input()
        bad_resp = LLMResponse(
            content="this is not json at all",
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            cost_usd=0.0,
            model="claude-sonnet-4-20250514",
        )

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=bad_resp)):
            output, _ = await agent.run(inp, context=[])

        assert output.review.patch_accepted is False
        # All scores are 0.0 → has_potential is False → should_retry is False
        assert output.should_retry is False

    async def test_revision_number_appears_in_system_prompt(
        self, app_config, dry_run_llm_client
    ):
        agent = ReviewerAgent(config=app_config, llm_client=dry_run_llm_client)
        inp = _make_reviewer_input(revision_number=2)
        captured: dict = {}

        async def capture(system_prompt, user_prompt, **kwargs):
            captured["system_prompt"] = system_prompt
            return _llm_review_response(
                patch_accepted=True,
                correctness_score=0.9,
                security_score=0.9,
                style_score=0.8,
                review_reasoning="LGTM.",
            )

        with patch.object(dry_run_llm_client, "complete", AsyncMock(side_effect=capture)):
            await agent.run(inp, context=[])

        assert "revision attempt 2" in captured["system_prompt"].lower() or "2" in captured["system_prompt"]

    async def test_test_result_in_user_prompt_when_provided(
        self, app_config, dry_run_llm_client
    ):
        agent = ReviewerAgent(config=app_config, llm_client=dry_run_llm_client)
        test_result = TestResult(
            passed=False,
            output="AssertionError: expected secure query",
            tests_run=5,
            tests_failed=2,
        )
        inp = _make_reviewer_input(test_result=test_result)
        captured: dict = {}

        async def capture(system_prompt, user_prompt, **kwargs):
            captured["user_prompt"] = user_prompt
            return _llm_review_response(
                patch_accepted=False,
                correctness_score=0.4,
                security_score=0.4,
                style_score=0.5,
                review_reasoning="Tests failed.",
                revision_request="Fix the broken tests.",
            )

        with patch.object(dry_run_llm_client, "complete", AsyncMock(side_effect=capture)):
            await agent.run(inp, context=[])

        assert "FAILED" in captured["user_prompt"]
        assert "AssertionError" in captured["user_prompt"]

    async def test_test_result_absent_from_user_prompt_when_none(
        self, app_config, dry_run_llm_client
    ):
        agent = ReviewerAgent(config=app_config, llm_client=dry_run_llm_client)
        inp = _make_reviewer_input(test_result=None)
        captured: dict = {}

        async def capture(system_prompt, user_prompt, **kwargs):
            captured["user_prompt"] = user_prompt
            return _llm_review_response(
                patch_accepted=True,
                correctness_score=0.9,
                security_score=0.9,
                style_score=0.8,
                review_reasoning="LGTM.",
            )

        with patch.object(dry_run_llm_client, "complete", AsyncMock(side_effect=capture)):
            await agent.run(inp, context=[])

        assert "Test Results" not in captured["user_prompt"]
        assert "tests_run" not in captured["user_prompt"]


# ---------------------------------------------------------------------------
# TestReviewerIntegration
# ---------------------------------------------------------------------------


class TestReviewerIntegration:
    async def test_empty_diff_patch_rejected_with_feedback(
        self, app_config, dry_run_llm_client
    ):
        """A deliberately bad patch (empty diff, no actual fix) should be rejected
        with actionable feedback and should_retry=True on the first attempt."""
        bad_patch = _make_patch(
            patched_code=_SQL_ORIGINAL,  # no actual change
            unified_diff="",
            patch_reasoning="No changes needed.",
        )
        inp = _make_reviewer_input(
            patch=bad_patch,
            patched_file_content=_SQL_ORIGINAL,  # same as original
            revision_number=1,
        )
        mock_resp = _llm_review_response(
            patch_accepted=False,
            correctness_score=0.1,
            security_score=0.0,
            style_score=0.5,
            review_reasoning="The patch is empty — it makes no changes to the vulnerable code.",
            revision_request="The f-string interpolation on line 7 must be replaced with a parameterised query using cursor.execute with a ? placeholder.",
        )

        agent = ReviewerAgent(config=app_config, llm_client=dry_run_llm_client)

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            output, _ = await agent.run(inp, context=[])

        assert output.review.patch_accepted is False
        assert output.review.revision_request
        assert len(output.review.revision_request) > 0
        assert output.should_retry is True
