"""Unit tests for the failure analysis / error taxonomy module."""
import pytest
from datetime import datetime, timezone

from multi_agent_security.eval.failure_analysis import FailureClassifier, generate_recommendations
from multi_agent_security.tools.test_runner import TestResult
from multi_agent_security.types import (
    BenchmarkExample,
    EvalResult,
    FailureAnalysis,
    FailureCategory,
    FixStrategy,
    Patch,
    ReviewResult,
    TaskState,
    TriageResult,
    Vulnerability,
    VulnSeverity,
    VulnType,
)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_example(
    *,
    negative: bool = False,
    vuln_type: VulnType = VulnType.SQL_INJECTION,
    severity: VulnSeverity = VulnSeverity.HIGH,
    vulnerable_files: list[str] | None = None,
    ground_truth_diff: str = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-bad\n+good\n",
) -> BenchmarkExample:
    return BenchmarkExample(
        id="BENCH-0001",
        repo_url="https://github.com/example/repo",
        repo_name="example/repo",
        language="python",
        vulnerable_files=vulnerable_files or ["app.py"],
        vuln_type=vuln_type,
        severity=severity,
        ground_truth_diff=ground_truth_diff,
        merge_status="merged",
        complexity_tag="single_file",
        negative=negative,
    )


def _make_eval_result(**kwargs) -> EvalResult:
    defaults = dict(
        example_id="BENCH-0001",
        architecture="sequential",
        memory_strategy="full_context",
        detection_recall=1.0,
        detection_precision=1.0,
        triage_accuracy=1.0,
        patch_correctness=1.0,
        end_to_end_success=True,
        total_tokens=1000,
        total_cost_usd=0.01,
        latency_seconds=5.0,
        revision_loops=0,
    )
    defaults.update(kwargs)
    return EvalResult(**defaults)


def _make_task_state(**kwargs) -> TaskState:
    defaults = dict(
        task_id="BENCH-0001-run1",
        repo_url="https://github.com/example/repo",
        language="python",
    )
    defaults.update(kwargs)
    return TaskState(**defaults)


def _make_vuln(
    file_path: str = "app.py",
    vuln_type: VulnType = VulnType.SQL_INJECTION,
) -> Vulnerability:
    return Vulnerability(
        id="VULN-001",
        file_path=file_path,
        line_start=10,
        line_end=12,
        vuln_type=vuln_type,
        description="SQL injection via unsanitised input",
        confidence=0.9,
        code_snippet="query = f'SELECT * FROM users WHERE id={user_id}'",
        scanner_reasoning="String interpolation in SQL query",
    )


def _make_patch(
    unified_diff: str = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-bad\n+good\n",
    file_path: str = "app.py",
) -> Patch:
    return Patch(
        vuln_id="VULN-001",
        file_path=file_path,
        original_code="bad",
        patched_code="good",
        unified_diff=unified_diff,
        patch_reasoning="Use parameterised query",
    )


def _make_review(*, accepted: bool = True, correctness: float = 0.9, security: float = 0.9) -> ReviewResult:
    return ReviewResult(
        vuln_id="VULN-001",
        patch_accepted=accepted,
        correctness_score=correctness,
        security_score=security,
        style_score=0.8,
        review_reasoning="Looks good" if accepted else "Does not fix the issue",
        revision_request=None if accepted else "Please use parameterised queries",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassifyFailure:
    def setup_method(self):
        self.clf = FailureClassifier()

    def test_classify_no_failures(self):
        """Perfect result returns empty list."""
        example = _make_example()
        result = _make_eval_result()
        state = _make_task_state()
        failures = self.clf.classify_failure(example, result, state)
        assert failures == []

    def test_scanner_miss(self):
        """detection_recall < 1.0 on non-negative example → SCANNER_MISS."""
        example = _make_example(negative=False)
        result = _make_eval_result(
            detection_recall=0.0,
            end_to_end_success=False,
        )
        state = _make_task_state()
        failures = self.clf.classify_failure(example, result, state)
        categories = [f.failure_category for f in failures]
        assert FailureCategory.SCANNER_MISS in categories
        miss = next(f for f in failures if f.failure_category == FailureCategory.SCANNER_MISS)
        assert miss.failure_stage == "scanner"
        assert miss.severity == "critical"

    def test_scanner_miss_not_raised_on_negative(self):
        """Negative example with recall=0 should NOT produce SCANNER_MISS."""
        example = _make_example(negative=True)
        result = _make_eval_result(
            detection_recall=0.0,
            detection_precision=1.0,
            end_to_end_success=True,
        )
        state = _make_task_state()
        failures = self.clf.classify_failure(example, result, state)
        categories = [f.failure_category for f in failures]
        assert FailureCategory.SCANNER_MISS not in categories

    def test_scanner_false_positive_on_negative(self):
        """Negative example with detected vulns → SCANNER_FALSE_POSITIVE."""
        example = _make_example(negative=True)
        result = _make_eval_result(
            detection_recall=0.0,
            detection_precision=0.0,
            end_to_end_success=False,
        )
        state = _make_task_state(
            vulnerabilities=[_make_vuln()],
        )
        failures = self.clf.classify_failure(example, result, state)
        categories = [f.failure_category for f in failures]
        assert FailureCategory.SCANNER_FALSE_POSITIVE in categories
        fp = next(f for f in failures if f.failure_category == FailureCategory.SCANNER_FALSE_POSITIVE)
        assert fp.severity == "critical"

    def test_scanner_false_positive_on_positive(self):
        """Positive example with precision < 1 → SCANNER_FALSE_POSITIVE (minor)."""
        example = _make_example(negative=False)
        result = _make_eval_result(
            detection_recall=1.0,
            detection_precision=0.5,
        )
        state = _make_task_state()
        failures = self.clf.classify_failure(example, result, state)
        categories = [f.failure_category for f in failures]
        assert FailureCategory.SCANNER_FALSE_POSITIVE in categories
        fp = next(f for f in failures if f.failure_category == FailureCategory.SCANNER_FALSE_POSITIVE)
        assert fp.severity == "minor"

    def test_scanner_wrong_type(self):
        """Scanner finds a vuln but wrong CWE type → SCANNER_WRONG_TYPE."""
        example = _make_example(vuln_type=VulnType.SQL_INJECTION, negative=False)
        result = _make_eval_result(detection_recall=1.0)
        wrong_vuln = _make_vuln(vuln_type=VulnType.XSS)
        state = _make_task_state(vulnerabilities=[wrong_vuln])
        failures = self.clf.classify_failure(example, result, state)
        categories = [f.failure_category for f in failures]
        assert FailureCategory.SCANNER_WRONG_TYPE in categories

    def test_patcher_empty_patch(self):
        """Empty unified_diff → PATCHER_EMPTY_PATCH."""
        example = _make_example()
        result = _make_eval_result(
            patch_correctness=0.1,
            end_to_end_success=False,
        )
        empty_patch = _make_patch(unified_diff="")
        state = _make_task_state(patches=[empty_patch])
        failures = self.clf.classify_failure(example, result, state)
        categories = [f.failure_category for f in failures]
        assert FailureCategory.PATCHER_EMPTY_PATCH in categories

    def test_patcher_breaks_code(self):
        """_classify_patch_failure with failing test → PATCHER_BREAKS_CODE."""
        clf = FailureClassifier()
        patch = _make_patch()
        test_result = TestResult(passed=False, output="FAILED test_app.py", tests_run=5, tests_failed=2)
        category = clf._classify_patch_failure(
            patch=patch,
            ground_truth_diff="--- a/app.py\n+++ b/app.py\n",
            review=None,
            test_result=test_result,
        )
        assert category == FailureCategory.PATCHER_BREAKS_CODE

    def test_patcher_wrong_file(self):
        """Patch targets wrong file → PATCHER_WRONG_FILE."""
        clf = FailureClassifier()
        patch = _make_patch(file_path="unrelated.py")
        ground_truth_diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-bad\n+good\n"
        category = clf._classify_patch_failure(
            patch=patch,
            ground_truth_diff=ground_truth_diff,
            review=None,
            test_result=None,
        )
        assert category == FailureCategory.PATCHER_WRONG_FILE

    def test_multiple_failures(self):
        """Scanner miss AND patcher partial fix → both categories returned."""
        example = _make_example(negative=False)
        result = _make_eval_result(
            detection_recall=0.5,
            patch_correctness=0.2,
            end_to_end_success=False,
        )
        patch = _make_patch(unified_diff="--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-x\n+y\n")
        state = _make_task_state(patches=[patch])
        failures = self.clf.classify_failure(example, result, state)
        categories = [f.failure_category for f in failures]
        assert FailureCategory.SCANNER_MISS in categories
        # Patch correctness < 0.5 should trigger a patcher failure category
        patcher_cats = {
            FailureCategory.PATCHER_EMPTY_PATCH,
            FailureCategory.PATCHER_WRONG_FILE,
            FailureCategory.PATCHER_BREAKS_CODE,
            FailureCategory.PATCHER_PARTIAL_FIX,
            FailureCategory.PATCHER_WRONG_APPROACH,
        }
        assert patcher_cats & set(categories), f"Expected a patcher failure in {categories}"

    def test_max_retries(self):
        """revision_count >= max_revisions with failed_vulns → MAX_RETRIES."""
        example = _make_example()
        result = _make_eval_result(
            revision_loops=3,
            end_to_end_success=False,
            patch_correctness=0.0,
        )
        state = _make_task_state(
            revision_count=3,
            max_revisions=3,
            failed_vulns=["VULN-001"],
        )
        failures = self.clf.classify_failure(example, result, state)
        categories = [f.failure_category for f in failures]
        assert FailureCategory.MAX_RETRIES in categories

    def test_reviewer_false_reject(self):
        """Good patch_correctness but no accepted patch → REVIEWER_FALSE_REJECT."""
        example = _make_example()
        result = _make_eval_result(
            patch_correctness=0.8,
            end_to_end_success=False,
        )
        patch = _make_patch()
        review = _make_review(accepted=False, correctness=0.3, security=0.9)
        state = _make_task_state(
            patches=[patch],
            reviews=[review],
        )
        failures = self.clf.classify_failure(example, result, state)
        categories = [f.failure_category for f in failures]
        assert (
            FailureCategory.REVIEWER_FALSE_REJECT in categories
            or FailureCategory.REVIEWER_UNHELPFUL_FEEDBACK in categories
        )

    def test_system_context_overflow(self):
        """failure_reason containing 'context' → CONTEXT_OVERFLOW."""
        example = _make_example()
        result = _make_eval_result(
            end_to_end_success=False,
            failure_stage="runner",
            failure_reason="Exceeded context window limit",
        )
        state = _make_task_state()
        failures = self.clf.classify_failure(example, result, state)
        categories = [f.failure_category for f in failures]
        assert FailureCategory.CONTEXT_OVERFLOW in categories

    def test_system_api_error(self):
        """failure_reason containing 'api error' → API_ERROR."""
        example = _make_example()
        result = _make_eval_result(
            end_to_end_success=False,
            failure_stage="runner",
            failure_reason="LLM API error: 500 Internal Server Error",
        )
        state = _make_task_state()
        failures = self.clf.classify_failure(example, result, state)
        categories = [f.failure_category for f in failures]
        assert FailureCategory.API_ERROR in categories


class TestGenerateRecommendations:
    def _make_failure(self, category: FailureCategory) -> FailureAnalysis:
        return FailureAnalysis(
            example_id="BENCH-0001",
            config_id="C1",
            failure_category=category,
            failure_stage="scanner",
            severity="critical",
            description="test",
            root_cause="test",
        )

    def test_empty_input(self):
        result = generate_recommendations([])
        assert "succeeded" in result.lower()

    def test_scanner_miss_dominant(self):
        failures = [self._make_failure(FailureCategory.SCANNER_MISS)] * 10
        result = generate_recommendations(failures)
        assert "scanner_miss" in result
        assert "static" in result.lower() or "semgrep" in result.lower()

    def test_patcher_breaks_dominant(self):
        failures = [self._make_failure(FailureCategory.PATCHER_BREAKS_CODE)] * 10
        result = generate_recommendations(failures)
        assert "patcher_breaks" in result
        assert "test" in result.lower()

    def test_info_loss_dominant(self):
        failures = [self._make_failure(FailureCategory.INFO_LOSS)] * 10
        result = generate_recommendations(failures)
        assert "info_loss" in result
        assert "full_context" in result.lower()

    def test_max_retries_dominant(self):
        failures = [self._make_failure(FailureCategory.MAX_RETRIES)] * 10
        result = generate_recommendations(failures)
        assert "max_retries" in result
        assert "reviewer" in result.lower() or "feedback" in result.lower()

    def test_multiple_categories(self):
        failures = (
            [self._make_failure(FailureCategory.SCANNER_MISS)] * 5
            + [self._make_failure(FailureCategory.PATCHER_BREAKS_CODE)] * 3
        )
        result = generate_recommendations(failures)
        assert "scanner_miss" in result
        assert "patcher_breaks" in result
