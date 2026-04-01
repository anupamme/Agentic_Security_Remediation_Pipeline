"""Unit tests for evaluation metric functions."""
import statistics
import pytest
from datetime import datetime, timezone

from multi_agent_security.eval.metrics import (
    aggregate_metrics,
    compute_context_utilization,
    compute_detection_precision,
    compute_detection_recall,
    compute_diff_similarity,
    compute_e2e_success,
    compute_patch_correctness,
    compute_triage_accuracy,
)
from multi_agent_security.types import (
    EvalResult,
    FixStrategy,
    Patch,
    TriageResult,
    Vulnerability,
    VulnSeverity,
    VulnType,
)


def _make_vuln(file_path: str, vuln_type: VulnType = VulnType.SQL_INJECTION) -> Vulnerability:
    return Vulnerability(
        id="VULN-001",
        file_path=file_path,
        line_start=1,
        line_end=1,
        vuln_type=vuln_type,
        description="test",
        confidence=0.9,
        code_snippet="code",
        scanner_reasoning="reason",
    )


def _make_triage(severity: VulnSeverity) -> TriageResult:
    return TriageResult(
        vuln_id="VULN-001",
        severity=severity,
        exploitability_score=0.5,
        fix_strategy=FixStrategy.ONE_LINER,
        estimated_complexity="low",
        triage_reasoning="reason",
    )


def _make_patch(diff: str = "") -> Patch:
    return Patch(
        vuln_id="VULN-001",
        file_path="app.py",
        original_code="original",
        patched_code="patched",
        unified_diff=diff,
        patch_reasoning="reason",
    )


def _make_eval_result(**kwargs) -> EvalResult:
    defaults = dict(
        example_id="BENCH-001",
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


# === Detection Recall ===

def test_detection_recall_partial():
    vulns = [_make_vuln("app/a.py"), _make_vuln("app/b.py")]
    gt = ["app/a.py", "app/b.py", "app/c.py"]
    score = compute_detection_recall(vulns, gt, VulnType.SQL_INJECTION)
    assert abs(score - 2 / 3) < 0.001


def test_detection_recall_full():
    vulns = [_make_vuln("app/a.py"), _make_vuln("app/b.py")]
    gt = ["app/a.py", "app/b.py"]
    assert compute_detection_recall(vulns, gt, VulnType.SQL_INJECTION) == 1.0


def test_detection_recall_empty_gt():
    vulns = [_make_vuln("app/a.py")]
    assert compute_detection_recall(vulns, [], VulnType.SQL_INJECTION) == 1.0


def test_detection_recall_normalizes_paths():
    vulns = [_make_vuln("./app/a.py")]
    gt = ["app/a.py"]
    assert compute_detection_recall(vulns, gt, VulnType.SQL_INJECTION) == 1.0


# === Detection Precision ===

def test_detection_precision_partial():
    vulns = [_make_vuln("app/a.py"), _make_vuln("app/b.py"), _make_vuln("app/c.py")]
    gt = ["app/a.py", "app/b.py"]
    score = compute_detection_precision(vulns, gt, is_negative_example=False)
    assert abs(score - 2 / 3) < 0.001


def test_detection_precision_no_predictions():
    assert compute_detection_precision([], ["app/a.py"], is_negative_example=False) == 1.0


def test_detection_precision_negative_example_with_predictions():
    vulns = [_make_vuln("app/a.py")]
    assert compute_detection_precision(vulns, [], is_negative_example=True) == 0.0


def test_detection_precision_negative_no_predictions():
    assert compute_detection_precision([], [], is_negative_example=True) == 1.0


# === Triage Accuracy ===

def test_triage_accuracy_exact():
    triage = [_make_triage(VulnSeverity.HIGH)]
    assert compute_triage_accuracy(triage, VulnSeverity.HIGH, VulnType.SQL_INJECTION) == 1.0


def test_triage_accuracy_off_by_one():
    triage = [_make_triage(VulnSeverity.MEDIUM)]
    score = compute_triage_accuracy(triage, VulnSeverity.HIGH, VulnType.SQL_INJECTION)
    assert score == 0.5


def test_triage_accuracy_off_by_two():
    triage = [_make_triage(VulnSeverity.LOW)]
    score = compute_triage_accuracy(triage, VulnSeverity.HIGH, VulnType.SQL_INJECTION)
    assert score == 0.0


def test_triage_accuracy_no_results():
    assert compute_triage_accuracy([], VulnSeverity.HIGH, VulnType.SQL_INJECTION) == 0.0


# === Diff Similarity ===

def test_diff_similarity_identical():
    diff = "+def fix():\n+    pass\n"
    assert compute_diff_similarity(diff, diff) == 1.0


def test_diff_similarity_completely_different():
    diff1 = "+def foo_bar_baz_qux():\n+    return 42\n"
    diff2 = "-class ZZZUnrelated:\n-    x = 999\n"
    score = compute_diff_similarity(diff1, diff2)
    assert score < 0.3


def test_diff_similarity_both_empty():
    assert compute_diff_similarity("", "") == 1.0


def test_diff_similarity_one_empty():
    assert compute_diff_similarity("+foo = 1\n", "") == 0.0


# === Patch Correctness ===

def test_patch_correctness_all_components():
    from multi_agent_security.tools.test_runner import TestResult
    patch = _make_patch("+fix()\n")
    tr = TestResult(passed=True, output="ok", tests_run=5, tests_failed=0)
    score = compute_patch_correctness(patch, "+fix()\n", test_result=tr, judge_score=0.9)
    # test: 1.0 * 0.4, diff: 1.0 * 0.3, judge: 0.9 * 0.3 = 0.4 + 0.3 + 0.27 = 0.97
    assert score > 0.9


def test_patch_correctness_no_test_result():
    patch = _make_patch("+fix()\n")
    # Only diff (0.3) and judge (0.3), redistributed to 0.5/0.5
    score = compute_patch_correctness(patch, "+fix()\n", test_result=None, judge_score=1.0)
    assert score > 0.9


# === E2E Success ===

def test_e2e_success_all_pass():
    assert compute_e2e_success(True, True, True, True, 0.8) is True


def test_e2e_success_patch_correct_below_threshold():
    assert compute_e2e_success(True, True, True, True, 0.4) is False


def test_e2e_success_not_detected():
    assert compute_e2e_success(False, True, True, True, 0.8) is False


def test_e2e_success_not_accepted():
    assert compute_e2e_success(True, True, True, False, 0.8) is False


# === Context Utilization ===

def test_context_utilization():
    assert compute_context_utilization(100_000) == 0.5
    assert compute_context_utilization(200_000) == 1.0
    assert compute_context_utilization(0) == 0.0


# === Aggregation ===

def test_aggregate_known_results():
    results = [
        _make_eval_result(
            example_id="A", detection_recall=0.5, detection_precision=1.0,
            triage_accuracy=1.0, patch_correctness=0.8, end_to_end_success=True,
            total_tokens=1000, total_cost_usd=0.01, latency_seconds=5.0, revision_loops=0,
            vuln_type="CWE-89", complexity_tag="single_file",
        ),
        _make_eval_result(
            example_id="B", detection_recall=1.0, detection_precision=0.5,
            triage_accuracy=0.5, patch_correctness=0.6, end_to_end_success=False,
            total_tokens=2000, total_cost_usd=0.02, latency_seconds=10.0, revision_loops=2,
            vuln_type="CWE-89", complexity_tag="multi_file",
        ),
        _make_eval_result(
            example_id="C", detection_recall=0.0, detection_precision=0.0,
            triage_accuracy=0.0, patch_correctness=0.0, end_to_end_success=False,
            total_tokens=500, total_cost_usd=0.005, latency_seconds=2.0, revision_loops=1,
            vuln_type="CWE-79", complexity_tag="single_file",
        ),
    ]
    agg = aggregate_metrics(results)
    assert agg.n_examples == 3
    assert abs(agg.detection_recall.mean - statistics.mean([0.5, 1.0, 0.0])) < 0.001
    assert abs(agg.e2e_success_rate - 1/3) < 0.001
    # Per-type breakdown
    assert "CWE-89" in agg.by_vuln_type
    assert agg.by_vuln_type["CWE-89"].n_examples == 2
    # Per-complexity breakdown
    assert "single_file" in agg.by_complexity
    assert agg.by_complexity["single_file"].n_examples == 2
