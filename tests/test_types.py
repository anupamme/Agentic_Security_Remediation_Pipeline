import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from multi_agent_security.types import (
    AgentMessage,
    BenchmarkExample,
    EvalResult,
    FileContext,
    FixStrategy,
    Patch,
    ReviewResult,
    TaskState,
    TriageResult,
    Vulnerability,
    VulnSeverity,
    VulnType,
)


def test_vuln_severity_values():
    assert VulnSeverity.CRITICAL == "critical"
    assert VulnSeverity.HIGH == "high"


def test_vuln_type_values():
    assert VulnType.SQL_INJECTION == "CWE-89"
    assert VulnType.XSS == "CWE-79"


def test_file_context():
    fc = FileContext(path="app/main.py", content="print('hello')", language="python", line_count=1)
    assert fc.path == "app/main.py"
    assert fc.line_count == 1


def test_vulnerability_valid(sample_vulnerability):
    assert sample_vulnerability.id == "VULN-001"
    assert sample_vulnerability.confidence == 0.95


def test_vulnerability_confidence_too_high():
    with pytest.raises(ValidationError):
        Vulnerability(
            id="VULN-X",
            file_path="foo.py",
            line_start=1,
            line_end=1,
            vuln_type=VulnType.XSS,
            description="test",
            confidence=1.5,  # invalid: > 1.0
            code_snippet="x",
            scanner_reasoning="y",
        )


def test_vulnerability_confidence_too_low():
    with pytest.raises(ValidationError):
        Vulnerability(
            id="VULN-X",
            file_path="foo.py",
            line_start=1,
            line_end=1,
            vuln_type=VulnType.XSS,
            description="test",
            confidence=-0.1,  # invalid: < 0.0
            code_snippet="x",
            scanner_reasoning="y",
        )


def test_triage_result():
    tr = TriageResult(
        vuln_id="VULN-001",
        severity=VulnSeverity.HIGH,
        exploitability_score=0.8,
        fix_strategy=FixStrategy.ONE_LINER,
        estimated_complexity="low",
        triage_reasoning="Easy fix.",
    )
    assert tr.severity == VulnSeverity.HIGH
    assert tr.exploitability_score == 0.8


def test_triage_result_invalid_exploitability():
    with pytest.raises(ValidationError):
        TriageResult(
            vuln_id="VULN-001",
            severity=VulnSeverity.HIGH,
            exploitability_score=2.0,  # invalid
            fix_strategy=FixStrategy.ONE_LINER,
            estimated_complexity="low",
            triage_reasoning="test",
        )


def test_patch():
    p = Patch(
        vuln_id="VULN-001",
        file_path="app/views.py",
        original_code="bad code",
        patched_code="good code",
        unified_diff="--- a\n+++ b\n",
        patch_reasoning="Fixed SQL injection.",
    )
    assert p.vuln_id == "VULN-001"


def test_review_result():
    rr = ReviewResult(
        vuln_id="VULN-001",
        patch_accepted=True,
        correctness_score=0.9,
        security_score=1.0,
        style_score=0.8,
        review_reasoning="Looks good.",
    )
    assert rr.patch_accepted is True
    assert rr.revision_request is None


def test_review_result_invalid_scores():
    with pytest.raises(ValidationError):
        ReviewResult(
            vuln_id="VULN-001",
            patch_accepted=False,
            correctness_score=1.5,  # invalid
            security_score=0.5,
            style_score=0.5,
            review_reasoning="test",
        )


def test_agent_message():
    msg = AgentMessage(
        agent_name="scanner",
        timestamp=datetime.now(timezone.utc),
        content="{}",
        token_count_input=10,
        token_count_output=5,
        latency_ms=100.0,
        cost_usd=0.0001,
    )
    assert msg.agent_name == "scanner"


def test_agent_message_naive_timestamp_rejected():
    with pytest.raises(ValidationError, match="timezone-aware"):
        AgentMessage(
            agent_name="scanner",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),  # naive — no tzinfo
            content="{}",
            token_count_input=10,
            token_count_output=5,
            latency_ms=100.0,
            cost_usd=0.0001,
        )


def test_task_state(sample_task_state):
    assert sample_task_state.task_id == "task-001"
    assert sample_task_state.status == "pending"
    assert len(sample_task_state.vulnerabilities) == 1


def test_benchmark_example():
    be = BenchmarkExample(
        id="bench-001",
        repo_url="https://github.com/example/repo",
        repo_name="repo",
        language="python",
        vulnerable_files=["app/views.py"],
        vuln_type=VulnType.SQL_INJECTION,
        severity=VulnSeverity.HIGH,
        ground_truth_diff="--- a\n+++ b\n",
        merge_status="merged",
        complexity_tag="single_file",
    )
    assert be.negative is False


def test_eval_result():
    er = EvalResult(
        example_id="bench-001",
        architecture="sequential",
        memory_strategy="full_context",
        detection_recall=0.9,
        detection_precision=0.85,
        triage_accuracy=0.8,
        patch_correctness=0.75,
        end_to_end_success=True,
        total_tokens=5000,
        total_cost_usd=0.05,
        latency_seconds=12.5,
        revision_loops=1,
    )
    assert er.failure_stage is None
    assert er.failure_reason is None
