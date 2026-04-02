"""Unit tests for LLMJudge and calibrate_judge."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from multi_agent_security.eval.judge import LLMJudge, calibrate_judge
from multi_agent_security.llm_client import LLMResponse
from multi_agent_security.types import (
    BenchmarkExample,
    FixStrategy,
    JudgeScore,
    Patch,
    TriageResult,
    Vulnerability,
    VulnSeverity,
    VulnType,
)


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=100,
        output_tokens=50,
        latency_ms=10.0,
        cost_usd=0.001,
        model="claude-sonnet-4-20250514",
    )


def _make_vulnerability() -> Vulnerability:
    return Vulnerability(
        id="VULN-001",
        file_path="app.py",
        line_start=10,
        line_end=12,
        vuln_type=VulnType.SQL_INJECTION,
        description="SQL injection via f-string",
        confidence=0.95,
        code_snippet='cursor.execute(f"SELECT * FROM users WHERE id={id}")',
        scanner_reasoning="Unsanitised input in SQL",
    )


def _make_patch() -> Patch:
    return Patch(
        vuln_id="VULN-001",
        file_path="app.py",
        original_code="original",
        patched_code="patched",
        unified_diff="+cursor.execute('SELECT * FROM users WHERE id=?', (id,))\n",
        patch_reasoning="Use parameterized query",
    )


def _make_benchmark_example(example_id: str = "BENCH-001") -> BenchmarkExample:
    return BenchmarkExample(
        id=example_id,
        repo_url="https://github.com/example/repo",
        repo_name="repo",
        language="python",
        vulnerable_files=["app.py"],
        vuln_type=VulnType.SQL_INJECTION,
        severity=VulnSeverity.HIGH,
        ground_truth_diff="+cursor.execute('SELECT * FROM users WHERE id=?', (id,))\n",
        merge_status="merged",
        complexity_tag="single_file",
    )


@pytest.mark.asyncio
async def test_judge_patch_correctness_parsing():
    """Mock LLM returns valid JSON → JudgeScore parsed correctly."""
    mock_llm = MagicMock()
    mock_response_content = json.dumps({
        "correctness": 0.85,
        "completeness": 0.90,
        "safety": 0.95,
        "reasoning": "Good parameterized query fix.",
    })
    mock_llm.complete = AsyncMock(return_value=_make_llm_response(mock_response_content))

    judge = LLMJudge(mock_llm)
    score = await judge.judge_patch_correctness(
        vulnerability=_make_vulnerability(),
        generated_patch=_make_patch(),
        ground_truth_diff="+cursor.execute('SELECT * FROM users WHERE id=?', (id,))\n",
        language="python",
    )

    assert isinstance(score, JudgeScore)
    assert abs(score.correctness - 0.85) < 0.001
    assert abs(score.completeness - 0.90) < 0.001
    assert abs(score.safety - 0.95) < 0.001
    assert score.reasoning == "Good parameterized query fix."
    mock_llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_judge_patch_quality_parsing():
    """Mock LLM returns valid JSON for quality evaluation → JudgeScore parsed correctly."""
    mock_llm = MagicMock()
    mock_response_content = json.dumps({
        "correctness": 0.7,
        "completeness": 0.8,
        "safety": 1.0,
        "reasoning": "Patch follows best practices.",
    })
    mock_llm.complete = AsyncMock(return_value=_make_llm_response(mock_response_content))

    judge = LLMJudge(mock_llm)
    score = await judge.judge_patch_quality(
        vulnerability=_make_vulnerability(),
        patch=_make_patch(),
        original_code='cursor.execute(f"SELECT * FROM users WHERE id={id}")',
        language="python",
    )

    assert isinstance(score, JudgeScore)
    assert abs(score.correctness - 0.7) < 0.001
    assert abs(score.safety - 1.0) < 0.001
    mock_llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_calibrate_judge_correlation():
    """Calibration with mock data computes correlation correctly."""
    # Create mock human scores and ensure judge returns known values
    human_scores = [0.2, 0.4, 0.6, 0.8, 1.0]
    judge_returns = [0.25, 0.35, 0.65, 0.75, 0.95]  # Highly correlated with human

    call_count = 0

    async def mock_complete(*args, **kwargs):
        nonlocal call_count
        score = judge_returns[call_count % len(judge_returns)]
        call_count += 1
        content = json.dumps({
            "correctness": score,
            "completeness": score,
            "safety": 1.0,
            "reasoning": "mock",
        })
        return _make_llm_response(content)

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=mock_complete)

    judge = LLMJudge(mock_llm)

    examples = [
        (_make_benchmark_example(f"BENCH-{i:03d}"), _make_patch(), h)
        for i, h in enumerate(human_scores)
    ]

    result = await calibrate_judge(judge, examples)

    assert result.n_examples == 5
    assert -1.0 <= result.pearson_r <= 1.0
    assert -1.0 <= result.spearman_r <= 1.0
    assert result.mean_absolute_error >= 0.0
    assert len(result.per_example) == 5
    # Given high correlation between human and judge scores, pearson_r should be high
    assert result.pearson_r > 0.9
    # Check per_example structure
    for entry in result.per_example:
        assert "example_id" in entry
        assert "human_score" in entry
        assert "judge_score" in entry
        assert "delta" in entry
        assert entry["delta"] >= 0
