"""Unit tests for generate_comparison.py: Pareto frontier, significance testing,
and table generation."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from multi_agent_security.eval.metrics import aggregate_metrics
from multi_agent_security.types import EvalResult

# Import functions under test from generate_comparison script
from generate_comparison import (
    ConfigSummary,
    compute_pareto_frontier,
    compute_significance_matrix,
    write_comparison_table,
    write_complexity_breakdown,
    write_failure_stages,
    write_vuln_type_breakdown,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_eval_result(
    example_id: str,
    e2e: bool,
    arch: str = "sequential",
    mem: str = "full_context",
    complexity_tag: str = "single_file",
    vuln_type: str = "SQL_INJECTION",
) -> EvalResult:
    return EvalResult(
        example_id=example_id,
        architecture=arch,
        memory_strategy=mem,
        detection_recall=1.0 if e2e else 0.0,
        detection_precision=1.0 if e2e else 0.0,
        triage_accuracy=1.0 if e2e else 0.0,
        patch_correctness=1.0 if e2e else 0.0,
        end_to_end_success=e2e,
        total_tokens=1000,
        total_cost_usd=0.01,
        latency_seconds=5.0,
        revision_loops=0,
        complexity_tag=complexity_tag,
        vuln_type=vuln_type,
    )


def _make_summary(
    config_id: str,
    results: list[EvalResult],
    arch: str = "sequential",
    mem: str = "full_context",
    orch: str = "rule_based",
) -> ConfigSummary:
    n = len(results)
    e2e_rate = sum(r.end_to_end_success for r in results) / n if n else 0.0
    stage_counts: dict[str, int] = {}
    for r in results:
        if not r.end_to_end_success and r.failure_stage:
            stage_counts[r.failure_stage] = stage_counts.get(r.failure_stage, 0) + 1
    return ConfigSummary(
        config_id=config_id,
        architecture=arch,
        memory_strategy=mem,
        orchestrator_type=orch,
        num_runs=1,
        all_results=results,
        agg=aggregate_metrics(results),
        per_run_e2e=[e2e_rate],
        failure_stage_counts=stage_counts,
        failure_stage_total=sum(stage_counts.values()),
    )


# ---------------------------------------------------------------------------
# Pareto frontier tests
# ---------------------------------------------------------------------------

def test_pareto_frontier_known_points():
    """Known 5-point layout: indices 0, 1, 4 form the Pareto frontier."""
    points = [
        (0.10, 0.90),  # 0 — best success
        (0.05, 0.80),  # 1 — cheapest
        (0.20, 0.70),  # 2 — dominated by 0
        (0.15, 0.60),  # 3 — dominated by 0 and 1
        (0.08, 0.85),  # 4 — on frontier between 0 and 1
    ]
    assert set(compute_pareto_frontier(points)) == {0, 1, 4}


def test_pareto_frontier_single_point():
    assert compute_pareto_frontier([(0.1, 0.9)]) == [0]


def test_pareto_frontier_all_on_frontier():
    """Strictly monotone trade-off: each point is Pareto optimal."""
    points = [(0.01, 0.5), (0.05, 0.7), (0.10, 0.9)]
    assert set(compute_pareto_frontier(points)) == {0, 1, 2}


def test_pareto_frontier_all_dominated_except_one():
    """One point dominates all others on both dimensions."""
    points = [(0.01, 1.0), (0.05, 0.8), (0.10, 0.6), (0.20, 0.4)]
    assert compute_pareto_frontier(points) == [0]


def test_pareto_frontier_identical_points():
    """Identical points: neither dominates the other (strict inequality fails)."""
    points = [(0.1, 0.8), (0.1, 0.8)]
    assert set(compute_pareto_frontier(points)) == {0, 1}


def test_pareto_frontier_empty():
    assert compute_pareto_frontier([]) == []


# ---------------------------------------------------------------------------
# Statistical significance tests
# ---------------------------------------------------------------------------

def test_significance_identical_configs_not_significant():
    """Same per-example E2E outcomes → all diffs zero → not significant."""
    ids = [f"ex-{i:02d}" for i in range(20)]
    outcomes = [i % 2 == 0 for i in range(20)]  # alternating T/F
    results_a = [_make_eval_result(eid, e2e) for eid, e2e in zip(ids, outcomes)]
    results_b = [_make_eval_result(eid, e2e) for eid, e2e in zip(ids, outcomes)]
    summaries = {
        "CA": _make_summary("CA", results_a),
        "CB": _make_summary("CB", results_b),
    }
    sig_matrix = compute_significance_matrix(summaries, alpha=0.05)
    is_sig, p_val = sig_matrix[("CA", "CB")]
    assert not is_sig
    assert p_val == 1.0


def test_significance_very_different_configs_significant():
    """Config A always succeeds, B always fails → should be significant."""
    ids = [f"ex-{i:02d}" for i in range(20)]
    results_a = [_make_eval_result(eid, True) for eid in ids]
    results_b = [_make_eval_result(eid, False) for eid in ids]
    summaries = {
        "CA": _make_summary("CA", results_a),
        "CB": _make_summary("CB", results_b),
    }
    sig_matrix = compute_significance_matrix(summaries, alpha=0.05)
    is_sig, p_val = sig_matrix[("CA", "CB")]
    assert is_sig
    assert p_val < 0.05


def test_significance_below_minimum_pairs_not_significant():
    """Fewer than 10 shared examples → returns (False, 1.0) gracefully."""
    ids = [f"ex-{i:02d}" for i in range(5)]
    results_a = [_make_eval_result(eid, True) for eid in ids]
    results_b = [_make_eval_result(eid, False) for eid in ids]
    summaries = {
        "CA": _make_summary("CA", results_a),
        "CB": _make_summary("CB", results_b),
    }
    sig_matrix = compute_significance_matrix(summaries, alpha=0.05)
    is_sig, _ = sig_matrix[("CA", "CB")]
    assert not is_sig


def test_significance_no_shared_examples():
    """No overlapping example IDs → returns (False, 1.0) gracefully."""
    results_a = [_make_eval_result(f"a-{i}", True) for i in range(15)]
    results_b = [_make_eval_result(f"b-{i}", False) for i in range(15)]
    summaries = {
        "CA": _make_summary("CA", results_a),
        "CB": _make_summary("CB", results_b),
    }
    sig_matrix = compute_significance_matrix(summaries, alpha=0.05)
    is_sig, _ = sig_matrix[("CA", "CB")]
    assert not is_sig


def test_significance_multi_run_averages_not_overwritten():
    """
    With 3 runs per example, per-example success rates must be averaged across
    runs rather than the last run overwriting previous ones.

    Config A: 2 of 3 runs succeed for every example → mean = 0.667
    Config B: always fails → mean = 0.0
    The averaged signal should still be detected as significant, confirming all
    run data contributes rather than just the final run.
    """
    ids = [f"ex-{i:02d}" for i in range(20)]
    results_a = []
    for eid in ids:
        results_a.append(_make_eval_result(eid, True))   # run 1: success
        results_a.append(_make_eval_result(eid, True))   # run 2: success
        results_a.append(_make_eval_result(eid, False))  # run 3: failure
    results_b = [_make_eval_result(eid, False) for eid in ids for _ in range(3)]

    summaries = {
        "CA": _make_summary("CA", results_a),
        "CB": _make_summary("CB", results_b),
    }
    sig_matrix = compute_significance_matrix(summaries, alpha=0.05)
    is_sig, p_val = sig_matrix[("CA", "CB")]
    # Averaged A = 0.667, B = 0.0 for all examples — should be significant
    assert is_sig
    assert p_val < 0.05


# ---------------------------------------------------------------------------
# Table generation tests
# ---------------------------------------------------------------------------

def test_comparison_table_contains_all_config_ids(tmp_path):
    """All config IDs appear in the generated markdown."""
    ids = ["CA", "CB", "CC"]
    summaries = {}
    for cid in ids:
        results = [_make_eval_result(f"ex-{i:02d}", i % 2 == 0) for i in range(10)]
        summaries[cid] = _make_summary(cid, results)

    sig = compute_significance_matrix(summaries)
    write_comparison_table(summaries, tmp_path, sig)

    table = (tmp_path / "comparison_table.md").read_text()
    for cid in ids:
        assert cid in table


def test_comparison_table_correct_e2e_format(tmp_path):
    """All-success config → E2E = 1.000 in the table."""
    results = [_make_eval_result(f"ex-{i:02d}", True) for i in range(10)]
    summaries = {"CA": _make_summary("CA", results)}
    sig = compute_significance_matrix(summaries)
    write_comparison_table(summaries, tmp_path, sig)
    table = (tmp_path / "comparison_table.md").read_text()
    assert "1.000" in table


def test_comparison_table_is_valid_markdown(tmp_path):
    """Output contains header row, separator row, and at least one data row."""
    results = [_make_eval_result(f"ex-{i:02d}", i < 5) for i in range(10)]
    summaries = {"CA": _make_summary("CA", results)}
    sig = compute_significance_matrix(summaries)
    write_comparison_table(summaries, tmp_path, sig)
    table = (tmp_path / "comparison_table.md").read_text()
    lines = [line for line in table.splitlines() if line.strip()]
    # Separator row has at least one |---|
    separator_lines = [line for line in lines if "|---" in line or "---|" in line]
    assert len(separator_lines) >= 1
    assert "Config" in table
    assert "E2E" in table


def test_complexity_breakdown_written(tmp_path):
    """Complexity breakdown file is created with section headers."""
    results = [
        _make_eval_result(f"ex-{i:02d}", i < 5, complexity_tag="single_file")
        for i in range(10)
    ] + [
        _make_eval_result(f"ex-{i+10:02d}", i < 3, complexity_tag="multi_file")
        for i in range(6)
    ]
    summaries = {"CA": _make_summary("CA", results)}
    write_complexity_breakdown(summaries, tmp_path)
    text = (tmp_path / "complexity_breakdown.md").read_text()
    assert "single_file" in text
    assert "multi_file" in text


def test_failure_stages_written(tmp_path):
    """Failure stages file is created and contains all stage names."""
    results = [_make_eval_result(f"ex-{i:02d}", False) for i in range(10)]
    # Assign failure stages manually
    for i, r in enumerate(results):
        object.__setattr__(r, "failure_stage", ["scanner", "patcher"][i % 2])
    summaries = {"CA": _make_summary("CA", results)}
    write_failure_stages(summaries, tmp_path)
    text = (tmp_path / "failure_stages.md").read_text()
    assert "scanner" in text
    assert "patcher" in text
    assert "CA" in text
