"""Tests for Issue #10: observability — token tracking, cost accounting, and run logging."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from multi_agent_security.utils.cost_tracker import (
    PRICING,
    AgentCostSummary,
    CostRecord,
    CostTracker,
    RunCostSummary,
)
from multi_agent_security.utils.logging import (
    RunEvent,
    RunLogger,
    RunReporter,
    generate_run_id,
)


# ---------------------------------------------------------------------------
# CostTracker tests
# ---------------------------------------------------------------------------


def test_cost_calculation_known_tokens():
    tracker = CostTracker()
    model = "claude-sonnet-4-20250514"
    input_tokens = 1_000_000
    output_tokens = 1_000_000
    rec = tracker.record("scanner", model, input_tokens, output_tokens, latency_ms=100.0)

    pricing = PRICING[model]
    expected = input_tokens * pricing["input"] + output_tokens * pricing["output"]
    assert abs(rec.cost_usd - expected) < 1e-9
    assert rec.cost_usd == pytest.approx(3.0 + 15.0)


def test_cost_calculation_haiku_model():
    tracker = CostTracker()
    model = "claude-haiku-3-5-20241022"
    rec = tracker.record("triager", model, 500_000, 200_000, latency_ms=50.0)

    pricing = PRICING[model]
    expected = 500_000 * pricing["input"] + 200_000 * pricing["output"]
    assert rec.cost_usd == pytest.approx(expected)


def test_cost_tracker_unknown_model_falls_back():
    tracker = CostTracker()
    rec = tracker.record("patcher", "unknown-future-model", 100, 50, latency_ms=10.0)
    # Should not raise; uses default pricing
    default = PRICING["default"]
    expected = 100 * default["input"] + 50 * default["output"]
    assert rec.cost_usd == pytest.approx(expected)


def test_per_agent_sums_to_total():
    tracker = CostTracker()
    model = "claude-sonnet-4-20250514"
    tracker.record("scanner", model, 1000, 200, latency_ms=300.0)
    tracker.record("scanner", model, 800, 150, latency_ms=250.0)
    tracker.record("triager", model, 400, 100, latency_ms=100.0)
    tracker.record("patcher", model, 600, 300, latency_ms=200.0)

    per_agent = tracker.get_per_agent_summary()
    total = tracker.get_total_summary()

    assert sum(s.calls for s in per_agent.values()) == total.total_calls
    assert sum(s.input_tokens for s in per_agent.values()) == total.total_input_tokens
    assert sum(s.output_tokens for s in per_agent.values()) == total.total_output_tokens
    assert sum(s.cost_usd for s in per_agent.values()) == pytest.approx(total.total_cost_usd)
    assert sum(s.total_latency_ms for s in per_agent.values()) == pytest.approx(total.total_latency_ms)


def test_per_agent_call_counts():
    tracker = CostTracker()
    model = "claude-sonnet-4-20250514"
    for _ in range(3):
        tracker.record("scanner", model, 100, 50, latency_ms=100.0)
    tracker.record("triager", model, 200, 80, latency_ms=150.0)

    per_agent = tracker.get_per_agent_summary()
    assert per_agent["scanner"].calls == 3
    assert per_agent["triager"].calls == 1


def test_per_agent_avg_latency():
    tracker = CostTracker()
    model = "claude-sonnet-4-20250514"
    tracker.record("scanner", model, 100, 50, latency_ms=100.0)
    tracker.record("scanner", model, 100, 50, latency_ms=300.0)

    per_agent = tracker.get_per_agent_summary()
    assert per_agent["scanner"].avg_latency_ms == pytest.approx(200.0)


def test_cost_tracker_reset():
    tracker = CostTracker()
    tracker.record("scanner", "claude-sonnet-4-20250514", 100, 50, latency_ms=100.0)
    tracker.reset()
    assert tracker.get_records() == []
    total = tracker.get_total_summary()
    assert total.total_calls == 0
    assert total.total_cost_usd == 0.0


def test_get_summary_backward_compat():
    tracker = CostTracker()
    tracker.record("scanner", "claude-sonnet-4-20250514", 100, 50, latency_ms=100.0)
    summary = tracker.get_summary()
    assert isinstance(summary, dict)
    assert "total_calls" in summary
    assert "per_agent" in summary


# ---------------------------------------------------------------------------
# generate_run_id tests
# ---------------------------------------------------------------------------


def test_generate_run_id_format_with_benchmark():
    run_id = generate_run_id("sequential", "full_context", benchmark_id="BENCH-0001")
    parts = run_id.split("_")
    # sequential_full_context_BENCH-0001_20250701T100000
    assert parts[0] == "sequential"
    assert parts[1] == "full"
    assert parts[2] == "context"
    assert "BENCH-0001" in run_id
    # Last part should be a timestamp like 20250701T100000
    assert len(parts[-1]) == 15  # YYYYMMDDTHHmmss


def test_generate_run_id_adhoc_when_no_benchmark():
    run_id = generate_run_id("hub_spoke", "sliding_window")
    assert "adhoc" in run_id
    assert run_id.startswith("hub_spoke_sliding_window_adhoc_")


def test_generate_run_id_unique():
    import time
    id1 = generate_run_id("sequential", "full_context")
    time.sleep(1.1)
    id2 = generate_run_id("sequential", "full_context")
    assert id1 != id2


def test_generate_run_id_sanitizes_path_traversal():
    """Verify that path traversal attempts in benchmark_id are sanitized."""
    # Test forward slash
    run_id = generate_run_id("sequential", "full_context", benchmark_id="../../../etc/passwd")
    assert "../" not in run_id
    assert "/" not in run_id
    assert "etc_passwd" in run_id

    # Test backslash (Windows)
    run_id = generate_run_id("sequential", "full_context", benchmark_id="..\\..\\..\\windows\\system32")
    assert "..\\" not in run_id
    assert "\\" not in run_id
    assert "windows_system32" in run_id

    # Test mixed
    run_id = generate_run_id("sequential", "full_context", benchmark_id="../../tmp/../evil")
    assert ".." not in run_id
    assert "/" not in run_id
    assert run_id.startswith("sequential_full_context_")
    assert run_id.count("_") >= 6  # arch_mem_sanitized_parts_timestamp


# ---------------------------------------------------------------------------
# RunLogger tests
# ---------------------------------------------------------------------------


def test_run_logger_produces_valid_jsonl(tmp_path):
    run_id = "test-run-001"
    logger = RunLogger(run_id, output_dir=str(tmp_path))

    # Log a variety of event types
    logger.log_pipeline_step("scan", "started", {"files": 5})
    logger.log_pipeline_step("scan", "complete", {"vulns": 2})
    logger.log_agent_error("patcher", "ValueError: bad input", "Traceback...")
    logger.log_pipeline_step("patch", "complete", {"patches": 1})

    lines = logger.file_path.read_text().strip().split("\n")
    assert len(lines) == 4
    for line in lines:
        obj = json.loads(line)
        assert "run_id" in obj
        assert "timestamp" in obj
        assert "event_type" in obj
        assert "data" in obj
        assert obj["run_id"] == run_id


def test_run_logger_event_types(tmp_path):
    run_id = "test-run-002"
    logger = RunLogger(run_id, output_dir=str(tmp_path))

    tracker = CostTracker()
    rec = tracker.record("scanner", "claude-sonnet-4-20250514", 100, 50, latency_ms=100.0)

    logger.log_agent_call("scanner", "100 tokens in", "50 tokens out", rec)
    logger.log_agent_error("triager", "timeout", "stack trace here")
    logger.log_pipeline_step("triage", "failed", {"reason": "timeout"})

    events = [json.loads(l) for l in logger.file_path.read_text().strip().split("\n")]
    types = [e["event_type"] for e in events]
    assert "agent_call" in types
    assert "agent_error" in types
    assert "pipeline_step" in types


def test_run_logger_agent_call_data(tmp_path):
    run_id = "test-run-003"
    logger = RunLogger(run_id, output_dir=str(tmp_path))

    tracker = CostTracker()
    rec = tracker.record("patcher", "claude-sonnet-4-20250514", 500, 200, latency_ms=250.0)

    logger.log_agent_call("patcher", "patch attempt", "patch result", rec)

    event = json.loads(logger.file_path.read_text().strip())
    assert event["event_type"] == "agent_call"
    data = event["data"]
    assert data["agent"] == "patcher"
    assert data["input_tokens"] == 500
    assert data["output_tokens"] == 200
    assert data["latency_ms"] == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# RunReporter tests
# ---------------------------------------------------------------------------


def _write_sample_jsonl(tmp_path: Path) -> Path:
    run_id = "seq_full_context_BENCH-0001_20250701T100000"
    events = [
        {
            "run_id": run_id,
            "timestamp": "2025-07-01T10:00:00+00:00",
            "event_type": "run_start",
            "data": {
                "config": {"architecture": "sequential", "memory": "full_context", "model": "claude-sonnet-4-20250514"},
                "benchmark_id": "BENCH-0001",
            },
        },
        {
            "run_id": run_id,
            "timestamp": "2025-07-01T10:00:02+00:00",
            "event_type": "agent_call",
            "data": {
                "agent": "scanner",
                "input_summary": "3500 tokens",
                "output_summary": "800 tokens",
                "input_tokens": 3500,
                "output_tokens": 800,
                "cost_usd": 0.0225,
                "latency_ms": 2100.0,
                "model": "claude-sonnet-4-20250514",
            },
        },
        {
            "run_id": run_id,
            "timestamp": "2025-07-01T10:00:05+00:00",
            "event_type": "agent_call",
            "data": {
                "agent": "triager",
                "input_summary": "1200 tokens",
                "output_summary": "600 tokens",
                "input_tokens": 1200,
                "output_tokens": 600,
                "cost_usd": 0.0126,
                "latency_ms": 1800.0,
                "model": "claude-sonnet-4-20250514",
            },
        },
        {
            "run_id": run_id,
            "timestamp": "2025-07-01T10:00:30+00:00",
            "event_type": "run_end",
            "data": {
                "status": "complete",
                "total_cost_usd": 0.0351,
                "total_input_tokens": 4700,
                "total_output_tokens": 1400,
                "total_tokens": 6100,
                "total_latency_ms": 3900.0,
                "vulns_detected": 2,
                "patches_accepted": 1,
                "revision_loops": 0,
                "e2e_success": True,
            },
        },
    ]
    path = tmp_path / "sample_run.jsonl"
    with path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return path


def test_run_reporter_loads_from_jsonl(tmp_path):
    path = _write_sample_jsonl(tmp_path)
    reporter = RunReporter.from_jsonl(str(path))
    d = reporter.to_dict()

    assert d["run_id"] == "seq_full_context_BENCH-0001_20250701T100000"
    assert d["status"] == "complete"
    assert d["e2e_success"] is True
    assert d["total_cost_usd"] == pytest.approx(0.0351)
    assert d["total_tokens"] == 6100


def test_run_reporter_per_agent_breakdown(tmp_path):
    path = _write_sample_jsonl(tmp_path)
    reporter = RunReporter.from_jsonl(str(path))
    d = reporter.to_dict()

    per_agent = d["per_agent"]
    assert "scanner" in per_agent
    assert "triager" in per_agent
    assert per_agent["scanner"]["calls"] == 1
    assert per_agent["scanner"]["input_tokens"] == 3500
    assert per_agent["triager"]["output_tokens"] == 600


def test_run_reporter_print_summary_runs_without_error(tmp_path, capsys):
    path = _write_sample_jsonl(tmp_path)
    reporter = RunReporter.from_jsonl(str(path))
    reporter.print_summary()
    captured = capsys.readouterr()
    assert "scanner" in captured.out
    assert "triager" in captured.out
    assert "TOTAL" in captured.out
