import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from multi_agent_security.config import AppConfig
    from multi_agent_security.types import TaskState
    from multi_agent_security.utils.cost_tracker import CostRecord, RunCostSummary


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "agent": getattr(record, "agent", "system"),
            "event": record.getMessage(),
            "data": getattr(record, "data", {}),
        }
        return json.dumps(entry)


def setup_logging(
    level: str = "INFO",
    output_file: Optional[str] = None,
) -> logging.Logger:
    """Configure root logger with console (human-readable) and optional file (JSON) handlers."""
    logger = logging.getLogger("masr")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(console)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(_JsonFormatter())
        logger.addHandler(file_handler)

    return logger


def generate_run_id(
    architecture: str,
    memory: str,
    benchmark_id: Optional[str] = None,
) -> str:
    """
    Format: {arch}_{memory}_{benchmark_id or 'adhoc'}_{timestamp}
    Example: sequential_full_context_BENCH-0001_20250701T100000

    Security: benchmark_id is sanitized to prevent path traversal attacks.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    bench = benchmark_id or "adhoc"
    # Sanitize benchmark_id: remove path separators and other problematic chars
    bench = bench.replace("/", "_").replace("\\", "_").replace(" ", "_")
    # Also remove any remaining path traversal attempts
    bench = bench.replace("..", "_")
    return f"{architecture}_{memory}_{bench}_{timestamp}"


class RunEvent(BaseModel):
    run_id: str
    timestamp: datetime
    event_type: str  # "run_start", "agent_call", "agent_error", "pipeline_step", "run_end"
    data: dict


class RunLogger:
    """Writes structured JSONL event logs for a pipeline run."""

    def __init__(self, run_id: str, output_dir: str = "data/results"):
        self.run_id = run_id
        self.file_path = Path(output_dir) / f"{run_id}.jsonl"
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: RunEvent) -> None:
        """Append a JSON line to the run log file."""
        with self.file_path.open("a") as f:
            f.write(event.model_dump_json() + "\n")

    def _make_event(self, event_type: str, data: dict) -> RunEvent:
        return RunEvent(
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            data=data,
        )

    def log_run_start(
        self,
        config: "AppConfig",
        benchmark_id: Optional[str],
    ) -> None:
        self.log_event(self._make_event("run_start", {
            "config": {
                "architecture": config.architecture,
                "memory": config.memory.strategy,
                "model": config.llm.model,
            },
            "benchmark_id": benchmark_id,
        }))

    def log_agent_call(
        self,
        agent: str,
        input_summary: str,
        output_summary: str,
        cost: "CostRecord",
    ) -> None:
        self.log_event(self._make_event("agent_call", {
            "agent": agent,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "input_tokens": cost.input_tokens,
            "output_tokens": cost.output_tokens,
            "cost_usd": cost.cost_usd,
            "latency_ms": cost.latency_ms,
            "model": cost.model,
        }))

    def log_agent_error(self, agent: str, error: str, traceback: str) -> None:
        self.log_event(self._make_event("agent_error", {
            "agent": agent,
            "error": error,
            "traceback": traceback,
        }))

    def log_pipeline_step(self, step: str, status: str, details: dict) -> None:
        self.log_event(self._make_event("pipeline_step", {
            "step": step,
            "status": status,
            **details,
        }))

    def log_run_end(
        self,
        task_state: "TaskState",
        cost_summary: "RunCostSummary",
    ) -> None:
        accepted = [r for r in task_state.reviews if r.patch_accepted]
        self.log_event(self._make_event("run_end", {
            "status": task_state.status,
            "total_cost_usd": cost_summary.total_cost_usd,
            "total_input_tokens": cost_summary.total_input_tokens,
            "total_output_tokens": cost_summary.total_output_tokens,
            "total_tokens": cost_summary.total_input_tokens + cost_summary.total_output_tokens,
            "total_latency_ms": cost_summary.total_latency_ms,
            "vulns_detected": len(task_state.vulnerabilities),
            "patches_accepted": len(accepted),
            "revision_loops": task_state.revision_count,
            "e2e_success": bool(accepted),
        }))


class RunReporter:
    """Loads a completed run from its JSONL log and prints a summary."""

    def __init__(self, events: list[RunEvent]):
        self._events = events

    @staticmethod
    def from_jsonl(file_path: str) -> "RunReporter":
        events: list[RunEvent] = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(RunEvent.model_validate_json(line))
        return RunReporter(events)

    def _run_start(self) -> Optional[RunEvent]:
        return next((e for e in self._events if e.event_type == "run_start"), None)

    def _run_end(self) -> Optional[RunEvent]:
        return next((e for e in self._events if e.event_type == "run_end"), None)

    def _agent_calls(self) -> list[RunEvent]:
        return [e for e in self._events if e.event_type == "agent_call"]

    def to_dict(self) -> dict:
        start = self._run_start()
        end = self._run_end()
        calls = self._agent_calls()

        per_agent: dict[str, dict] = {}
        for ev in calls:
            agent = ev.data.get("agent", "unknown")
            if agent not in per_agent:
                per_agent[agent] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_ms": 0.0,
                }
            per_agent[agent]["calls"] += 1
            per_agent[agent]["input_tokens"] += ev.data.get("input_tokens", 0)
            per_agent[agent]["output_tokens"] += ev.data.get("output_tokens", 0)
            per_agent[agent]["cost_usd"] += ev.data.get("cost_usd", 0.0)
            per_agent[agent]["latency_ms"] += ev.data.get("latency_ms", 0.0)

        return {
            "run_id": self._events[0].run_id if self._events else None,
            "config": start.data.get("config") if start else {},
            "status": end.data.get("status") if end else "unknown",
            "total_cost_usd": end.data.get("total_cost_usd") if end else None,
            "total_tokens": end.data.get("total_tokens") if end else None,
            "e2e_success": end.data.get("e2e_success") if end else None,
            "per_agent": per_agent,
        }

    def print_summary(self) -> None:
        d = self.to_dict()
        run_id = d["run_id"] or "unknown"
        config = d.get("config") or {}
        arch = config.get("architecture", "?")
        mem = config.get("memory", "?")
        status = d.get("status", "unknown")
        e2e = "SUCCESS" if d.get("e2e_success") else "PARTIAL/FAIL"
        total_cost = d.get("total_cost_usd") or 0.0
        total_tokens = d.get("total_tokens") or 0

        per_agent = d.get("per_agent") or {}
        total_calls = sum(v["calls"] for v in per_agent.values())
        total_in = sum(v["input_tokens"] for v in per_agent.values())
        total_out = sum(v["output_tokens"] for v in per_agent.values())

        width = 67
        print("=" * width)
        print(f"Run: {run_id}")
        print(f"Architecture: {arch} | Memory: {mem}")
        print(f"Status: {status} | End-to-end: {e2e}")
        print(f"Total cost: ${total_cost:.4f} | Total tokens: {total_tokens:,}")
        print("=" * width)
        print()
        print("Agent Breakdown:")
        col = (14, 7, 10, 10, 10)
        header = (
            f"{'Agent':<{col[0]}}{'Calls':>{col[1]}}"
            f"{'In Tokens':>{col[2]}}{'Out Tokens':>{col[3]}}{'Cost ($)':>{col[4]}}"
        )
        sep = (
            f"{'-' * col[0]}"
            f"{'-' * col[1]}{'-' * col[2]}{'-' * col[3]}{'-' * col[4]}"
        )
        print(header)
        print(sep)
        for agent, vals in sorted(per_agent.items()):
            print(
                f"{agent:<{col[0]}}{vals['calls']:>{col[1]}}"
                f"{vals['input_tokens']:>{col[2]},}{vals['output_tokens']:>{col[3]},}"
                f"{vals['cost_usd']:>{col[4]}.4f}"
            )
        print(sep)
        print(
            f"{'TOTAL':<{col[0]}}{total_calls:>{col[1]}}"
            f"{total_in:>{col[2]},}{total_out:>{col[3]},}"
            f"{total_cost:>{col[4]}.4f}"
        )
        print()
