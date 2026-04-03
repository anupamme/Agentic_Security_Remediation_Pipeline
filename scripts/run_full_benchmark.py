"""
Run the full benchmark matrix across all architecture × memory × orchestrator configs.

Usage:
  # Run everything (expensive!)
  python scripts/run_full_benchmark.py --config config/arch_sequential.yaml \\
      --split test --runs 3 --parallel 4

  # Run a subset (for testing)
  python scripts/run_full_benchmark.py --config config/arch_sequential.yaml \\
      --split dev --configs C1,C4,C7 --runs 1

  # Resume from checkpoint (if interrupted)
  python scripts/run_full_benchmark.py --config config/arch_sequential.yaml \\
      --split test --runs 3 --resume data/results/benchmark_test_20260101T000000/checkpoint.json
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_agent_security.config import AppConfig, load_config
from multi_agent_security.eval.runner import EvalRunner
from multi_agent_security.utils.logging import setup_logging

# The full 9-config experiment matrix
MATRIX: list[dict] = [
    {"id": "C1", "architecture": "sequential",  "memory": "full_context",   "orchestrator": "rule_based"},
    {"id": "C2", "architecture": "sequential",  "memory": "sliding_window", "orchestrator": "rule_based"},
    {"id": "C3", "architecture": "sequential",  "memory": "retrieval",      "orchestrator": "rule_based"},
    {"id": "C4", "architecture": "hub_spoke",   "memory": "full_context",   "orchestrator": "rule_based"},
    {"id": "C5", "architecture": "hub_spoke",   "memory": "full_context",   "orchestrator": "llm_based"},
    {"id": "C6", "architecture": "hub_spoke",   "memory": "sliding_window", "orchestrator": "llm_based"},
    {"id": "C7", "architecture": "blackboard",  "memory": "full_context",   "orchestrator": "rule_based"},
    {"id": "C8", "architecture": "blackboard",  "memory": "sliding_window", "orchestrator": "rule_based"},
    {"id": "C9", "architecture": "blackboard",  "memory": "retrieval",      "orchestrator": "rule_based"},
]


@dataclass
class CheckpointEntry:
    config_id: str
    run_num: int
    report_path: str


@dataclass
class Checkpoint:
    split: str
    completed: list[CheckpointEntry]
    in_progress: Optional[CheckpointEntry]
    started_at: str


_checkpoint_lock: asyncio.Lock


def _build_config_for_cell(
    base: AppConfig,
    arch: str,
    mem: str,
    orch: str,
) -> AppConfig:
    """Return a new frozen AppConfig with the three matrix dimensions applied."""
    d = base.model_dump()
    d["architecture"] = arch
    d["memory"]["strategy"] = mem
    d["orchestrator"]["type"] = orch
    return AppConfig.model_validate(d)


def _load_or_init_checkpoint(
    resume_path: Optional[str],
    split: str,
    output_dir: Path,
) -> Checkpoint:
    if resume_path:
        path = Path(resume_path)
        if path.exists():
            raw = json.loads(path.read_text())
            completed = [
                CheckpointEntry(
                    config_id=e["config"],
                    run_num=e["run"],
                    report_path=e["report_path"],
                )
                for e in raw.get("completed", [])
            ]
            print(f"Resuming from checkpoint: {len(completed)} completed run(s) skipped.")
            return Checkpoint(
                split=raw.get("split", split),
                completed=completed,
                in_progress=None,
                started_at=raw.get("started_at", datetime.now(timezone.utc).isoformat()),
            )
        else:
            print(f"WARNING: checkpoint not found at {resume_path}, starting fresh.")

    return Checkpoint(
        split=split,
        completed=[],
        in_progress=None,
        started_at=datetime.now(timezone.utc).isoformat(),
    )


def _write_checkpoint(checkpoint: Checkpoint, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    data = {
        "split": checkpoint.split,
        "completed": [
            {"config": e.config_id, "run": e.run_num, "report_path": e.report_path}
            for e in checkpoint.completed
        ],
        "in_progress": None,
        "started_at": checkpoint.started_at,
    }
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)  # atomic on POSIX


async def _append_checkpoint(
    checkpoint: Checkpoint,
    entry: CheckpointEntry,
    output_dir: Path,
) -> None:
    async with _checkpoint_lock:
        checkpoint.completed.append(entry)
        checkpoint.in_progress = None
        _write_checkpoint(checkpoint, output_dir / "checkpoint.json")


async def _run_cell(
    base_config: AppConfig,
    cell: dict,
    run_num: int,
    split: str,
    output_dir: Path,
    checkpoint: Checkpoint,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        config_id = cell["id"]
        arch = cell["architecture"]
        mem = cell["memory"]
        orch = cell["orchestrator"]
        print(
            f"[START] {config_id} run {run_num}: "
            f"arch={arch} mem={mem} orch={orch}"
        )

        cell_config = _build_config_for_cell(base_config, arch, mem, orch)
        runner = EvalRunner(cell_config, output_dir=str(output_dir))
        benchmark_dir = f"data/benchmark/{split}"

        try:
            report = await runner.run_eval(
                benchmark_dir=benchmark_dir,
                architecture=arch,
                memory_strategy=mem,
                num_runs=1,
                parallel_workers=1,
            )
        except Exception as exc:
            print(f"[ERROR] {config_id} run {run_num}: {exc}", file=sys.stderr)
            return

        # Write deterministic copy with augmented config metadata
        dest = output_dir / f"{config_id}_run{run_num}_report.json"
        report_dict = json.loads(report.model_dump_json())
        report_dict["config"]["orchestrator_type"] = orch
        report_dict["config"]["config_id"] = config_id
        dest.write_text(json.dumps(report_dict, indent=2))

        entry = CheckpointEntry(
            config_id=config_id,
            run_num=run_num,
            report_path=str(dest),
        )
        await _append_checkpoint(checkpoint, entry, output_dir)
        print(f"[DONE]  {config_id} run {run_num} -> {dest.name}")


async def run_benchmark(args: argparse.Namespace) -> None:
    global _checkpoint_lock
    _checkpoint_lock = asyncio.Lock()

    base_config = load_config(args.config)
    setup_logging(base_config.logging.level)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_or_init_checkpoint(args.resume, args.split, output_dir)

    completed_keys: set[tuple[str, int]] = {
        (e.config_id, e.run_num) for e in checkpoint.completed
    }

    # Filter matrix to requested subset
    selected_ids: Optional[set[str]] = None
    if args.configs:
        selected_ids = {c.strip() for c in args.configs.split(",")}
        unknown = selected_ids - {c["id"] for c in MATRIX}
        if unknown:
            print(f"WARNING: unknown config IDs: {unknown}", file=sys.stderr)

    semaphore = asyncio.Semaphore(args.parallel)
    tasks = []
    for cell in MATRIX:
        if selected_ids and cell["id"] not in selected_ids:
            continue
        for run_num in range(1, args.runs + 1):
            key = (cell["id"], run_num)
            if key in completed_keys:
                print(f"[SKIP]  {cell['id']} run {run_num} (already completed)")
                continue
            tasks.append(
                _run_cell(
                    base_config, cell, run_num,
                    args.split, output_dir, checkpoint, semaphore,
                )
            )

    if not tasks:
        print("All runs already completed (or no configs matched). Nothing to do.")
        return

    await asyncio.gather(*tasks)

    total = len(checkpoint.completed)
    print(f"\nBenchmark complete. {total} run(s) saved to: {output_dir}")
    print(f"Checkpoint: {output_dir / 'checkpoint.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full architecture × memory benchmark matrix"
    )
    parser.add_argument(
        "--config", required=True,
        help="Base config YAML (e.g. config/arch_sequential.yaml)",
    )
    parser.add_argument(
        "--split", default="test",
        choices=["dev", "test", "hard", "local"],
        help="Benchmark split to use (default: test)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per config (default: 3)",
    )
    parser.add_argument(
        "--parallel", type=int, default=2,
        help="Max concurrent (config, run) pairs (default: 2)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: data/results/benchmark_<split>_<ts>)",
    )
    parser.add_argument(
        "--resume", default=None, metavar="CHECKPOINT_JSON",
        help="Resume from a checkpoint.json, skipping completed pairs",
    )
    parser.add_argument(
        "--configs", default=None,
        help="Comma-separated config IDs to run (e.g. C1,C4,C7); default: all 9",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        args.output_dir = f"data/results/benchmark_{args.split}_{ts}"

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
