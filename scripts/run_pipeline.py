"""
Pipeline runner for the Multi-Agent Security Remediation Pipeline.

Usage:
  # Single repo
  python scripts/run_pipeline.py --repo /path/to/repo --config config/arch_sequential.yaml

  # Single benchmark example
  python scripts/run_pipeline.py --benchmark data/benchmark/dev/BENCH-0001.json \
      --config config/arch_sequential.yaml

  # Batch benchmark directory
  python scripts/run_pipeline.py --benchmark-dir data/benchmark/dev/ \
      --config config/arch_sequential.yaml --parallel 4 --output-dir data/results/
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

# Ensure src/ is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_agent_security.agents.patcher import PatcherAgent
from multi_agent_security.agents.reviewer import ReviewerAgent
from multi_agent_security.agents.scanner import ScannerAgent
from multi_agent_security.agents.triager import TriagerAgent
from multi_agent_security.config import load_config
from multi_agent_security.llm_client import LLMClient
from multi_agent_security.memory.full_context import FullContextMemory
from multi_agent_security.orchestration.blackboard import BlackboardOrchestrator
from multi_agent_security.orchestration.hub_spoke import HubSpokeOrchestrator
from multi_agent_security.orchestration.sequential import SequentialOrchestrator
from multi_agent_security.tools.repo_cloner import RepoCloner
from multi_agent_security.types import BenchmarkExample, EvalResult, TaskState
from multi_agent_security.utils.logging import setup_logging

_ORCHESTRATORS = {
    "sequential": SequentialOrchestrator,
    "hub_spoke": HubSpokeOrchestrator,
    "blackboard": BlackboardOrchestrator,
}


def _build_orchestrator(config):
    llm_client = LLMClient(config.llm)
    agents = {
        "scanner": ScannerAgent(config, llm_client),
        "triager": TriagerAgent(config, llm_client),
        "patcher": PatcherAgent(config, llm_client),
        "reviewer": ReviewerAgent(config, llm_client),
    }
    memory = FullContextMemory()
    orch_class = _ORCHESTRATORS.get(config.architecture)
    if orch_class is None:
        raise ValueError(
            f"Unknown architecture '{config.architecture}'. "
            f"Valid options: {list(_ORCHESTRATORS)}"
        )
    return orch_class(config, agents, memory)


async def run_single_repo(args) -> TaskState:
    config = load_config(args.config)
    setup_logging(config.logging.level)

    orchestrator = _build_orchestrator(config)

    repo_path = args.repo
    if repo_path.startswith("http://") or repo_path.startswith("https://"):
        cloner = RepoCloner()
        repo_path = cloner.clone(args.repo, timeout=args.clone_timeout)
        if repo_path is None:
            print(
                f"ERROR: Could not clone {args.repo}. "
                f"Check the URL, your network connection, and consider raising "
                f"--clone-timeout (current: {args.clone_timeout}s).",
                file=sys.stderr,
            )
            sys.exit(1)

    task_state = TaskState(
        task_id=str(uuid.uuid4()),
        repo_url=args.repo,
        language=args.language,
    )
    result = await orchestrator.run(repo_path, task_state)
    print(result.model_dump_json(indent=2))
    return result


async def run_single_benchmark(args) -> EvalResult:
    import time

    config = load_config(args.config)
    setup_logging(config.logging.level)

    with open(args.benchmark) as f:
        example_data = json.load(f)
    example = BenchmarkExample.model_validate(example_data)

    orchestrator = _build_orchestrator(config)

    cloner = RepoCloner()
    repo_path = cloner.clone(example.repo_url)

    task_state = TaskState(
        task_id=str(uuid.uuid4()),
        repo_url=example.repo_url,
        language=example.language,
    )

    start = time.monotonic()
    result = await orchestrator.run(repo_path, task_state)
    elapsed = time.monotonic() - start

    total_tokens = sum(
        m.token_count_input + m.token_count_output for m in result.messages
    )
    total_cost = sum(m.cost_usd for m in result.messages)

    # Basic detection metrics: did we find the expected vuln type?
    found_types = {v.vuln_type for v in result.vulnerabilities}
    expected_type = example.vuln_type
    detection_recall = 1.0 if expected_type in found_types else 0.0
    # Precision = TP / (TP + FP): if the expected type was found, TP=1 and
    # FP = number of other distinct types reported; otherwise precision = 0.
    detection_precision = (
        1.0 / len(found_types) if expected_type in found_types else 0.0
    )

    accepted_patches = [r for r in result.reviews if r.patch_accepted]
    end_to_end_success = bool(accepted_patches)
    patch_correctness = (
        sum(r.correctness_score for r in accepted_patches) / len(accepted_patches)
        if accepted_patches
        else 0.0
    )

    eval_result = EvalResult(
        example_id=example.id,
        architecture=config.architecture,
        memory_strategy=config.memory.strategy,
        detection_recall=detection_recall,
        detection_precision=detection_precision,
        triage_accuracy=0.0,  # Requires ground-truth comparison beyond scope here
        patch_correctness=patch_correctness,
        end_to_end_success=end_to_end_success,
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        latency_seconds=elapsed,
        revision_loops=result.revision_count,
        failure_stage=result.status if result.status == "failed" else None,
    )

    print(eval_result.model_dump_json(indent=2))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{example.id}.json"
        out_file.write_text(eval_result.model_dump_json(indent=2))

    return eval_result


async def run_benchmark_dir(args) -> list[EvalResult]:
    benchmark_dir = Path(args.benchmark_dir)
    json_files = sorted(benchmark_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {benchmark_dir}", file=sys.stderr)
        return []

    semaphore = asyncio.Semaphore(args.parallel)

    async def _run_one(json_path: Path) -> EvalResult | None:
        async with semaphore:
            sub_args = argparse.Namespace(
                benchmark=str(json_path),
                config=args.config,
                output_dir=args.output_dir,
            )
            try:
                return await run_single_benchmark(sub_args)
            except Exception as exc:
                print(f"ERROR processing {json_path.name}: {exc}", file=sys.stderr)
                return None

    results = await asyncio.gather(*(_run_one(p) for p in json_files))
    successful = [r for r in results if r is not None]
    print(f"\nCompleted {len(successful)}/{len(json_files)} examples.")
    return successful


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Security Remediation Pipeline runner"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--repo", help="Repo URL or local path to scan")
    mode.add_argument("--benchmark", help="Path to a single BenchmarkExample JSON file")
    mode.add_argument("--benchmark-dir", help="Directory of BenchmarkExample JSON files")

    parser.add_argument(
        "--language", default="python",
        help="Language for --repo mode (default: python)"
    )
    parser.add_argument(
        "--parallel", type=int, default=4,
        help="Max parallel workers for --benchmark-dir mode (default: 4)"
    )
    parser.add_argument(
        "--clone-timeout", type=int, default=300,
        help="Seconds to wait for git clone before giving up (default: 300)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write EvalResult JSON files (benchmark modes)"
    )

    args = parser.parse_args()

    if args.repo:
        asyncio.run(run_single_repo(args))
    elif args.benchmark:
        asyncio.run(run_single_benchmark(args))
    else:
        asyncio.run(run_benchmark_dir(args))


if __name__ == "__main__":
    main()
