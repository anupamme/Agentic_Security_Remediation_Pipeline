"""
Evaluation runner for the Multi-Agent Security Remediation Pipeline.

Usage:
  # Run eval on dev set with sequential architecture
  python scripts/run_eval.py --split dev --arch sequential --memory full_context \\
      --config config/arch_sequential.yaml

  # Run eval on test set with all architectures (generates comparison)
  python scripts/run_eval.py --split test --arch all --memory all --runs 3 \\
      --config config/arch_sequential.yaml

  # Run judge calibration
  python scripts/run_eval.py --calibrate --calibration-data data/calibration/ \\
      --config config/arch_sequential.yaml
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_agent_security.config import load_config
from multi_agent_security.eval.runner import EvalRunner
from multi_agent_security.utils.logging import setup_logging

_ALL_ARCHITECTURES = ["sequential", "hub_spoke", "blackboard"]
_ALL_MEMORY = ["full_context", "sliding_window", "retrieval"]


async def run_eval(args) -> None:
    config = load_config(args.config)
    setup_logging(config.logging.level)

    benchmark_dir = f"data/benchmark/{args.split}"
    runner = EvalRunner(config, output_dir=args.output_dir or "data/results")

    architectures = _ALL_ARCHITECTURES if args.arch == "all" else [args.arch]
    memories = _ALL_MEMORY if args.memory == "all" else [args.memory]

    reports = []
    for arch in architectures:
        for mem in memories:
            print(f"\n=== Running eval: arch={arch} memory={mem} ===")
            report = await runner.run_eval(
                benchmark_dir=benchmark_dir,
                architecture=arch,
                memory_strategy=mem,
                num_runs=args.runs,
                parallel_workers=args.parallel,
            )
            reports.append((arch, mem, report))

    if len(reports) > 1:
        _print_comparison_table(reports)


def _print_comparison_table(reports) -> None:
    print("\n## Architecture Comparison")
    print(f"{'Arch':<15} {'Memory':<20} {'Recall':>8} {'Precision':>10} {'E2E':>6} {'Cost':>10}")
    print("-" * 75)
    for arch, mem, r in reports:
        agg = r.aggregate_metrics
        print(
            f"{arch:<15} {mem:<20} "
            f"{agg.detection_recall.mean:>8.3f} "
            f"{agg.detection_precision.mean:>10.3f} "
            f"{agg.e2e_success_rate:>6.3f} "
            f"${agg.total_cost_usd.mean:>9.4f}"
        )


async def run_calibration(args) -> None:
    """Run judge calibration against human-labeled data."""
    import json as _json
    from multi_agent_security.config import load_config
    from multi_agent_security.eval.judge import LLMJudge, calibrate_judge
    from multi_agent_security.llm_client import LLMClient
    from multi_agent_security.types import BenchmarkExample, Patch

    config = load_config(args.config)
    setup_logging(config.logging.level)

    llm_client = LLMClient(config.llm)
    judge = LLMJudge(llm_client)

    cal_dir = Path(args.calibration_data)
    examples = []
    for jf in sorted(cal_dir.glob("*.json")):
        with open(jf) as f:
            data = _json.load(f)
        example = BenchmarkExample.model_validate(data["example"])
        patch = Patch.model_validate(data["patch"])
        human_score = float(data["human_score"])
        examples.append((example, patch, human_score))

    if not examples:
        print(f"No calibration examples found in {cal_dir}", file=sys.stderr)
        return

    result = await calibrate_judge(judge, examples)
    print(f"\nCalibration Results ({result.n_examples} examples):")
    print(f"  Pearson r:  {result.pearson_r:.4f}")
    print(f"  Spearman r: {result.spearman_r:.4f}")
    print(f"  MAE:        {result.mean_absolute_error:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluation runner for Multi-Agent Security Pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output-dir", default=None, help="Directory for reports")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--split", choices=["dev", "test", "hard"], help="Benchmark split to evaluate")
    mode.add_argument("--calibrate", action="store_true", help="Run judge calibration")

    parser.add_argument("--arch", default="sequential", help="Architecture (or 'all')")
    parser.add_argument("--memory", default="full_context", help="Memory strategy (or 'all')")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per example")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel workers")
    parser.add_argument("--calibration-data", default="data/calibration/", help="Path to calibration JSON files")

    args = parser.parse_args()

    if args.calibrate:
        asyncio.run(run_calibration(args))
    else:
        asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
