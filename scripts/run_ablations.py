"""
Run ablation configurations and compare against the full multi-agent pipeline.

Answers: "Does multi-agent actually help, and which agents contribute the most?"

Usage:
  # Run all ablations on dev split, 1 run each
  python scripts/run_ablations.py --split dev --runs 1 \\
      --config config/arch_sequential.yaml

  # Run specific ablations, compare against existing C1 results
  python scripts/run_ablations.py --split test --runs 3 \\
      --ablations A0,A1 --baseline-config C1 \\
      --config config/arch_sequential.yaml

  # Custom output directory
  python scripts/run_ablations.py --split dev --runs 1 \\
      --config config/arch_sequential.yaml \\
      --output-dir data/results/ablation_smoke
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_agent_security.agents.patcher import PatcherAgent
from multi_agent_security.agents.reviewer import ReviewerAgent
from multi_agent_security.agents.scanner import ScannerAgent
from multi_agent_security.agents.single_agent import SingleAgent
from multi_agent_security.agents.triager import TriagerAgent
from multi_agent_security.config import AppConfig, load_config
from multi_agent_security.eval.judge import LLMJudge
from multi_agent_security.eval.metrics import (
    aggregate_metrics,
    compute_detection_precision,
    compute_detection_recall,
    compute_e2e_success,
    compute_patch_correctness,
    compute_triage_accuracy,
)
from multi_agent_security.llm_client import LLMClient
from multi_agent_security.memory import create_memory
from multi_agent_security.orchestration.ablation import AblationOrchestrator, SingleAgentOrchestrator
from multi_agent_security.orchestration.sequential import SequentialOrchestrator
from multi_agent_security.tools.repo_cloner import RepoCloner
from multi_agent_security.types import (
    AggregateMetrics,
    BenchmarkExample,
    EvalReport,
    EvalResult,
    TaskState,
)
from multi_agent_security.utils.cost_tracker import CostTracker
from multi_agent_security.utils.logging import setup_logging

# ---------------------------------------------------------------------------
# Ablation configuration table
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    {"id": "A0", "description": "Single agent",              "type": "single_agent",  "skip_agents": []},
    {"id": "A1", "description": "No Reviewer",               "type": "ablation",      "skip_agents": ["reviewer"]},
    {"id": "A2", "description": "No Triager",                "type": "ablation",      "skip_agents": ["triager"]},
    {"id": "A3", "description": "No Triager + No Reviewer",  "type": "ablation",      "skip_agents": ["triager", "reviewer"]},
    {"id": "A4", "description": "Scanner only",              "type": "ablation",      "skip_agents": ["triager", "patcher", "reviewer"]},
]

_ABLATION_BY_ID = {cfg["id"]: cfg for cfg in ABLATION_CONFIGS}


# ---------------------------------------------------------------------------
# Orchestrator factory
# ---------------------------------------------------------------------------

def _build_ablation_orchestrator(
    config: AppConfig,
    cost_tracker: CostTracker,
    ablation_cfg: dict,
):
    """Build the appropriate orchestrator for a given ablation config."""
    llm_client = LLMClient(config.llm, cost_tracker=cost_tracker)
    memory = create_memory(config, llm_client)

    if ablation_cfg["type"] == "single_agent":
        agents = {"single_agent": SingleAgent(config, llm_client)}
        return SingleAgentOrchestrator(config, agents, memory), llm_client

    # Standard agents for ablation orchestrator
    agents = {
        "scanner": ScannerAgent(config, llm_client),
        "triager": TriagerAgent(config, llm_client),
        "patcher": PatcherAgent(config, llm_client),
        "reviewer": ReviewerAgent(config, llm_client),
    }
    return AblationOrchestrator(
        config, agents, memory, skip_agents=ablation_cfg["skip_agents"]
    ), llm_client


def _build_baseline_orchestrator(config: AppConfig, cost_tracker: CostTracker):
    """Build the full sequential pipeline (baseline)."""
    llm_client = LLMClient(config.llm, cost_tracker=cost_tracker)
    agents = {
        "scanner": ScannerAgent(config, llm_client),
        "triager": TriagerAgent(config, llm_client),
        "patcher": PatcherAgent(config, llm_client),
        "reviewer": ReviewerAgent(config, llm_client),
    }
    memory = create_memory(config, llm_client)
    return SequentialOrchestrator(config, agents, memory), llm_client


# ---------------------------------------------------------------------------
# Single-example evaluation
# ---------------------------------------------------------------------------

async def _eval_single_example(
    config: AppConfig,
    example: BenchmarkExample,
    repo_path: Path,
    run_number: int,
    ablation_cfg: dict,
) -> EvalResult:
    """Run one ablation config on one example and compute metrics."""
    cost_tracker = CostTracker()
    orchestrator, llm_client = _build_ablation_orchestrator(config, cost_tracker, ablation_cfg)

    task_state = TaskState(
        task_id=f"{example.id}-{ablation_cfg['id']}-run{run_number}",
        repo_url=example.repo_url,
        language=example.language,
    )

    start = time.monotonic()
    result = await orchestrator.run(str(repo_path), task_state)
    elapsed = time.monotonic() - start

    cost_summary = cost_tracker.get_total_summary()
    total_tokens = cost_summary.total_input_tokens + cost_summary.total_output_tokens

    detection_recall = compute_detection_recall(
        result.vulnerabilities, example.vulnerable_files, example.vuln_type
    )
    detection_precision = compute_detection_precision(
        result.vulnerabilities, example.vulnerable_files, example.negative
    )
    triage_accuracy = compute_triage_accuracy(
        result.triage_results, example.severity, example.vuln_type,
        predicted_vulns=result.vulnerabilities,
    )

    judge_score: Optional[float] = None
    patch_correctness = 0.0
    if result.patches and example.ground_truth_diff:
        # Select the *last* patch for a ground-truth vulnerable file so that
        # revision-loop refinements are preferred over the initial attempt.
        # Fall back to the last patch overall if no file matches.
        gt_files = {f.lstrip("./") for f in example.vulnerable_files}
        best_patch = next(
            (p for p in reversed(result.patches) if p.file_path.lstrip("./") in gt_files),
            result.patches[-1],
        )
        # Also prefer the vuln for the same file when calling the judge.
        vuln_for_judge = next(
            (v for v in result.vulnerabilities if v.file_path.lstrip("./") in gt_files),
            result.vulnerabilities[0] if result.vulnerabilities else None,
        )
        try:
            judge = LLMJudge(llm_client)
            if vuln_for_judge:
                js = await judge.judge_patch_correctness(
                    vulnerability=vuln_for_judge,
                    generated_patch=best_patch,
                    ground_truth_diff=example.ground_truth_diff,
                    language=example.language,
                )
                judge_score = js.correctness
        except Exception:
            pass

        patch_correctness = compute_patch_correctness(
            generated_patch=best_patch,
            ground_truth_diff=example.ground_truth_diff,
            test_result=None,
            judge_score=judge_score,
        )

    detected = detection_recall > 0.0
    triaged = triage_accuracy > 0.0 or bool(result.triage_results)
    patched = bool(result.patches)
    accepted = bool([r for r in result.reviews if r.patch_accepted])
    e2e = compute_e2e_success(detected, triaged, patched, accepted, patch_correctness)

    return EvalResult(
        example_id=example.id,
        architecture=ablation_cfg["id"],
        memory_strategy=config.memory.strategy,
        detection_recall=detection_recall,
        detection_precision=detection_precision,
        triage_accuracy=triage_accuracy,
        patch_correctness=patch_correctness,
        end_to_end_success=e2e,
        total_tokens=total_tokens,
        total_cost_usd=cost_summary.total_cost_usd,
        latency_seconds=elapsed,
        revision_loops=result.revision_count,
        failure_stage=result.status if result.status == "failed" else None,
        vuln_type=example.vuln_type.value,
        complexity_tag=example.complexity_tag,
    )


async def _eval_baseline_example(
    config: AppConfig,
    example: BenchmarkExample,
    repo_path: Path,
    run_number: int,
) -> EvalResult:
    """Run the full sequential pipeline (baseline) on one example."""
    cost_tracker = CostTracker()
    orchestrator, llm_client = _build_baseline_orchestrator(config, cost_tracker)

    task_state = TaskState(
        task_id=f"{example.id}-FULL-run{run_number}",
        repo_url=example.repo_url,
        language=example.language,
    )

    start = time.monotonic()
    result = await orchestrator.run(str(repo_path), task_state)
    elapsed = time.monotonic() - start

    cost_summary = cost_tracker.get_total_summary()
    total_tokens = cost_summary.total_input_tokens + cost_summary.total_output_tokens

    detection_recall = compute_detection_recall(
        result.vulnerabilities, example.vulnerable_files, example.vuln_type
    )
    detection_precision = compute_detection_precision(
        result.vulnerabilities, example.vulnerable_files, example.negative
    )
    triage_accuracy = compute_triage_accuracy(
        result.triage_results, example.severity, example.vuln_type,
        predicted_vulns=result.vulnerabilities,
    )

    judge_score = None
    patch_correctness = 0.0
    if result.patches and example.ground_truth_diff:
        gt_files = {f.lstrip("./") for f in example.vulnerable_files}
        best_patch = next(
            (p for p in reversed(result.patches) if p.file_path.lstrip("./") in gt_files),
            result.patches[-1],
        )
        vuln_for_judge = next(
            (v for v in result.vulnerabilities if v.file_path.lstrip("./") in gt_files),
            result.vulnerabilities[0] if result.vulnerabilities else None,
        )
        try:
            judge = LLMJudge(llm_client)
            if vuln_for_judge:
                js = await judge.judge_patch_correctness(
                    vulnerability=vuln_for_judge,
                    generated_patch=best_patch,
                    ground_truth_diff=example.ground_truth_diff,
                    language=example.language,
                )
                judge_score = js.correctness
        except Exception:
            pass
        patch_correctness = compute_patch_correctness(
            generated_patch=best_patch,
            ground_truth_diff=example.ground_truth_diff,
            test_result=None,
            judge_score=judge_score,
        )

    detected = detection_recall > 0.0
    triaged = triage_accuracy > 0.0 or bool(result.triage_results)
    patched = bool(result.patches)
    accepted = bool([r for r in result.reviews if r.patch_accepted])
    e2e = compute_e2e_success(detected, triaged, patched, accepted, patch_correctness)

    return EvalResult(
        example_id=example.id,
        architecture="FULL",
        memory_strategy=config.memory.strategy,
        detection_recall=detection_recall,
        detection_precision=detection_precision,
        triage_accuracy=triage_accuracy,
        patch_correctness=patch_correctness,
        end_to_end_success=e2e,
        total_tokens=total_tokens,
        total_cost_usd=cost_summary.total_cost_usd,
        latency_seconds=elapsed,
        revision_loops=result.revision_count,
        failure_stage=result.status if result.status == "failed" else None,
        vuln_type=example.vuln_type.value,
        complexity_tag=example.complexity_tag,
    )


# ---------------------------------------------------------------------------
# Baseline loading
# ---------------------------------------------------------------------------

def _load_baseline_from_reports(
    results_dir: Path,
    baseline_config_id: str,
) -> Optional[list[EvalResult]]:
    """Load existing EvalResults for a config ID from saved report JSON files."""
    pattern = f"{baseline_config_id}_*_report.json"
    matches = list(results_dir.glob(pattern))
    if not matches:
        return None
    all_results: list[EvalResult] = []
    for path in sorted(matches):
        try:
            raw = json.loads(path.read_text())
            report = EvalReport.model_validate(raw)
            all_results.extend(report.per_example_results)
        except Exception as exc:
            print(f"WARNING: could not load {path.name}: {exc}", file=sys.stderr)
    return all_results if all_results else None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_ablation_report(
    results_by_config: dict[str, list[EvalResult]],
    baseline_config_id: str,
    output_dir: Path,
) -> None:
    """Write ablation_report.md and ablation_chart.png."""
    output_dir.mkdir(parents=True, exist_ok=True)

    agg_by_config: dict[str, AggregateMetrics] = {
        cfg_id: aggregate_metrics(results)
        for cfg_id, results in results_by_config.items()
    }

    baseline_e2e = (
        agg_by_config[baseline_config_id].e2e_success_rate
        if baseline_config_id in agg_by_config
        else None
    )

    # --- Markdown report ---
    lines = [
        "## Ablation Study Results",
        "",
        "### Impact of Each Agent",
        "",
        "| Config | Description | E2E Success | Patch Correct | Cost ($) | Delta vs Full Pipeline |",
        "|--------|-------------|-------------|---------------|----------|------------------------|",
    ]

    # Print baseline first (if present), then ablations in order
    ordered_ids = (
        [baseline_config_id]
        + [cfg["id"] for cfg in ABLATION_CONFIGS if cfg["id"] in agg_by_config and cfg["id"] != baseline_config_id]
    )
    # Include any extra configs not in the above lists
    for cfg_id in agg_by_config:
        if cfg_id not in ordered_ids:
            ordered_ids.append(cfg_id)

    for cfg_id in ordered_ids:
        if cfg_id not in agg_by_config:
            continue
        agg = agg_by_config[cfg_id]
        e2e = agg.e2e_success_rate
        patch = agg.patch_correctness.mean
        cost = agg.total_cost_usd.mean

        if cfg_id == baseline_config_id:
            delta_str = "—"
            desc = "Full pipeline (baseline)"
        else:
            cfg_meta = _ABLATION_BY_ID.get(cfg_id, {})
            desc = cfg_meta.get("description", cfg_id)
            if baseline_e2e is not None:
                delta = e2e - baseline_e2e
                sign = "+" if delta >= 0 else ""
                delta_str = f"{sign}{delta:.2f}"
            else:
                delta_str = "N/A"

        # A4 has no patches/reviews — show N/A
        if cfg_id == "A4":
            lines.append(
                f"| {cfg_id} | {desc} | {e2e:.2f} | N/A | ${cost:.3f} | {delta_str} |"
            )
        else:
            lines.append(
                f"| {cfg_id} | {desc} | {e2e:.2f} | {patch:.2f} | ${cost:.3f} | {delta_str} |"
            )

    lines += [
        "",
        "### Key Findings",
    ]

    # Auto-generate key findings if baseline is present
    for cfg_id in ["A1", "A2", "A0"]:
        if cfg_id in agg_by_config and baseline_e2e is not None:
            delta = agg_by_config[cfg_id].e2e_success_rate - baseline_e2e
            sign = "+" if delta >= 0 else ""
            label = _ABLATION_BY_ID.get(cfg_id, {}).get("description", cfg_id)
            lines.append(f"- {label}: {sign}{delta:.2f} E2E success vs full pipeline")

    lines += [""]

    report_path = output_dir / "ablation_report.md"
    report_path.write_text("\n".join(lines))
    print(f"  Written: {report_path}")

    # --- Bar chart ---
    _plot_ablation_chart(agg_by_config, baseline_config_id, output_dir)


def _plot_ablation_chart(
    agg_by_config: dict[str, AggregateMetrics],
    baseline_config_id: str,
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  WARNING: matplotlib/numpy not installed. Skipping ablation_chart.png")
        print("    Install with: pip install 'multi-agent-security[viz]'")
        return

    # Order: baseline first, then A0–A4
    ordered = (
        ([baseline_config_id] if baseline_config_id in agg_by_config else [])
        + [cfg["id"] for cfg in ABLATION_CONFIGS if cfg["id"] in agg_by_config]
    )
    # Deduplicate while preserving order
    seen: set[str] = set()
    config_ids = []
    for c in ordered:
        if c not in seen:
            config_ids.append(c)
            seen.add(c)

    e2e_values = [agg_by_config[c].e2e_success_rate for c in config_ids]
    bar_colors = [
        "steelblue" if c == baseline_config_id else "darkorange"
        for c in config_ids
    ]

    x = np.arange(len(config_ids))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, e2e_values, color=bar_colors, edgecolor="black", linewidth=0.5)

    # Annotate bars with values
    for bar, val in zip(bars, e2e_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(config_ids)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("E2E Success Rate")
    ax.set_title("Ablation Study: E2E Success Rate by Configuration")
    ax.set_ylim(0, min(1.1, max(e2e_values) + 0.15) if e2e_values else 1.0)
    ax.grid(axis="y", alpha=0.3)

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color="steelblue", label="Full pipeline (baseline)"),
        mpatches.Patch(color="darkorange", label="Ablation config"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    chart_path = output_dir / "ablation_chart.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  Written: {chart_path}")


# ---------------------------------------------------------------------------
# Main async driver
# ---------------------------------------------------------------------------

async def run_ablations(args) -> None:
    config = load_config(args.config)
    setup_logging(config.logging.level)

    output_dir = Path(args.output_dir or "data/results")
    benchmark_dir = Path(f"data/benchmark/{args.split}")

    # Load benchmark examples
    json_files = sorted(benchmark_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: No JSON files found in {benchmark_dir}", file=sys.stderr)
        sys.exit(1)

    examples: list[BenchmarkExample] = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        try:
            examples.append(BenchmarkExample.model_validate(data))
        except Exception:
            print(f"WARNING: skipping {jf.name} (not a valid BenchmarkExample)", file=sys.stderr)

    if not examples:
        print(f"ERROR: No valid BenchmarkExample files in {benchmark_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse ablation filter
    requested_ids: Optional[set[str]] = None
    if args.ablations:
        requested_ids = set(args.ablations.split(","))
        unknown = requested_ids - {cfg["id"] for cfg in ABLATION_CONFIGS}
        if unknown:
            print(f"ERROR: Unknown ablation IDs: {unknown}", file=sys.stderr)
            sys.exit(1)

    ablations_to_run = [
        cfg for cfg in ABLATION_CONFIGS
        if requested_ids is None or cfg["id"] in requested_ids
    ]

    # Clone repos
    cloner = RepoCloner()

    async def _clone(ex: BenchmarkExample) -> tuple[str, Path | None]:
        # If repo_url is an absolute path that already exists on disk, use it directly.
        local = Path(ex.repo_url)
        if local.is_absolute() and local.is_dir():
            return ex.id, local
        path = await asyncio.to_thread(cloner.clone, ex.repo_url)
        # Guard against partial clone dirs left by concurrent clone failures.
        # A valid shallow clone has tracked files; failed clones have 0.
        if path is not None and cloner.count_repo_files(path) == 0:
            return ex.id, None
        return ex.id, path

    print(f"Cloning {len(examples)} repo(s)...")
    clone_results = await asyncio.gather(*(_clone(ex) for ex in examples))

    repo_paths: dict[str, Path] = {}
    cloneable: list[BenchmarkExample] = []
    id_to_example = {ex.id: ex for ex in examples}
    for ex_id, path in clone_results:
        if path is None:
            print(f"WARNING: Could not clone {id_to_example[ex_id].repo_url} — skipping", file=sys.stderr)
        else:
            repo_paths[ex_id] = path
            cloneable.append(id_to_example[ex_id])

    semaphore = asyncio.Semaphore(args.parallel)
    results_by_config: dict[str, list[EvalResult]] = {}

    # --- Run ablation configs ---
    for ablation_cfg in ablations_to_run:
        print(f"\n=== Running ablation {ablation_cfg['id']}: {ablation_cfg['description']} ===")
        config_results: list[EvalResult] = []

        async def _run_one(ex: BenchmarkExample, run_num: int, cfg=ablation_cfg) -> EvalResult:
            async with semaphore:
                try:
                    return await _eval_single_example(
                        config, ex, repo_paths[ex.id], run_num, cfg
                    )
                except Exception as exc:
                    print(f"ERROR: {ex.id} {cfg['id']} run {run_num}: {exc}", file=sys.stderr)
                    return EvalResult(
                        example_id=ex.id,
                        architecture=cfg["id"],
                        memory_strategy=config.memory.strategy,
                        detection_recall=0.0,
                        detection_precision=0.0,
                        triage_accuracy=0.0,
                        patch_correctness=0.0,
                        end_to_end_success=False,
                        total_tokens=0,
                        total_cost_usd=0.0,
                        latency_seconds=0.0,
                        revision_loops=0,
                        failure_stage="runner",
                        failure_reason=str(exc),
                        vuln_type=ex.vuln_type.value,
                        complexity_tag=ex.complexity_tag,
                    )

        tasks = [
            _run_one(ex, run_num)
            for ex in cloneable
            for run_num in range(1, args.runs + 1)
        ]
        ablation_results = await asyncio.gather(*tasks)
        config_results = [r for r in ablation_results if r is not None]
        results_by_config[ablation_cfg["id"]] = config_results

        agg = aggregate_metrics(config_results)
        print(f"  E2E: {agg.e2e_success_rate:.3f}  "
              f"Recall: {agg.detection_recall.mean:.3f}  "
              f"Cost: ${agg.total_cost_usd.mean:.4f}/ex")

    # --- Baseline results ---
    # Determine the config ID under which baseline results will be stored.
    # If a named config is requested and its reports are found on disk, use that
    # name as the key.  If the load fails (or no --baseline-config was given),
    # run the full sequential pipeline inline and store it under "FULL".
    # baseline_config_id is set *after* we know which path was taken so that
    # generate_ablation_report always receives a key that exists in results_by_config.
    run_inline_baseline = True

    if args.baseline_config:
        loaded = _load_baseline_from_reports(output_dir, args.baseline_config)
        if loaded:
            print(f"\nLoaded existing baseline results for {args.baseline_config} "
                  f"({len(loaded)} records)")
            results_by_config[args.baseline_config] = loaded
            baseline_config_id = args.baseline_config
            run_inline_baseline = False
        else:
            print(
                f"\nWARNING: No saved results found for --baseline-config {args.baseline_config}. "
                f"Running full sequential pipeline as baseline instead.",
                file=sys.stderr,
            )
            baseline_config_id = "FULL"
    else:
        baseline_config_id = "FULL"

    if run_inline_baseline:
        print("\n=== Running full sequential pipeline (baseline) ===")
        baseline_results: list[EvalResult] = []

        async def _run_baseline(ex: BenchmarkExample, run_num: int) -> EvalResult:
            async with semaphore:
                try:
                    return await _eval_baseline_example(config, ex, repo_paths[ex.id], run_num)
                except Exception as exc:
                    print(f"ERROR: {ex.id} FULL run {run_num}: {exc}", file=sys.stderr)
                    return EvalResult(
                        example_id=ex.id,
                        architecture="FULL",
                        memory_strategy=config.memory.strategy,
                        detection_recall=0.0,
                        detection_precision=0.0,
                        triage_accuracy=0.0,
                        patch_correctness=0.0,
                        end_to_end_success=False,
                        total_tokens=0,
                        total_cost_usd=0.0,
                        latency_seconds=0.0,
                        revision_loops=0,
                        failure_stage="runner",
                        failure_reason=str(exc),
                        vuln_type=ex.vuln_type.value,
                        complexity_tag=ex.complexity_tag,
                    )

        baseline_tasks = [
            _run_baseline(ex, run_num)
            for ex in cloneable
            for run_num in range(1, args.runs + 1)
        ]
        b_results = await asyncio.gather(*baseline_tasks)
        baseline_results = [r for r in b_results if r is not None]
        results_by_config["FULL"] = baseline_results

        if baseline_results:
            agg = aggregate_metrics(baseline_results)
            print(f"  E2E: {agg.e2e_success_rate:.3f}  "
                  f"Recall: {agg.detection_recall.mean:.3f}  "
                  f"Cost: ${agg.total_cost_usd.mean:.4f}/ex")

    # --- Generate report ---
    print(f"\nGenerating ablation report in: {output_dir}")
    generate_ablation_report(results_by_config, baseline_config_id, output_dir)
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation studies for the Multi-Agent Security Pipeline"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--split", required=True, choices=["dev", "test", "hard"],
        help="Benchmark split to evaluate",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per example")
    parser.add_argument(
        "--ablations", default=None,
        help="Comma-separated ablation IDs to run (default: all). E.g. A0,A1",
    )
    parser.add_argument(
        "--baseline-config", default=None,
        help="Load existing baseline results by config ID (e.g. C1). "
             "Falls back to running the full pipeline inline if not found.",
    )
    parser.add_argument("--parallel", type=int, default=1, help="Parallel workers")
    parser.add_argument("--output-dir", default=None, help="Directory for output files")

    args = parser.parse_args()
    asyncio.run(run_ablations(args))


if __name__ == "__main__":
    main()
