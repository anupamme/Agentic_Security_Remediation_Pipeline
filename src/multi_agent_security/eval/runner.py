"""
EvalRunner: orchestrates the full evaluation pipeline.
"""
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from multi_agent_security.agents.patcher import PatcherAgent
from multi_agent_security.agents.reviewer import ReviewerAgent
from multi_agent_security.agents.scanner import ScannerAgent
from multi_agent_security.agents.triager import TriagerAgent
from multi_agent_security.config import AppConfig
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
from multi_agent_security.orchestration.blackboard import BlackboardOrchestrator
from multi_agent_security.orchestration.hub_spoke import HubSpokeOrchestrator
from multi_agent_security.orchestration.sequential import SequentialOrchestrator
from multi_agent_security.tools.repo_cloner import RepoCloner
from multi_agent_security.types import (
    BenchmarkExample,
    EvalReport,
    EvalResult,
    TaskState,
)
from multi_agent_security.utils.cost_tracker import CostTracker
from multi_agent_security.utils.logging import generate_run_id

_ORCHESTRATORS = {
    "sequential": SequentialOrchestrator,
    "hub_spoke": HubSpokeOrchestrator,
    "blackboard": BlackboardOrchestrator,
}


def _build_orchestrator(config: AppConfig, cost_tracker: CostTracker):
    llm_client = LLMClient(config.llm, cost_tracker=cost_tracker)
    agents = {
        "scanner": ScannerAgent(config, llm_client),
        "triager": TriagerAgent(config, llm_client),
        "patcher": PatcherAgent(config, llm_client),
        "reviewer": ReviewerAgent(config, llm_client),
    }
    memory = create_memory(config, llm_client)
    orch_class = _ORCHESTRATORS[config.architecture]
    return orch_class(config, agents, memory), llm_client


def _render_markdown_report(report: EvalReport) -> str:
    """Render a human-readable Markdown summary of an EvalReport."""
    agg = report.aggregate_metrics
    lines = [
        f"# Eval Report: {report.run_id}",
        f"**Generated:** {report.timestamp.isoformat()}",
        f"**Split:** {report.benchmark_split}  |  **Examples:** {report.num_examples}  |  **Runs:** {report.num_runs}",
        "",
        "## Configuration",
        "```json",
        json.dumps(report.config, indent=2),
        "```",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Mean | Std | Median | Min | Max |",
        "|--------|------|-----|--------|-----|-----|",
    ]

    def _row(name: str, s) -> str:
        return f"| {name} | {s.mean:.3f} | {s.std:.3f} | {s.median:.3f} | {s.min:.3f} | {s.max:.3f} |"

    lines += [
        _row("Detection Recall", agg.detection_recall),
        _row("Detection Precision", agg.detection_precision),
        _row("Triage Accuracy", agg.triage_accuracy),
        _row("Patch Correctness", agg.patch_correctness),
        f"| E2E Success Rate | {agg.e2e_success_rate:.3f} | - | - | - | - |",
        _row("Latency (s)", agg.latency_seconds),
        _row("Total Tokens", agg.total_tokens),
        _row("Total Cost (USD)", agg.total_cost_usd),
        _row("Revision Loops", agg.revision_loops),
        "",
    ]

    if agg.by_vuln_type:
        lines += ["## Per-Vuln-Type Breakdown", "", "| Vuln Type | N | Recall | Precision | Triage Acc | Patch Correct | E2E |", "|-----------|---|--------|-----------|------------|---------------|-----|"]
        for vtype, sub in agg.by_vuln_type.items():
            lines.append(f"| {vtype} | {sub.n_examples} | {sub.detection_recall.mean:.3f} | {sub.detection_precision.mean:.3f} | {sub.triage_accuracy.mean:.3f} | {sub.patch_correctness.mean:.3f} | {sub.e2e_success_rate:.3f} |")
        lines.append("")

    if agg.by_complexity:
        lines += ["## Per-Complexity Breakdown", "", "| Complexity | N | Recall | Precision | Triage Acc | Patch Correct | E2E |", "|------------|---|--------|-----------|------------|---------------|-----|"]
        for ctag, sub in agg.by_complexity.items():
            lines.append(f"| {ctag} | {sub.n_examples} | {sub.detection_recall.mean:.3f} | {sub.detection_precision.mean:.3f} | {sub.triage_accuracy.mean:.3f} | {sub.patch_correctness.mean:.3f} | {sub.e2e_success_rate:.3f} |")
        lines.append("")

    # Top 5 failures
    failures = [r for r in report.per_example_results if not r.end_to_end_success][:5]
    if failures:
        lines += ["## Top Failures (up to 5)", ""]
        for r in failures:
            lines.append(f"- **{r.example_id}**: stage=`{r.failure_stage or 'unknown'}` reason=`{r.failure_reason or 'n/a'}` patch_correctness={r.patch_correctness:.3f}")
        lines.append("")

    lines += [
        "## Cost Summary",
        "```json",
        json.dumps(report.cost_summary, indent=2),
        "```",
    ]
    return "\n".join(lines)


class EvalRunner:
    def __init__(self, config: AppConfig, output_dir: str = "data/results"):
        self._config = config
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def run_eval(
        self,
        benchmark_dir: str,
        architecture: str,
        memory_strategy: str,
        num_runs: int = 1,
        parallel_workers: int = 1,
    ) -> EvalReport:
        """Run evaluation on all examples in benchmark_dir."""
        from multi_agent_security.config import AppConfig

        # Override architecture/memory in config
        config_dict = self._config.model_dump()
        config_dict["architecture"] = architecture
        config_dict["memory"]["strategy"] = memory_strategy
        self._eval_config = AppConfig.model_validate(config_dict)

        benchmark_path = Path(benchmark_dir)
        json_files = sorted(benchmark_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {benchmark_dir}")

        examples = []
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
            try:
                examples.append(BenchmarkExample.model_validate(data))
            except Exception:
                print(f"WARNING: skipping {jf.name} (not a valid BenchmarkExample)", file=sys.stderr)

        if not examples:
            raise ValueError(f"No valid BenchmarkExample files found in {benchmark_dir}")

        # Clone each repo once, before dispatching any runs.
        # Clones run concurrently in a thread pool so git network I/O does not
        # block the event loop. RepoCloner caches by URL so concurrent calls for
        # the same URL are safe (dest.exists() short-circuits the second clone).
        cloner = RepoCloner()

        async def _clone(ex: BenchmarkExample) -> tuple[str, Path | None]:
            path = await asyncio.to_thread(cloner.clone, ex.repo_url)
            return ex.id, path

        clone_results = await asyncio.gather(*(_clone(ex) for ex in examples))

        repo_paths: dict[str, Path] = {}
        cloneable: list[BenchmarkExample] = []
        id_to_example = {ex.id: ex for ex in examples}
        for ex_id, path in clone_results:
            if path is None:
                print(
                    f"WARNING: Could not clone {id_to_example[ex_id].repo_url} "
                    f"for {ex_id} — skipping",
                    file=sys.stderr,
                )
            else:
                repo_paths[ex_id] = path
                cloneable.append(id_to_example[ex_id])

        semaphore = asyncio.Semaphore(parallel_workers)
        all_results: list[EvalResult] = []

        async def _run_with_semaphore(ex: BenchmarkExample, run_num: int) -> EvalResult | None:
            async with semaphore:
                try:
                    return await self._eval_single_example(ex, run_num, repo_paths[ex.id])
                except Exception as exc:
                    print(f"ERROR: {ex.id} run {run_num}: {exc}", file=sys.stderr)
                    return EvalResult(
                        example_id=ex.id,
                        architecture=architecture,
                        memory_strategy=memory_strategy,
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
            _run_with_semaphore(ex, run_num)
            for ex in cloneable
            for run_num in range(1, num_runs + 1)
        ]
        results = await asyncio.gather(*tasks)
        all_results = [r for r in results if r is not None]

        agg = aggregate_metrics(all_results)

        run_id = generate_run_id(architecture, memory_strategy)
        split_name = Path(benchmark_dir).name

        report = EvalReport(
            run_id=run_id,
            config={"architecture": architecture, "memory_strategy": memory_strategy},
            benchmark_split=split_name,
            num_examples=len(examples),
            num_runs=num_runs,
            aggregate_metrics=agg,
            per_example_results=all_results,
            cost_summary={},
            timestamp=datetime.now(timezone.utc),
        )

        # Save JSON report
        json_path = self._output_dir / f"{run_id}_report.json"
        json_path.write_text(report.model_dump_json(indent=2))

        # Save Markdown report
        md_path = self._output_dir / f"{run_id}_report.md"
        md_path.write_text(_render_markdown_report(report))

        print(f"Report saved: {json_path}")
        print(f"Markdown:     {md_path}")
        return report

    async def _eval_single_example(
        self,
        example: BenchmarkExample,
        run_number: int,
        repo_path: Path,
    ) -> EvalResult:
        """Run pipeline on one example and compute all metrics."""
        cost_tracker = CostTracker()
        orchestrator, llm_client = _build_orchestrator(self._eval_config, cost_tracker)

        task_state = TaskState(
            task_id=f"{example.id}-run{run_number}",
            repo_url=example.repo_url,
            language=example.language,
        )

        start = time.monotonic()
        result = await orchestrator.run(str(repo_path), task_state)
        elapsed = time.monotonic() - start

        cost_summary = cost_tracker.get_total_summary()
        total_tokens = cost_summary.total_input_tokens + cost_summary.total_output_tokens

        # Detection metrics
        detection_recall = compute_detection_recall(
            result.vulnerabilities, example.vulnerable_files, example.vuln_type
        )
        detection_precision = compute_detection_precision(
            result.vulnerabilities, example.vulnerable_files, example.negative
        )

        # Triage accuracy
        triage_accuracy = compute_triage_accuracy(
            result.triage_results, example.severity, example.vuln_type,
            predicted_vulns=result.vulnerabilities,
        )

        # Patch correctness
        accepted_patches = [r for r in result.reviews if r.patch_accepted]
        patched = bool(result.patches)
        accepted = bool(accepted_patches)

        judge_score: float | None = None
        patch_correctness = 0.0
        test_result = None

        if result.patches and example.ground_truth_diff:
            best_patch = result.patches[0]
            # Run judge if LLM client available
            try:
                judge = LLMJudge(llm_client)
                vuln_for_judge = result.vulnerabilities[0] if result.vulnerabilities else None
                if vuln_for_judge:
                    js = await judge.judge_patch_correctness(
                        vulnerability=vuln_for_judge,
                        generated_patch=best_patch,
                        ground_truth_diff=example.ground_truth_diff,
                        language=example.language,
                    )
                    judge_score = js.correctness
            except Exception:
                judge_score = None

            patch_correctness = compute_patch_correctness(
                generated_patch=best_patch,
                ground_truth_diff=example.ground_truth_diff,
                test_result=test_result,
                judge_score=judge_score,
            )

        detected = detection_recall > 0.0
        triaged = triage_accuracy > 0.0 or bool(result.triage_results)
        e2e = compute_e2e_success(detected, triaged, patched, accepted, patch_correctness)

        return EvalResult(
            example_id=example.id,
            architecture=self._eval_config.architecture,
            memory_strategy=self._eval_config.memory.strategy,
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
            test_passed=test_result.passed if test_result is not None else None,
            vuln_type=example.vuln_type.value,
            complexity_tag=example.complexity_tag,
        )
