"""
Analyze all failures from benchmark runs and generate a comprehensive failure report.

Usage:
  python scripts/generate_failure_report.py \\
      --results-dir data/results/ \\
      --benchmark-dir data/benchmark/dev/ \\
      --output data/results/failure_analysis/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_agent_security.eval.failure_analysis import FailureClassifier, generate_recommendations
from multi_agent_security.types import (
    BenchmarkExample,
    EvalReport,
    EvalResult,
    FailureAnalysis,
    FailureCategory,
    TaskState,
)

# Re-use the loader from generate_comparison
sys.path.insert(0, str(Path(__file__).parent))
from generate_comparison import load_reports


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_benchmark_examples(benchmark_dir: str) -> dict[str, BenchmarkExample]:
    """Load all BenchmarkExample JSON files, keyed by example ID."""
    examples: dict[str, BenchmarkExample] = {}
    for path in Path(benchmark_dir).glob("*.json"):
        try:
            data = json.loads(path.read_text())
            ex = BenchmarkExample.model_validate(data)
            examples[ex.id] = ex
        except Exception as exc:
            print(f"  WARNING: skipping {path.name}: {exc}", file=sys.stderr)
    return examples


def _minimal_task_state(result: EvalResult) -> TaskState:
    """
    Reconstruct a minimal TaskState from an EvalResult.
    EvalReports do not preserve the full TaskState, so we synthesise what we can.
    revision_count is set to max_revisions when max_retries is likely so that
    the classifier can detect MAX_RETRIES.
    """
    state = TaskState(
        task_id=result.example_id,
        repo_url="",
        language="python",
        status=result.failure_stage or ("complete" if result.end_to_end_success else "failed"),
        revision_count=result.revision_loops,
        max_revisions=3,
    )
    # If revision_loops has reached the default cap and run still failed, mark failed_vulns
    if result.revision_loops >= 3 and not result.end_to_end_success:
        state.failed_vulns = [result.example_id]
    return state


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_all(
    reports_by_config: dict[str, list[EvalReport]],
    examples_by_id: dict[str, BenchmarkExample],
) -> list[FailureAnalysis]:
    """Run FailureClassifier over every EvalResult and collect analyses."""
    classifier = FailureClassifier()
    all_failures: list[FailureAnalysis] = []

    for config_id, reports in sorted(reports_by_config.items()):
        for report in reports:
            for result in report.per_example_results:
                example = examples_by_id.get(result.example_id)
                if example is None:
                    continue
                task_state = _minimal_task_state(result)
                failures = classifier.classify_failure(
                    example=example,
                    eval_result=result,
                    task_state=task_state,
                    config_id=config_id,
                )
                all_failures.extend(failures)

    return all_failures


# ---------------------------------------------------------------------------
# Output 1: Failure distribution table
# ---------------------------------------------------------------------------

def write_failure_distribution(
    all_failures: list[FailureAnalysis],
    config_ids: list[str],
    output_dir: Path,
) -> None:
    # Count by (config_id, category)
    counts: dict[tuple[str, str], int] = Counter(
        (fa.config_id, fa.failure_category.value) for fa in all_failures
    )
    totals_by_config: dict[str, int] = Counter(fa.config_id for fa in all_failures)
    all_categories = sorted({fa.failure_category.value for fa in all_failures})

    # Header
    cols = config_ids + ["Overall"]
    header = "| Failure Category" + "".join(f" | {c}" for c in cols) + " |"
    sep = "|---" + "|---" * len(cols) + "|"

    lines = ["## Failure Distribution Across Configurations", "", header, sep]

    overall_total = len(all_failures)
    for cat in all_categories:
        row = f"| `{cat}`"
        for cid in config_ids:
            n = counts.get((cid, cat), 0)
            total = totals_by_config.get(cid, 0)
            pct = f"{100 * n / total:.0f}%" if total else "N/A"
            row += f" | {pct}"
        overall = sum(counts.get((cid, cat), 0) for cid in config_ids)
        overall_pct = f"{100 * overall / overall_total:.0f}%" if overall_total else "N/A"
        row += f" | {overall_pct} |"
        lines.append(row)

    lines += ["", f"*Total classified failures: {overall_total}*"]
    (output_dir / "failure_distribution.md").write_text("\n".join(lines) + "\n")
    print("  Written: failure_distribution.md")


# ---------------------------------------------------------------------------
# Output 2: Failure case studies
# ---------------------------------------------------------------------------

def write_case_studies(
    all_failures: list[FailureAnalysis],
    examples_by_id: dict[str, BenchmarkExample],
    output_dir: Path,
    top_n: int = 10,
) -> None:
    # Select top-N most frequent category × example pairs
    counted: Counter[str] = Counter(fa.failure_category.value for fa in all_failures)
    top_categories = {cat for cat, _ in counted.most_common(top_n)}

    # Pick one representative failure per top category
    seen_cats: set[str] = set()
    selected: list[FailureAnalysis] = []
    for fa in all_failures:
        cat = fa.failure_category.value
        if cat in top_categories and cat not in seen_cats:
            selected.append(fa)
            seen_cats.add(cat)
        if len(selected) >= top_n:
            break

    lines = ["# Failure Case Studies", ""]
    for i, fa in enumerate(selected, 1):
        ex = examples_by_id.get(fa.example_id)
        vuln_type = ex.vuln_type.value if ex else "unknown"
        complexity = ex.complexity_tag if ex else "unknown"
        lines += [
            f"## Case {i}: `{fa.failure_category.value}`",
            "",
            f"**Example:** `{fa.example_id}`  |  **Config:** `{fa.config_id}`  "
            f"|  **Vuln type:** `{vuln_type}`  |  **Complexity:** `{complexity}`",
            "",
            f"**Stage:** `{fa.failure_stage}`  |  **Severity:** `{fa.severity}`",
            "",
            f"**What happened:** {fa.description}",
            "",
            f"**Root cause:** {fa.root_cause}",
            "",
            _prevention_advice(fa.failure_category),
            "",
        ]

    (output_dir / "failure_case_studies.md").write_text("\n".join(lines) + "\n")
    print("  Written: failure_case_studies.md")


def _prevention_advice(category: FailureCategory) -> str:
    advice = {
        FailureCategory.SCANNER_MISS: (
            "**Prevention:** Augment LLM scanning with Semgrep rules for this CWE category; "
            "include all files in the repository rather than a pre-filtered subset."
        ),
        FailureCategory.SCANNER_FALSE_POSITIVE: (
            "**Prevention:** Add a confidence threshold filter (≥ 0.7) and cross-check "
            "with a lightweight static analyser before forwarding to triage."
        ),
        FailureCategory.SCANNER_WRONG_TYPE: (
            "**Prevention:** Add CWE-specific few-shot examples to the scanner prompt."
        ),
        FailureCategory.TRIAGER_WRONG_SEVERITY: (
            "**Prevention:** Provide the triager with CVSS base score examples for each severity level."
        ),
        FailureCategory.PATCHER_EMPTY_PATCH: (
            "**Prevention:** Validate patch output before submission; retry if diff is empty."
        ),
        FailureCategory.PATCHER_WRONG_FILE: (
            "**Prevention:** Pass the list of vulnerable files from the scanner to the patcher."
        ),
        FailureCategory.PATCHER_BREAKS_CODE: (
            "**Prevention:** Execute the test suite after patching and feed failures back "
            "to the patcher as additional context."
        ),
        FailureCategory.PATCHER_PARTIAL_FIX: (
            "**Prevention:** Provide full file context to the patcher, not just the snippet."
        ),
        FailureCategory.PATCHER_WRONG_APPROACH: (
            "**Prevention:** Include CWE-specific remediation guidelines in the patcher prompt."
        ),
        FailureCategory.REVIEWER_FALSE_REJECT: (
            "**Prevention:** Calibrate reviewer threshold against known-good patches; "
            "add a secondary judge to adjudicate borderline cases."
        ),
        FailureCategory.REVIEWER_UNHELPFUL_FEEDBACK: (
            "**Prevention:** Require structured feedback (file, line, specific change) "
            "rather than free-form rejection reasons."
        ),
        FailureCategory.MAX_RETRIES: (
            "**Prevention:** Improve reviewer feedback quality; increase max_revisions "
            "or add an escape hatch that accepts the best available patch."
        ),
        FailureCategory.CONTEXT_OVERFLOW: (
            "**Prevention:** Switch to sliding-window or retrieval memory strategy; "
            "truncate verbose agent messages."
        ),
        FailureCategory.INFO_LOSS: (
            "**Prevention:** Use full_context memory strategy so the orchestrator "
            "retains all scanner findings when routing to the patcher."
        ),
    }
    return advice.get(category, "**Prevention:** Investigate the specific failure for targeted remediation.")


# ---------------------------------------------------------------------------
# Output 3: Architecture-specific failure patterns
# ---------------------------------------------------------------------------

def write_arch_failure_patterns(
    all_failures: list[FailureAnalysis],
    reports_by_config: dict[str, list[EvalReport]],
    output_dir: Path,
) -> None:
    # Map config_id → architecture
    arch_by_config: dict[str, str] = {}
    for config_id, reports in reports_by_config.items():
        if reports:
            arch_by_config[config_id] = reports[0].config.get("architecture", "unknown")

    # Group failures by architecture
    by_arch: dict[str, list[FailureAnalysis]] = defaultdict(list)
    for fa in all_failures:
        arch = arch_by_config.get(fa.config_id, "unknown")
        by_arch[arch].append(fa)

    lines = ["# Architecture-Specific Failure Patterns", ""]

    arch_notes = {
        "sequential": (
            "In sequential pipelines, information flows strictly from one agent to the next. "
            "Key risks: scanner misses cascade (later agents never see the vulnerability), "
            "and the fixed handoff format can cause INFO_LOSS if the orchestrator truncates context."
        ),
        "hub_spoke": (
            "In hub-spoke pipelines, a central orchestrator routes tasks between agents. "
            "Key risks: the orchestrator may drop critical scanner findings when summarising for "
            "the patcher (INFO_LOSS), and conflicting outputs from parallel agent calls can "
            "create CONFLICTING_AGENTS failures."
        ),
        "blackboard": (
            "In blackboard pipelines, agents share a common state object. "
            "Key risks: the shared context grows with every agent write, making CONTEXT_OVERFLOW "
            "more likely on large repositories. However, INFO_LOSS is less common because agents "
            "read the full blackboard directly."
        ),
        "unknown": "Architecture could not be determined from report config.",
    }

    for arch in sorted(by_arch.keys()):
        failures = by_arch[arch]
        cats = Counter(fa.failure_category.value for fa in failures)
        top3 = cats.most_common(3)

        lines += [
            f"## {arch.replace('_', ' ').title()}",
            "",
            arch_notes.get(arch, ""),
            "",
            f"**Total failures:** {len(failures)}",
            "",
            "**Top failure categories:**",
        ]
        for cat, cnt in top3:
            pct = 100 * cnt / len(failures)
            lines.append(f"- `{cat}`: {cnt} ({pct:.0f}%)")
        lines.append("")

    (output_dir / "arch_failure_patterns.md").write_text("\n".join(lines) + "\n")
    print("  Written: arch_failure_patterns.md")


# ---------------------------------------------------------------------------
# Output 4: Failure heatmap
# ---------------------------------------------------------------------------

def plot_failure_heatmap(
    all_failures: list[FailureAnalysis],
    config_ids: list[str],
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  WARNING: matplotlib/numpy not installed. Skipping failure_heatmap.png")
        print("    Install with: pip install 'multi-agent-security[viz]'")
        return

    all_cats = sorted({fa.failure_category.value for fa in all_failures})
    if not all_cats or not config_ids:
        print("  WARNING: no failure data — skipping failure_heatmap.png")
        return

    counts: dict[tuple[str, str], int] = Counter(
        (fa.config_id, fa.failure_category.value) for fa in all_failures
    )

    matrix = np.zeros((len(config_ids), len(all_cats)), dtype=float)
    for i, cid in enumerate(config_ids):
        for j, cat in enumerate(all_cats):
            matrix[i, j] = counts.get((cid, cat), 0)

    fig, ax = plt.subplots(figsize=(max(10, len(all_cats) * 0.9), max(4, len(config_ids) * 0.7)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Failure count")

    ax.set_xticks(range(len(all_cats)))
    ax.set_xticklabels(all_cats, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(config_ids)))
    ax.set_yticklabels(config_ids)
    ax.set_title("Failure Category Heatmap (count per config)")

    for i in range(len(config_ids)):
        for j in range(len(all_cats)):
            val = int(matrix[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7,
                        color="black" if matrix[i, j] < matrix.max() * 0.6 else "white")

    fig.tight_layout()
    fig.savefig(output_dir / "failure_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Written: failure_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze pipeline failures and generate a comprehensive failure report."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing C*_run*_report.json files",
    )
    parser.add_argument(
        "--benchmark-dir",
        required=True,
        help="Directory containing BenchmarkExample JSON files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <results-dir>/failure_analysis/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output or (Path(args.results_dir) / "failure_analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading reports from: {args.results_dir}")
    reports_by_config = load_reports(args.results_dir)
    if not reports_by_config:
        print("ERROR: No C*_run*_report.json files found.", file=sys.stderr)
        sys.exit(1)
    config_ids = sorted(reports_by_config.keys())
    print(f"Configs found: {config_ids}")

    print(f"Loading benchmark examples from: {args.benchmark_dir}")
    examples_by_id = load_benchmark_examples(args.benchmark_dir)
    print(f"Examples loaded: {len(examples_by_id)}")

    print("Classifying failures…")
    all_failures = classify_all(reports_by_config, examples_by_id)
    print(f"Total classified failures: {len(all_failures)}")

    if not all_failures:
        print("No failures found — all runs succeeded. Report will be minimal.")

    print(f"\nGenerating reports in: {output_dir}")
    write_failure_distribution(all_failures, config_ids, output_dir)
    write_case_studies(all_failures, examples_by_id, output_dir)
    write_arch_failure_patterns(all_failures, reports_by_config, output_dir)
    plot_failure_heatmap(all_failures, config_ids, output_dir)

    rec_text = generate_recommendations(all_failures)
    (output_dir / "recommendations.md").write_text(rec_text)
    print("  Written: recommendations.md")

    print(f"\nDone. All failure analysis artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
