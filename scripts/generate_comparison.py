"""
Load all EvalReports from a benchmark run and generate comparison tables + plots.

Usage:
  python scripts/generate_comparison.py data/results/benchmark_test_20260402T000000/
  python scripts/generate_comparison.py data/results/benchmark_test_20260402T000000/ \\
      --output-dir data/results/comparison/
"""

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_agent_security.eval.metrics import aggregate_metrics
from multi_agent_security.types import AggregateMetrics, EvalReport, EvalResult

# Known config IDs for completeness checks
_KNOWN_CONFIGS = {"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"}


@dataclass
class ConfigSummary:
    config_id: str
    architecture: str
    memory_strategy: str
    orchestrator_type: str
    num_runs: int
    all_results: list[EvalResult]
    agg: AggregateMetrics
    per_run_e2e: list[float]
    failure_stage_counts: dict[str, int]
    failure_stage_total: int


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_reports(results_dir: str) -> dict[str, list[EvalReport]]:
    """
    Load all C*_run*_report.json files and group by config ID.
    Returns {"C1": [report1, report2, ...], ...}
    """
    reports_by_config: dict[str, list[EvalReport]] = {}
    for path in sorted(Path(results_dir).glob("C*_run*_report.json")):
        # Filename: C1_run1_report.json  →  config_id = "C1"
        config_id = path.name.split("_")[0]
        raw = json.loads(path.read_text())
        reports_by_config.setdefault(config_id, []).append(
            EvalReport.model_validate(raw)
        )
    return reports_by_config


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_cross_run_aggregates(
    reports_by_config: dict[str, list[EvalReport]],
) -> dict[str, ConfigSummary]:
    """Pool per-example results across all runs for each config."""
    summaries: dict[str, ConfigSummary] = {}
    for config_id, reports in reports_by_config.items():
        all_results = [r for report in reports for r in report.per_example_results]
        agg = aggregate_metrics(all_results)
        per_run_e2e = [r.aggregate_metrics.e2e_success_rate for r in reports]

        stage_counts: dict[str, int] = {}
        for result in all_results:
            if not result.end_to_end_success and result.failure_stage:
                stage_counts[result.failure_stage] = (
                    stage_counts.get(result.failure_stage, 0) + 1
                )

        # Recover orchestrator_type and config_id from augmented config dict
        cfg = reports[0].config
        summaries[config_id] = ConfigSummary(
            config_id=config_id,
            architecture=cfg.get("architecture", "unknown"),
            memory_strategy=cfg.get("memory_strategy", "unknown"),
            orchestrator_type=cfg.get("orchestrator_type", "rule_based"),
            num_runs=len(reports),
            all_results=all_results,
            agg=agg,
            per_run_e2e=per_run_e2e,
            failure_stage_counts=stage_counts,
            failure_stage_total=sum(stage_counts.values()),
        )
    return summaries


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def compute_pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """
    Given points as (cost_usd, e2e_success_rate), return indices of
    Pareto-optimal configs (minimize cost, maximize success rate).

    A point i is dominated if there exists j such that:
      cost[j] <= cost[i] AND success[j] >= success[i]
      with at least one strict inequality.
    """
    n = len(points)
    dominated = [False] * n
    for i in range(n):
        ci, si = points[i]
        for j in range(n):
            if i == j:
                continue
            cj, sj = points[j]
            if cj <= ci and sj >= si and (cj < ci or sj > si):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def compute_significance_matrix(
    summaries: dict[str, ConfigSummary],
    alpha: float = 0.05,
) -> dict[tuple[str, str], tuple[bool, float]]:
    """
    Wilcoxon signed-rank test on paired per-example E2E outcomes.
    Returns {(id_a, id_b): (is_significant, p_value)} for all pairs i < j.
    """
    from scipy.stats import wilcoxon  # already in pyproject.toml

    results: dict[tuple[str, str], tuple[bool, float]] = {}
    config_ids = sorted(summaries.keys())

    for i, id_a in enumerate(config_ids):
        for id_b in config_ids[i + 1:]:
            a_by_id = {
                r.example_id: float(r.end_to_end_success)
                for r in summaries[id_a].all_results
            }
            b_by_id = {
                r.example_id: float(r.end_to_end_success)
                for r in summaries[id_b].all_results
            }
            shared = sorted(set(a_by_id) & set(b_by_id))

            if len(shared) < 10:
                results[(id_a, id_b)] = (False, 1.0)
                continue

            a_vals = [a_by_id[k] for k in shared]
            b_vals = [b_by_id[k] for k in shared]
            diffs = [a - b for a, b in zip(a_vals, b_vals)]

            if all(d == 0.0 for d in diffs):
                results[(id_a, id_b)] = (False, 1.0)
                continue

            try:
                _, p = wilcoxon(a_vals, b_vals, zero_method="wilcox")
            except ValueError:
                results[(id_a, id_b)] = (False, 1.0)
                continue

            results[(id_a, id_b)] = (p < alpha, p)

    return results


# ---------------------------------------------------------------------------
# Output 1: Main comparison table
# ---------------------------------------------------------------------------

def write_comparison_table(
    summaries: dict[str, ConfigSummary],
    output_dir: Path,
    sig_matrix: dict[tuple[str, str], tuple[bool, float]],
) -> None:
    lines = [
        "# Architecture × Memory Comparison",
        "",
        "| Config | Arch | Memory | Orch | E2E ± std | Recall | Precision | Patch | Cost/ex | Tokens/ex | Latency(s) | Rev Loops |",
        "|--------|------|--------|------|-----------|--------|-----------|-------|---------|-----------|------------|-----------|",
    ]
    for cid in sorted(summaries.keys()):
        s = summaries[cid]
        agg = s.agg
        e2e_std = statistics.stdev(s.per_run_e2e) if len(s.per_run_e2e) > 1 else 0.0
        lines.append(
            f"| {cid} | {s.architecture} | {s.memory_strategy} | {s.orchestrator_type} "
            f"| {agg.e2e_success_rate:.3f}±{e2e_std:.3f} "
            f"| {agg.detection_recall.mean:.3f}±{agg.detection_recall.std:.3f} "
            f"| {agg.detection_precision.mean:.3f}±{agg.detection_precision.std:.3f} "
            f"| {agg.patch_correctness.mean:.3f}±{agg.patch_correctness.std:.3f} "
            f"| ${agg.total_cost_usd.mean:.4f} "
            f"| {agg.total_tokens.mean:.0f} "
            f"| {agg.latency_seconds.mean:.1f} "
            f"| {agg.revision_loops.mean:.2f} |"
        )

    if sig_matrix:
        lines += [
            "",
            "## Statistical Significance (Wilcoxon, p < 0.05)",
            "",
            "Pairs with significant E2E differences:",
        ]
        found_any = False
        for (id_a, id_b), (is_sig, p) in sorted(sig_matrix.items()):
            if is_sig:
                lines.append(f"- **{id_a}** vs **{id_b}**: p = {p:.4f}")
                found_any = True
        if not found_any:
            lines.append("- No pairs differ significantly.")

    (output_dir / "comparison_table.md").write_text("\n".join(lines) + "\n")
    print(f"  Written: comparison_table.md")


# ---------------------------------------------------------------------------
# Output 2: Per-complexity breakdown
# ---------------------------------------------------------------------------

def write_complexity_breakdown(
    summaries: dict[str, ConfigSummary],
    output_dir: Path,
) -> None:
    # Collect all complexity tags across configs
    all_tags: set[str] = set()
    for s in summaries.values():
        all_tags.update(s.agg.by_complexity.keys())

    lines = ["# Per-Complexity Breakdown", ""]
    for tag in sorted(all_tags):
        lines += [
            f"## {tag}",
            "",
            "| Config | Arch | Memory | N | Recall | Precision | Patch | E2E |",
            "|--------|------|--------|---|--------|-----------|-------|-----|",
        ]
        for cid in sorted(summaries.keys()):
            s = summaries[cid]
            sub = s.agg.by_complexity.get(tag)
            if sub is None or sub.n_examples == 0:
                continue
            lines.append(
                f"| {cid} | {s.architecture} | {s.memory_strategy} "
                f"| {sub.n_examples} "
                f"| {sub.detection_recall.mean:.3f} "
                f"| {sub.detection_precision.mean:.3f} "
                f"| {sub.patch_correctness.mean:.3f} "
                f"| {sub.e2e_success_rate:.3f} |"
            )
        lines.append("")

    (output_dir / "complexity_breakdown.md").write_text("\n".join(lines))
    print(f"  Written: complexity_breakdown.md")


# ---------------------------------------------------------------------------
# Output 3: Per-vuln-type breakdown
# ---------------------------------------------------------------------------

def write_vuln_type_breakdown(
    summaries: dict[str, ConfigSummary],
    output_dir: Path,
) -> None:
    all_types: set[str] = set()
    for s in summaries.values():
        all_types.update(s.agg.by_vuln_type.keys())

    lines = ["# Per-Vulnerability-Type Breakdown", ""]
    for vtype in sorted(all_types):
        lines += [
            f"## {vtype}",
            "",
            "| Config | Arch | Memory | N | Recall | Precision | Patch | E2E |",
            "|--------|------|--------|---|--------|-----------|-------|-----|",
        ]
        for cid in sorted(summaries.keys()):
            s = summaries[cid]
            sub = s.agg.by_vuln_type.get(vtype)
            if sub is None or sub.n_examples == 0:
                continue
            lines.append(
                f"| {cid} | {s.architecture} | {s.memory_strategy} "
                f"| {sub.n_examples} "
                f"| {sub.detection_recall.mean:.3f} "
                f"| {sub.detection_precision.mean:.3f} "
                f"| {sub.patch_correctness.mean:.3f} "
                f"| {sub.e2e_success_rate:.3f} |"
            )
        lines.append("")

    (output_dir / "vuln_type_breakdown.md").write_text("\n".join(lines))
    print(f"  Written: vuln_type_breakdown.md")


# ---------------------------------------------------------------------------
# Output 4: Failure stage distribution
# ---------------------------------------------------------------------------

def write_failure_stages(
    summaries: dict[str, ConfigSummary],
    output_dir: Path,
) -> None:
    known_stages = ["scanner", "triager", "patcher", "reviewer", "runner"]

    lines = [
        "# Failure Stage Distribution",
        "",
        "| Config | Total Failures | scanner% | triager% | patcher% | reviewer% | runner% | Success% |",
        "|--------|---------------|----------|----------|----------|-----------|---------|---------|",
    ]
    for cid in sorted(summaries.keys()):
        s = summaries[cid]
        n_total = len(s.all_results)
        n_fail = s.failure_stage_total
        n_success = sum(1 for r in s.all_results if r.end_to_end_success)

        def _pct(stage: str) -> str:
            count = s.failure_stage_counts.get(stage, 0)
            return f"{100 * count / n_total:.1f}%" if n_total else "0%"

        success_pct = f"{100 * n_success / n_total:.1f}%" if n_total else "0%"
        lines.append(
            f"| {cid} | {n_fail} "
            + " | ".join(_pct(stage) for stage in known_stages)
            + f" | {success_pct} |"
        )

    (output_dir / "failure_stages.md").write_text("\n".join(lines) + "\n")
    print(f"  Written: failure_stages.md")


# ---------------------------------------------------------------------------
# Output 5: Pareto frontier plot
# ---------------------------------------------------------------------------

def plot_pareto_frontier(
    summaries: dict[str, ConfigSummary],
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not installed. Skipping pareto_frontier.png")
        print("    Install with: pip install 'multi-agent-security[viz]'")
        return

    config_ids = sorted(summaries.keys())
    costs = [summaries[cid].agg.total_cost_usd.mean for cid in config_ids]
    successes = [summaries[cid].agg.e2e_success_rate for cid in config_ids]

    frontier_indices = compute_pareto_frontier(list(zip(costs, successes)))
    frontier_sorted = sorted(frontier_indices, key=lambda i: costs[i])

    arch_colors = {
        "sequential": "steelblue",
        "hub_spoke": "darkorange",
        "blackboard": "forestgreen",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, cid in enumerate(config_ids):
        arch = summaries[cid].architecture
        color = arch_colors.get(arch, "gray")
        marker = "D" if i in frontier_indices else "o"
        ax.scatter(costs[i], successes[i], c=color, marker=marker, s=130, zorder=5)
        ax.annotate(
            cid, (costs[i], successes[i]),
            textcoords="offset points", xytext=(7, 4), fontsize=9,
        )

    if len(frontier_sorted) > 1:
        fx = [costs[i] for i in frontier_sorted]
        fy = [successes[i] for i in frontier_sorted]
        ax.plot(fx, fy, "k--", linewidth=1.2, label="Pareto frontier", zorder=4)

    ax.set_xlabel("Mean Cost per Example (USD)")
    ax.set_ylabel("E2E Success Rate")
    ax.set_title("Success Rate vs. Cost: Architecture × Memory Comparison")

    legend_handles = [
        mpatches.Patch(color=c, label=a) for a, c in arch_colors.items()
    ]
    if len(frontier_sorted) > 1:
        import matplotlib.lines as mlines
        legend_handles.append(
            mlines.Line2D([], [], color="black", linestyle="--", label="Pareto frontier")
        )
    ax.legend(handles=legend_handles, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_frontier.png", dpi=150)
    plt.close(fig)
    print(f"  Written: pareto_frontier.png")


# ---------------------------------------------------------------------------
# Output 6: Token distribution plot
# ---------------------------------------------------------------------------

def plot_token_distribution(
    summaries: dict[str, ConfigSummary],
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  WARNING: matplotlib/numpy not installed. Skipping token_distribution.png")
        print("    Install with: pip install 'multi-agent-security[viz]'")
        return

    config_ids = sorted(summaries.keys())
    means = [summaries[cid].agg.total_tokens.mean for cid in config_ids]
    stds = [summaries[cid].agg.total_tokens.std for cid in config_ids]

    arch_colors = {
        "sequential": "steelblue",
        "hub_spoke": "darkorange",
        "blackboard": "forestgreen",
    }
    bar_colors = [
        arch_colors.get(summaries[cid].architecture, "gray") for cid in config_ids
    ]

    x = np.arange(len(config_ids))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, means, yerr=stds, capsize=4, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids)
    ax.set_xlabel("Config")
    ax.set_ylabel("Mean Total Tokens per Example")
    ax.set_title("Token Usage Distribution by Configuration")
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=c, label=a) for a, c in arch_colors.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "token_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Written: token_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison artifacts from benchmark results"
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing C*_run*_report.json files",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for artifacts (default: <results_dir>/comparison/)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold for Wilcoxon tests (default: 0.05)",
    )
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir or (Path(args.results_dir) / "comparison")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading reports from: {args.results_dir}")
    reports_by_config = load_reports(args.results_dir)

    if not reports_by_config:
        print(
            f"ERROR: No C*_run*_report.json files found in {args.results_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = _KNOWN_CONFIGS - set(reports_by_config.keys())
    if missing:
        print(f"WARNING: missing results for config(s): {sorted(missing)}")

    print(f"Found results for: {sorted(reports_by_config.keys())}")
    summaries = compute_cross_run_aggregates(reports_by_config)
    sig_matrix = compute_significance_matrix(summaries, alpha=args.alpha)

    print(f"\nGenerating artifacts in: {output_dir}")
    write_comparison_table(summaries, output_dir, sig_matrix)
    write_complexity_breakdown(summaries, output_dir)
    write_vuln_type_breakdown(summaries, output_dir)
    write_failure_stages(summaries, output_dir)
    plot_pareto_frontier(summaries, output_dir)
    plot_token_distribution(summaries, output_dir)

    print(f"\nDone. All artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
