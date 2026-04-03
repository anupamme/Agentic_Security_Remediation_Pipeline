"""Generate publication-quality figures from benchmark results.

Reads result files from data/results/ and writes PNG figures to docs/figures/.

Usage:
    python scripts/generate_figures.py --results-dir data/results/ --output docs/figures/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional matplotlib import — give a clear error if the viz extra is missing
# ---------------------------------------------------------------------------
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.use("Agg")
    plt.style.use("seaborn-v0_8-paper")
except ImportError as exc:
    print(f"ERROR: {exc}")
    print("Install the viz extras:  pip install -e '.[viz]'")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
FIGSIZE = (8, 5)
DPI = 150
FONT_FAMILY = "DejaVu Sans"

# Config IDs → architecture × memory label
CONFIG_LABELS: dict[str, str] = {
    "C1": "Seq / Full",
    "C2": "Seq / Slide",
    "C3": "Seq / RAG",
    "C4": "Hub / Full",
    "C5": "Hub / Slide",
    "C6": "Hub / RAG",
    "C7": "BB / Full",
    "C8": "BB / Slide",
    "C9": "BB / RAG",
}

ARCH_COLORS: dict[str, str] = {
    "sequential": "#2196F3",
    "hub_spoke": "#4CAF50",
    "blackboard": "#FF9800",
}

ABLATION_COLORS: dict[str, str] = {
    "FULL": "#2196F3",
    "A0": "#9C27B0",
    "A1": "#F44336",
    "A2": "#FF9800",
    "A3": "#795548",
    "A4": "#607D8B",
}

matplotlib.rcParams["font.family"] = FONT_FAMILY


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_config_reports(results_dir: Path) -> dict[str, list[dict]]:
    """Load all C1–C9 run reports, grouped by config_id."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for cfg in CONFIG_LABELS:
        for run_file in sorted(results_dir.glob(f"**/{cfg}_run*.json")):
            try:
                data = json.loads(run_file.read_text())
                grouped[cfg].append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    return grouped


def _aggregate(reports: list[dict]) -> dict:
    """Average aggregate_metrics across multiple run reports."""
    if not reports:
        return {}
    keys = ["e2e_success_rate"]
    nested_keys = ["detection_recall", "detection_precision", "triage_accuracy",
                   "patch_correctness", "total_cost_usd", "total_tokens", "latency_seconds"]
    out: dict = {}
    for k in keys:
        vals = [r["aggregate_metrics"][k] for r in reports if k in r.get("aggregate_metrics", {})]
        out[k] = float(np.mean(vals)) if vals else 0.0
    for k in nested_keys:
        vals = [r["aggregate_metrics"][k]["mean"]
                for r in reports
                if k in r.get("aggregate_metrics", {}) and r["aggregate_metrics"][k].get("mean") is not None]
        out[k] = float(np.mean(vals)) if vals else 0.0
    # config metadata from first report
    cfg = reports[0].get("config", {})
    out["architecture"] = cfg.get("architecture", "unknown")
    out["memory_strategy"] = cfg.get("memory_strategy", "unknown")
    out["config_id"] = cfg.get("config_id", "?")
    return out


# ---------------------------------------------------------------------------
# Figure 1: Architecture comparison bar chart
# ---------------------------------------------------------------------------

def fig_architecture_comparison(results_dir: Path, output_dir: Path) -> None:
    """Bar chart of E2E success rate per config (C1–C9)."""
    all_reports = _load_config_reports(results_dir)

    configs = [c for c in CONFIG_LABELS if all_reports.get(c)]
    if not configs:
        print("  [skip] No C1–C9 run reports found — skipping architecture comparison chart")
        return

    agg = {c: _aggregate(all_reports[c]) for c in configs}
    labels = [CONFIG_LABELS[c] for c in configs]
    values = [agg[c]["e2e_success_rate"] * 100 for c in configs]
    colors = [ARCH_COLORS.get(agg[c]["architecture"], "#999") for c in configs]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("E2E Success Rate (%)")
    ax.set_title("Architecture × Memory Strategy: E2E Success Rate")
    ax.set_ylim(0, max(max(values) * 1.25, 10))
    ax.tick_params(axis="x", rotation=30)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    # Legend for architectures
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=a.replace("_", "-").title())
                       for a, c in ARCH_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    out_path = output_dir / "architecture_comparison.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Written: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 2: Pareto frontier — E2E success vs cost
# ---------------------------------------------------------------------------

def fig_pareto_frontier(results_dir: Path, output_dir: Path) -> None:
    """Scatter plot of E2E success vs average cost per config."""
    all_reports = _load_config_reports(results_dir)
    configs = [c for c in CONFIG_LABELS if all_reports.get(c)]
    if not configs:
        print("  [skip] No run reports found — skipping Pareto frontier chart")
        return

    agg = {c: _aggregate(all_reports[c]) for c in configs}
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for c in configs:
        x = agg[c]["total_cost_usd"]
        y = agg[c]["e2e_success_rate"] * 100
        color = ARCH_COLORS.get(agg[c]["architecture"], "#999")
        ax.scatter(x, y, color=color, s=80, zorder=3)
        ax.annotate(CONFIG_LABELS[c], (x, y), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)

    ax.set_xlabel("Avg Cost per Vulnerability (USD)")
    ax.set_ylabel("E2E Success Rate (%)")
    ax.set_title("Pareto Frontier: Success vs. Cost")
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=a.replace("_", "-").title())
                       for a, c in ARCH_COLORS.items()]
    ax.legend(handles=legend_elements, fontsize=9)

    fig.tight_layout()
    out_path = output_dir / "pareto_frontier.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Written: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 3: Token distribution chart
# ---------------------------------------------------------------------------

def fig_token_distribution(results_dir: Path, output_dir: Path) -> None:
    """Grouped bar chart of avg token usage by architecture and memory strategy."""
    all_reports = _load_config_reports(results_dir)
    configs = [c for c in CONFIG_LABELS if all_reports.get(c)]
    if not configs:
        print("  [skip] No run reports found — skipping token distribution chart")
        return

    agg = {c: _aggregate(all_reports[c]) for c in configs}

    archs = ["sequential", "hub_spoke", "blackboard"]
    strategies = ["full_context", "sliding_window", "retrieval"]

    data: dict[str, dict[str, float]] = {a: {} for a in archs}
    for c in configs:
        arch = agg[c]["architecture"]
        mem = agg[c]["memory_strategy"]
        if arch in data and mem in strategies:
            data[arch][mem] = agg[c]["total_tokens"]

    x = np.arange(len(strategies))
    width = 0.25
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, (arch, color) in enumerate(ARCH_COLORS.items()):
        values = [data[arch].get(s, 0) for s in strategies]
        ax.bar(x + i * width, values, width, label=arch.replace("_", "-").title(),
               color=color, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace("_", "\n") for s in strategies])
    ax.set_ylabel("Avg Tokens per Vulnerability")
    ax.set_title("Token Usage by Architecture and Memory Strategy")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "token_distribution.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Written: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 4: Ablation bar chart
# ---------------------------------------------------------------------------

def fig_ablation(results_dir: Path, output_dir: Path) -> None:
    """Bar chart of E2E success for each ablation configuration."""
    ablation_md = None
    for candidate in [
        results_dir / "ablation_full" / "ablation_report.md",
        results_dir / "ablation_smoke" / "ablation_report.md",
    ]:
        if candidate.exists():
            ablation_md = candidate
            break

    if ablation_md is None:
        print("  [skip] No ablation_report.md found — skipping ablation chart")
        return

    rows: list[dict] = []
    for line in ablation_md.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| Config"):
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if len(parts) >= 3:
            cfg = parts[0]
            try:
                e2e = float(parts[2])
                rows.append({"config": cfg, "e2e": e2e})
            except ValueError:
                continue

    if not rows:
        print("  [skip] Could not parse ablation results — skipping ablation chart")
        return

    configs = [r["config"] for r in rows]
    values = [r["e2e"] * 100 for r in rows]
    colors = [ABLATION_COLORS.get(c, "#999") for c in configs]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(configs, values, color=colors, edgecolor="white")
    ax.set_ylabel("E2E Success Rate (%)")
    ax.set_title("Ablation Study: Marginal Value of Each Agent")
    ax.set_ylim(0, max(max(values) * 1.25, 10))

    descriptions = {
        "FULL": "All agents",
        "A0": "Single agent",
        "A1": "No Reviewer",
        "A2": "No Triager",
        "A3": "No Triager\n+ Reviewer",
        "A4": "Scanner\nonly",
    }
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([descriptions.get(c, c) for c in configs], fontsize=9)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = output_dir / "ablation_chart.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Written: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 5: Failure heatmap
# ---------------------------------------------------------------------------

def fig_failure_heatmap(results_dir: Path, output_dir: Path) -> None:
    """Heatmap of failure category × architecture."""
    failure_md = None
    for candidate in [
        results_dir / "benchmark_test_20260402T144259" / "comparison" / "failure_stages.md",
        results_dir / "bedrock_smoke_test" / "comparison" / "failure_stages.md",
    ] + sorted(results_dir.glob("**/failure_stages.md")):
        if candidate.exists():
            failure_md = candidate
            break

    if failure_md is None:
        print("  [skip] No failure_stages.md found — skipping failure heatmap")
        return

    rows: list[dict] = []
    for line in failure_md.read_text().splitlines():
        if not line.startswith("| C") or "Config" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if len(parts) < 7:
            continue
        cfg = parts[0]
        try:
            stage_vals = [float(parts[i + 2].rstrip("%")) for i in range(5)]
            rows.append({"config": cfg, "stages": stage_vals})
        except (ValueError, IndexError):
            continue

    if not rows:
        print("  [skip] Could not parse failure_stages.md — skipping failure heatmap")
        return

    matrix = np.array([r["stages"] for r in rows])
    configs = [r["config"] for r in rows]
    stages = ["Scanner", "Triager", "Patcher", "Reviewer", "Runner"]

    fig, ax = plt.subplots(figsize=(8, max(3, len(configs) * 0.6 + 1.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    fig.colorbar(im, ax=ax, label="Failure %")
    ax.set_xticks(range(len(stages)))
    ax.set_yticks(range(len(configs)))
    ax.set_xticklabels(stages)
    ax.set_yticklabels(configs)
    ax.set_title("Failure Distribution by Stage and Configuration")

    for i in range(len(configs)):
        for j in range(len(stages)):
            ax.text(j, i, f"{matrix[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                    color="black" if matrix[i, j] < 50 else "white")

    fig.tight_layout()
    out_path = output_dir / "failure_heatmap.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Written: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 6: Dataset statistics
# ---------------------------------------------------------------------------

def fig_dataset_statistics(results_dir: Path, output_dir: Path) -> None:
    """Pie/bar charts showing benchmark dataset distribution."""
    benchmark_dir = results_dir.parent / "benchmark" / "test"
    if not benchmark_dir.exists():
        print("  [skip] data/benchmark/test/ not found — skipping dataset statistics")
        return

    examples = []
    for f in sorted(benchmark_dir.glob("*.json")):
        try:
            examples.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue

    if not examples:
        print("  [skip] No benchmark examples loaded — skipping dataset statistics")
        return

    vuln_counts: dict[str, int] = defaultdict(int)
    complexity_counts: dict[str, int] = defaultdict(int)
    lang_counts: dict[str, int] = defaultdict(int)

    for ex in examples:
        vuln_counts[ex.get("vuln_type", "UNKNOWN")] += 1
        complexity_counts[ex.get("complexity_tag", "unknown")] += 1
        lang_counts[ex.get("language", "unknown")] += 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Vuln type distribution
    ax = axes[0]
    sorted_vulns = sorted(vuln_counts.items(), key=lambda x: -x[1])
    labels, vals = zip(*sorted_vulns)
    ax.barh(labels, vals, color="#2196F3")
    ax.set_xlabel("Count")
    ax.set_title("Vulnerability Types")
    ax.invert_yaxis()

    # Complexity distribution
    ax = axes[1]
    sorted_comp = sorted(complexity_counts.items(), key=lambda x: -x[1])
    labels2, vals2 = zip(*sorted_comp)
    ax.bar(labels2, vals2, color="#4CAF50")
    ax.set_ylabel("Count")
    ax.set_title("Complexity Tags")
    ax.tick_params(axis="x", rotation=30)

    # Language distribution
    ax = axes[2]
    sorted_lang = sorted(lang_counts.items(), key=lambda x: -x[1])[:8]
    labels3, vals3 = zip(*sorted_lang)
    ax.bar(labels3, vals3, color="#FF9800")
    ax.set_ylabel("Count")
    ax.set_title("Programming Languages")
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle(f"Benchmark Dataset Statistics (n={len(examples)})", fontsize=12)
    fig.tight_layout()
    out_path = output_dir / "dataset_statistics.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Written: {out_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures from benchmark results")
    parser.add_argument("--results-dir", default="data/results/", help="Path to results directory")
    parser.add_argument("--output", default="docs/figures/", help="Output directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output).resolve()

    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures from: {results_dir}")
    print(f"Output directory:        {output_dir}\n")

    fig_architecture_comparison(results_dir, output_dir)
    fig_pareto_frontier(results_dir, output_dir)
    fig_token_distribution(results_dir, output_dir)
    fig_ablation(results_dir, output_dir)
    fig_failure_heatmap(results_dir, output_dir)
    fig_dataset_statistics(results_dir, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
