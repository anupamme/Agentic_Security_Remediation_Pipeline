"""
Metric computation functions for the evaluation framework.

All functions take structured pipeline output and ground-truth data,
returning a float score (or bool for e2e_success).
"""
import re
import statistics
from typing import Optional

from multi_agent_security.types import (
    AggregateMetrics,
    EvalResult,
    MetricStats,
    TriageResult,
    Vulnerability,
    VulnSeverity,
    VulnType,
)
from multi_agent_security.tools.test_runner import TestResult
from multi_agent_security.types import Patch

_SEVERITY_ORDER = [
    VulnSeverity.INFO,
    VulnSeverity.LOW,
    VulnSeverity.MEDIUM,
    VulnSeverity.HIGH,
    VulnSeverity.CRITICAL,
]


def _normalize_path(path: str) -> str:
    """Strip leading ./ from a file path for comparison."""
    return path.lstrip("./")


# === Detection Metrics ===

def compute_detection_recall(
    predicted_vulns: list[Vulnerability],
    ground_truth_files: list[str],
    ground_truth_vuln_type: VulnType,
) -> float:
    """What fraction of ground-truth vulnerable files were flagged?"""
    if not ground_truth_files:
        return 1.0
    normalized_gt = {_normalize_path(f) for f in ground_truth_files}
    normalized_pred = {_normalize_path(v.file_path) for v in predicted_vulns}
    matched = len(normalized_gt & normalized_pred)
    return matched / len(normalized_gt)


def compute_detection_precision(
    predicted_vulns: list[Vulnerability],
    ground_truth_files: list[str],
    is_negative_example: bool,
) -> float:
    """What fraction of scanner findings are real vulnerabilities?"""
    if is_negative_example:
        return 0.0 if predicted_vulns else 1.0
    if not predicted_vulns:
        return 1.0
    normalized_gt = {_normalize_path(f) for f in ground_truth_files}
    tp = sum(1 for v in predicted_vulns if _normalize_path(v.file_path) in normalized_gt)
    return tp / len(predicted_vulns)


# === Triage Metrics ===

def compute_triage_accuracy(
    predicted_triage: list[TriageResult],
    ground_truth_severity: VulnSeverity,
    ground_truth_vuln_type: VulnType,
) -> float:
    """Does the triager's severity match ground truth?"""
    # Find triage result matching the ground truth vuln type
    matching = None
    for t in predicted_triage:
        # Match by checking if the vuln_id references the right vuln type
        # Since TriageResult has vuln_id not vuln_type, we pick the best severity match
        matching = t
        break  # Use first result if no type info available

    if matching is None:
        return 0.0

    gt_idx = _SEVERITY_ORDER.index(ground_truth_severity)
    pred_idx = _SEVERITY_ORDER.index(matching.severity)
    diff = abs(gt_idx - pred_idx)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.5
    else:
        return 0.0


# === Patch Metrics ===

def _extract_changed_lines(diff: str) -> list[str]:
    """Extract added/removed lines from a unified diff (ignore context/header lines)."""
    lines = []
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            lines.append(line[1:].strip())
    return lines


def _extract_function_names(diff: str) -> set[str]:
    """Extract function/method names modified in a diff."""
    pattern = re.compile(r"(?:def |function )\s*(\w+)|(\w+)\s*\(")
    names = set()
    for line in diff.splitlines():
        if line.startswith("+") or line.startswith("-"):
            for m in pattern.finditer(line):
                name = m.group(1) or m.group(2)
                if name:
                    names.add(name)
    return names


def compute_diff_similarity(generated_diff: str, ground_truth_diff: str) -> float:
    """
    Semantic similarity between two diffs using token-level Jaccard similarity.
    """
    if not generated_diff and not ground_truth_diff:
        return 1.0
    if not generated_diff or not ground_truth_diff:
        return 0.0

    gen_lines = _extract_changed_lines(generated_diff)
    gt_lines = _extract_changed_lines(ground_truth_diff)

    gen_tokens = set(" ".join(gen_lines).split())
    gt_tokens = set(" ".join(gt_lines).split())

    if not gen_tokens and not gt_tokens:
        return 1.0
    if not gen_tokens or not gt_tokens:
        return 0.0

    intersection = len(gen_tokens & gt_tokens)
    union = len(gen_tokens | gt_tokens)
    jaccard = intersection / union if union > 0 else 0.0

    # Bonus: +0.2 if both modify the same functions/methods
    gen_funcs = _extract_function_names(generated_diff)
    gt_funcs = _extract_function_names(ground_truth_diff)
    bonus = 0.2 if gen_funcs and gt_funcs and (gen_funcs & gt_funcs) else 0.0

    return min(1.0, jaccard + bonus)


def compute_patch_correctness(
    generated_patch: Patch,
    ground_truth_diff: str,
    test_result: Optional[TestResult],
    judge_score: Optional[float],
) -> float:
    """
    Composite score for patch quality.

    Components (weighted average):
    - Test pass (weight 0.4): 1.0 if tests pass, 0.0 if fail
    - Diff similarity (weight 0.3): semantic similarity to ground truth diff
    - LLM judge score (weight 0.3): from judge.py
    """
    components: list[tuple[float, float]] = []  # (score, weight)

    if test_result is not None:
        test_score = 1.0 if test_result.passed else 0.0
        components.append((test_score, 0.4))

    diff_sim = compute_diff_similarity(generated_patch.unified_diff, ground_truth_diff)
    components.append((diff_sim, 0.3))

    if judge_score is not None:
        components.append((judge_score, 0.3))

    if not components:
        return 0.0

    total_weight = sum(w for _, w in components)
    weighted_sum = sum(s * w for s, w in components)
    return weighted_sum / total_weight


# === End-to-End Metrics ===

def compute_e2e_success(
    detected: bool,
    triaged: bool,
    patched: bool,
    accepted: bool,
    patch_correct: float,
) -> bool:
    """End-to-end success requires ALL stages to pass."""
    return detected and triaged and patched and accepted and patch_correct >= 0.5


# === System Metrics ===

def compute_context_utilization(
    total_tokens_used: int,
    max_context_window: int = 200_000,
) -> float:
    """Peak context utilization as a fraction of max window."""
    return total_tokens_used / max_context_window


# === Aggregation ===

def _metric_stats(values: list[float]) -> MetricStats:
    """Compute descriptive statistics for a list of float values."""
    if not values:
        return MetricStats(mean=0.0, std=0.0, median=0.0, min=0.0, max=0.0)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    return MetricStats(mean=mean, std=std, median=median, min=min(values), max=max(values))


def _aggregate_subset(results: list[EvalResult]) -> AggregateMetrics:
    """Aggregate metrics for a subset of results (no breakdown sub-keys)."""
    n = len(results)
    if n == 0:
        zero = MetricStats(mean=0.0, std=0.0, median=0.0, min=0.0, max=0.0)
        return AggregateMetrics(
            n_examples=0,
            detection_recall=zero,
            detection_precision=zero,
            triage_accuracy=zero,
            patch_correctness=zero,
            e2e_success_rate=0.0,
            total_cost_usd=zero,
            total_tokens=zero,
            latency_seconds=zero,
            revision_loops=zero,
        )
    return AggregateMetrics(
        n_examples=n,
        detection_recall=_metric_stats([r.detection_recall for r in results]),
        detection_precision=_metric_stats([r.detection_precision for r in results]),
        triage_accuracy=_metric_stats([r.triage_accuracy for r in results]),
        patch_correctness=_metric_stats([r.patch_correctness for r in results]),
        e2e_success_rate=sum(1 for r in results if r.end_to_end_success) / n,
        total_cost_usd=_metric_stats([r.total_cost_usd for r in results]),
        total_tokens=_metric_stats([float(r.total_tokens) for r in results]),
        latency_seconds=_metric_stats([r.latency_seconds for r in results]),
        revision_loops=_metric_stats([float(r.revision_loops) for r in results]),
    )


def aggregate_metrics(results: list[EvalResult]) -> AggregateMetrics:
    """
    Compute mean, std, median, min, max for each metric across all examples.
    Also compute per-vuln-type and per-complexity breakdowns.
    """
    base = _aggregate_subset(results)

    # Per-vuln-type breakdown
    by_type: dict[str, list[EvalResult]] = {}
    for r in results:
        if r.vuln_type:
            by_type.setdefault(r.vuln_type, []).append(r)
    base.by_vuln_type = {k: _aggregate_subset(v) for k, v in by_type.items()}

    # Per-complexity breakdown
    by_complexity: dict[str, list[EvalResult]] = {}
    for r in results:
        if r.complexity_tag:
            by_complexity.setdefault(r.complexity_tag, []).append(r)
    base.by_complexity = {k: _aggregate_subset(v) for k, v in by_complexity.items()}

    return base
