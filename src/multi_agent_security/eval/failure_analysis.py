"""
Automated failure classification for the evaluation framework.

Classifies pipeline failures into a structured taxonomy so that
weaknesses can be reported honestly and acted upon.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from multi_agent_security.eval.metrics import compute_diff_similarity
from multi_agent_security.types import (
    BenchmarkExample,
    EvalResult,
    FailureAnalysis,
    FailureCategory,
    Patch,
    ReviewResult,
    TaskState,
)


class FailureClassifier:
    """Classify pipeline failures into structured FailureAnalysis records."""

    def __init__(self, llm_judge=None):
        self.judge = llm_judge

    def classify_failure(
        self,
        example: BenchmarkExample,
        eval_result: EvalResult,
        task_state: TaskState,
        config_id: str = "unknown",
    ) -> list[FailureAnalysis]:
        """
        Analyze a pipeline run and classify its failure(s).
        Returns an empty list if no failures are detected.
        A single run can have multiple failure categories.
        """
        failures: list[FailureAnalysis] = []
        reason = (eval_result.failure_reason or "").lower()

        # --- System-level errors first (from failure_reason string) ---
        if re.search(r"context|token.?limit|context.?window", reason):
            failures.append(FailureAnalysis(
                example_id=example.id,
                config_id=config_id,
                failure_category=FailureCategory.CONTEXT_OVERFLOW,
                failure_stage="system",
                severity="critical",
                description="Pipeline ran out of context window tokens.",
                root_cause="Input to one or more agents exceeded the model context limit.",
            ))
        if re.search(r"timeout|timed.?out", reason):
            failures.append(FailureAnalysis(
                example_id=example.id,
                config_id=config_id,
                failure_category=FailureCategory.TIMEOUT,
                failure_stage="system",
                severity="critical",
                description="Pipeline exceeded the time limit.",
                root_cause="A stage took longer than the configured timeout.",
            ))
        if re.search(r"api.?error|rate.?limit|status.?5\d\d|500|503", reason):
            failures.append(FailureAnalysis(
                example_id=example.id,
                config_id=config_id,
                failure_category=FailureCategory.API_ERROR,
                failure_stage="system",
                severity="critical",
                description="LLM API returned an error.",
                root_cause="Rate limit, server error, or network issue caused an API failure.",
            ))

        # --- Max retries ---
        if (
            task_state.revision_count >= task_state.max_revisions
            and task_state.failed_vulns
        ):
            failures.append(FailureAnalysis(
                example_id=example.id,
                config_id=config_id,
                failure_category=FailureCategory.MAX_RETRIES,
                failure_stage="system",
                severity="critical",
                description=f"Exhausted {task_state.max_revisions} revision loops without acceptance.",
                root_cause="Reviewer kept rejecting patches or patcher could not produce acceptable fixes.",
            ))

        # --- Scanner failures ---
        if not example.negative and eval_result.detection_recall < 1.0:
            missed = round((1.0 - eval_result.detection_recall) * 100)
            failures.append(FailureAnalysis(
                example_id=example.id,
                config_id=config_id,
                failure_category=FailureCategory.SCANNER_MISS,
                failure_stage="scanner",
                severity="critical",
                description=f"Scanner missed {missed}% of ground-truth vulnerable files.",
                root_cause="Scanner did not flag one or more files listed in vulnerable_files.",
            ))

        if eval_result.detection_precision < 1.0:
            fp_severity = "critical" if example.negative else "minor"
            failures.append(FailureAnalysis(
                example_id=example.id,
                config_id=config_id,
                failure_category=FailureCategory.SCANNER_FALSE_POSITIVE,
                failure_stage="scanner",
                severity=fp_severity,
                description="Scanner flagged files that are not in the ground-truth vulnerable set.",
                root_cause="Over-sensitive scanner produced spurious findings.",
            ))

        if task_state.vulnerabilities and not example.negative:
            detected_types = {v.vuln_type for v in task_state.vulnerabilities}
            if example.vuln_type not in detected_types:
                failures.append(FailureAnalysis(
                    example_id=example.id,
                    config_id=config_id,
                    failure_category=FailureCategory.SCANNER_WRONG_TYPE,
                    failure_stage="scanner",
                    severity="minor",
                    description=(
                        f"Scanner detected {[t.value for t in detected_types]} "
                        f"but ground truth is {example.vuln_type.value}."
                    ),
                    root_cause="Scanner misclassified the CWE category of the vulnerability.",
                ))

        # --- Triager failures ---
        if task_state.triage_results and eval_result.triage_accuracy < 0.5:
            failures.append(FailureAnalysis(
                example_id=example.id,
                config_id=config_id,
                failure_category=FailureCategory.TRIAGER_WRONG_SEVERITY,
                failure_stage="triager",
                severity="minor",
                description=f"Triage accuracy {eval_result.triage_accuracy:.2f} < 0.5.",
                root_cause="Triager assigned a severity that differs by 2+ levels from ground truth.",
            ))

        # --- Patch failures ---
        if task_state.patches and eval_result.patch_correctness < 0.5:
            best_patch = task_state.patches[-1]  # last (most recent) attempt
            # Find the review for this patch if available
            review = next(
                (r for r in reversed(task_state.reviews) if r.vuln_id == best_patch.vuln_id),
                None,
            )
            category = self._classify_patch_failure(
                patch=best_patch,
                ground_truth_diff=example.ground_truth_diff,
                review=review,
                tests_passed=eval_result.test_passed,
            )
            failures.append(FailureAnalysis(
                example_id=example.id,
                config_id=config_id,
                failure_category=category,
                failure_stage="patcher",
                severity="critical",
                description=f"Patch correctness {eval_result.patch_correctness:.2f} < 0.5 ({category.value}).",
                root_cause=_patch_failure_root_cause(category),
            ))

        # --- Reviewer false reject ---
        accepted_count = sum(1 for r in task_state.reviews if r.patch_accepted)
        if (
            eval_result.patch_correctness >= 0.5
            and not eval_result.end_to_end_success
            and accepted_count == 0
            and task_state.patches
        ):
            feedback = ""
            last_review = task_state.reviews[-1] if task_state.reviews else None
            if last_review and last_review.revision_request:
                words = len(last_review.revision_request.split())
                if words < 10:
                    feedback = " Reviewer feedback was vague."
            failures.append(FailureAnalysis(
                example_id=example.id,
                config_id=config_id,
                failure_category=(
                    FailureCategory.REVIEWER_UNHELPFUL_FEEDBACK
                    if feedback
                    else FailureCategory.REVIEWER_FALSE_REJECT
                ),
                failure_stage="reviewer",
                severity="critical",
                description=f"Reviewer rejected a patch with correctness {eval_result.patch_correctness:.2f}.{feedback}",
                root_cause="Reviewer threshold too strict or reviewer reasoning was incorrect.",
            ))

        return failures

    def _classify_patch_failure(
        self,
        patch: Patch,
        ground_truth_diff: str,
        review: Optional[ReviewResult],
        tests_passed: Optional[bool],
    ) -> FailureCategory:
        """Determine specific patch failure category."""
        if not patch.unified_diff or not patch.unified_diff.strip():
            return FailureCategory.PATCHER_EMPTY_PATCH

        # Extract files modified in ground truth diff
        gt_files = _extract_diff_files(ground_truth_diff)
        if gt_files and _normalize_path(patch.file_path) not in gt_files:
            return FailureCategory.PATCHER_WRONG_FILE

        if tests_passed is False:
            return FailureCategory.PATCHER_BREAKS_CODE

        if review is not None:
            # High security but low correctness → wrong approach
            if review.security_score >= 0.6 and review.correctness_score < 0.4:
                return FailureCategory.PATCHER_WRONG_APPROACH

        # Low diff similarity → partial fix
        similarity = compute_diff_similarity(patch.unified_diff, ground_truth_diff)
        if similarity < 0.2:
            return FailureCategory.PATCHER_PARTIAL_FIX

        return FailureCategory.PATCHER_WRONG_APPROACH

    async def analyze_with_llm(
        self,
        example: BenchmarkExample,
        task_state: TaskState,
        failures: list[FailureAnalysis],
    ) -> list[FailureAnalysis]:
        """
        Optionally enhance root_cause fields with LLM-generated analysis.
        Returns failures unchanged if no judge is configured.
        """
        if not self.judge or not failures:
            return failures

        enriched: list[FailureAnalysis] = []
        for fa in failures:
            try:
                prompt = (
                    f"A security remediation pipeline failed on example '{example.id}' "
                    f"with category '{fa.failure_category.value}'.\n"
                    f"Current root cause: {fa.root_cause}\n"
                    f"In 1-2 sentences, give a more specific reason why this failure likely occurred "
                    f"given the pipeline architecture and the vulnerability type {example.vuln_type.value}."
                )
                response = await self.judge.llm_client.complete(prompt)
                if response:
                    updated = fa.model_copy(update={"root_cause": response.strip()})
                    enriched.append(updated)
                    continue
            except Exception:
                pass
            enriched.append(fa)
        return enriched


def generate_recommendations(failure_analyses: list[FailureAnalysis]) -> str:
    """
    Generate actionable markdown recommendations from observed failure patterns.
    """
    if not failure_analyses:
        return "No failures to analyze — all runs succeeded.\n"

    counts = Counter(fa.failure_category for fa in failure_analyses)
    total = len(failure_analyses)
    dominant = counts.most_common()

    lines = ["## Failure-Based Recommendations", ""]

    recommendations: list[tuple[int, str]] = []
    for category, count in dominant:
        pct = 100 * count / total
        if pct < 2:
            continue
        label = f"`{category.value}` ({count}/{total}, {pct:.0f}%)"
        if category == FailureCategory.SCANNER_MISS:
            recommendations.append((count, (
                f"- **{label}**: Scanner misses are the dominant failure. "
                "Recommend integrating Semgrep or Bandit static analysis to supplement "
                "LLM-based scanning, and expanding the file set provided per scan."
            )))
        elif category == FailureCategory.SCANNER_FALSE_POSITIVE:
            recommendations.append((count, (
                f"- **{label}**: High false positive rate. "
                "Recommend raising the confidence threshold in the scanner prompt or "
                "adding a post-filter step that cross-checks findings against static tools."
            )))
        elif category == FailureCategory.PATCHER_BREAKS_CODE:
            recommendations.append((count, (
                f"- **{label}**: Patches frequently break existing tests. "
                "Recommend mandatory test execution after patching and feeding test "
                "failure output back to the patcher as context."
            )))
        elif category == FailureCategory.PATCHER_PARTIAL_FIX:
            recommendations.append((count, (
                f"- **{label}**: Patches are incomplete. "
                "Recommend providing the patcher with the full file context rather than "
                "only the vulnerable snippet."
            )))
        elif category == FailureCategory.INFO_LOSS:
            recommendations.append((count, (
                f"- **{label}**: Critical information is lost in routing. "
                "Recommend switching to `full_context` memory strategy instead of "
                "compressed or sliding-window strategies."
            )))
        elif category == FailureCategory.MAX_RETRIES:
            recommendations.append((count, (
                f"- **{label}**: Revision loops exhaust retries. "
                "Recommend improving reviewer feedback quality — require structured "
                "feedback with specific line-level change requests."
            )))
        elif category == FailureCategory.REVIEWER_FALSE_REJECT:
            recommendations.append((count, (
                f"- **{label}**: Reviewer rejects good patches. "
                "Recommend calibrating reviewer acceptance threshold or adding a "
                "secondary judge to resolve disagreements."
            )))
        elif category == FailureCategory.CONTEXT_OVERFLOW:
            recommendations.append((count, (
                f"- **{label}**: Context window overflow. "
                "Recommend switching to sliding-window or retrieval memory strategy for "
                "large repositories, or truncating agent messages aggressively."
            )))
        elif category == FailureCategory.TRIAGER_WRONG_SEVERITY:
            recommendations.append((count, (
                f"- **{label}**: Triage severity frequently wrong. "
                "Recommend adding few-shot examples for severity classification in the "
                "triager prompt."
            )))

    if recommendations:
        recommendations.sort(key=lambda x: -x[0])
        lines.extend(rec for _, rec in recommendations)
    else:
        lines.append("- No dominant failure pattern detected. Failures are evenly distributed.")

    lines += [
        "",
        f"*Based on {total} classified failure(s) across all configurations.*",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_path(path: str) -> str:
    return path.lstrip("./")


def _extract_diff_files(diff: str) -> set[str]:
    """Extract normalized file paths from unified diff headers."""
    files: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            path = line[4:].strip()
            path = re.sub(r"^[ab]/", "", path)
            files.add(_normalize_path(path))
    return files


def _patch_failure_root_cause(category: FailureCategory) -> str:
    return {
        FailureCategory.PATCHER_EMPTY_PATCH: (
            "Patcher generated an empty or no-op diff, producing no code change."
        ),
        FailureCategory.PATCHER_WRONG_FILE: (
            "Patcher modified a file not involved in the vulnerability."
        ),
        FailureCategory.PATCHER_BREAKS_CODE: (
            "The generated patch introduced a regression that caused tests to fail."
        ),
        FailureCategory.PATCHER_WRONG_APPROACH: (
            "Patch used the wrong security fix approach despite targeting the right code."
        ),
        FailureCategory.PATCHER_PARTIAL_FIX: (
            "Patch addressed only part of the vulnerability; root cause was not fully resolved."
        ),
    }.get(category, "Patch quality was insufficient for acceptance.")
