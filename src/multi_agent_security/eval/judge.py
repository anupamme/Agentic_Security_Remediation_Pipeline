"""
LLM-as-Judge for evaluating patch correctness and quality.
"""

from pydantic import BaseModel, Field

from multi_agent_security.llm_client import LLMClient, LLMResponse
from multi_agent_security.types import (
    BenchmarkExample,
    CalibrationResult,
    JudgeScore,
    Patch,
    Vulnerability,
)

_CORRECTNESS_SYSTEM_PROMPT = """You are an expert security engineer evaluating a code patch. You must compare a generated patch against a ground-truth patch that was actually merged by the repository maintainers.

Evaluate on three dimensions:
1. **Correctness** (0.0-1.0): Does the generated patch address the same vulnerability as the ground truth? It doesn't need to be identical — different approaches to fixing the same vuln are fine.
2. **Completeness** (0.0-1.0): Does the generated patch fully fix the vulnerability, or is it a partial fix?
3. **Safety** (0.0-1.0): Does the generated patch introduce any new security issues?

Respond with JSON only:
{"correctness": 0.0-1.0, "completeness": 0.0-1.0, "safety": 0.0-1.0, "reasoning": "..."}"""

_QUALITY_SYSTEM_PROMPT = """You are an expert security engineer evaluating the quality of a code patch.

Evaluate on three dimensions:
1. **Correctness** (0.0-1.0): Does the patch correctly fix the described vulnerability using secure coding best practices?
2. **Completeness** (0.0-1.0): Does the patch fully address the vulnerability, or is it a partial fix?
3. **Safety** (0.0-1.0): Does the patch introduce any new security issues or regressions?

Respond with JSON only:
{"correctness": 0.0-1.0, "completeness": 0.0-1.0, "safety": 0.0-1.0, "reasoning": "..."}"""


class _RawJudgeResponse(BaseModel):
    correctness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)
    reasoning: str


class LLMJudge:
    def __init__(self, llm_client: LLMClient, model: str = "claude-sonnet-4-20250514"):
        self._llm = llm_client
        self._model = model

    async def judge_patch_correctness(
        self,
        vulnerability: Vulnerability,
        generated_patch: Patch,
        ground_truth_diff: str,
        language: str,
    ) -> JudgeScore:
        """Ask the judge LLM to compare the generated patch against the ground truth."""
        user_prompt = (
            f"## Vulnerability: {vulnerability.vuln_type}\n"
            f"## File: {vulnerability.file_path}\n"
            f"## Description: {vulnerability.description}\n\n"
            f"### Generated Patch:\n```diff\n{generated_patch.unified_diff}\n```\n\n"
            f"### Ground Truth Patch (merged by maintainers):\n```diff\n{ground_truth_diff}\n```\n\n"
            "Compare these patches and score the generated one."
        )
        response: LLMResponse = await self._llm.complete(
            system_prompt=_CORRECTNESS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format=_RawJudgeResponse,
            agent_name="llm_judge",
        )
        raw = _RawJudgeResponse.model_validate_json(response.content)
        return JudgeScore(
            correctness=raw.correctness,
            completeness=raw.completeness,
            safety=raw.safety,
            reasoning=raw.reasoning,
        )

    async def judge_patch_quality(
        self,
        vulnerability: Vulnerability,
        patch: Patch,
        original_code: str,
        language: str,
    ) -> JudgeScore:
        """Evaluate patch quality independent of ground truth."""
        user_prompt = (
            f"## Vulnerability: {vulnerability.vuln_type}\n"
            f"## File: {vulnerability.file_path}\n"
            f"## Description: {vulnerability.description}\n\n"
            f"### Original Code:\n```{language}\n{original_code}\n```\n\n"
            f"### Generated Patch:\n```diff\n{patch.unified_diff}\n```\n\n"
            "Evaluate the quality of this patch."
        )
        response: LLMResponse = await self._llm.complete(
            system_prompt=_QUALITY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format=_RawJudgeResponse,
            agent_name="llm_judge",
        )
        raw = _RawJudgeResponse.model_validate_json(response.content)
        return JudgeScore(
            correctness=raw.correctness,
            completeness=raw.completeness,
            safety=raw.safety,
            reasoning=raw.reasoning,
        )


async def calibrate_judge(
    judge: LLMJudge,
    calibration_examples: list[tuple[BenchmarkExample, Patch, float]],
) -> CalibrationResult:
    """
    Run the judge on examples with known human scores.
    Compute Pearson and Spearman correlations between judge and human scores.
    """
    from scipy import stats as scipy_stats

    human_scores: list[float] = []
    judge_scores: list[float] = []
    per_example: list[dict] = []

    for example, patch, human_score in calibration_examples:
        # Build a minimal Vulnerability from the BenchmarkExample for the judge
        vuln = Vulnerability(
            id=f"CALIB-{example.id}",
            file_path=example.vulnerable_files[0] if example.vulnerable_files else "unknown",
            line_start=1,
            line_end=1,
            vuln_type=example.vuln_type,
            description=f"Calibration example {example.id}",
            confidence=1.0,
            code_snippet="",
            scanner_reasoning="calibration",
        )
        score = await judge.judge_patch_correctness(
            vulnerability=vuln,
            generated_patch=patch,
            ground_truth_diff=example.ground_truth_diff,
            language=example.language,
        )
        j_score = score.correctness
        human_scores.append(human_score)
        judge_scores.append(j_score)
        per_example.append({
            "example_id": example.id,
            "human_score": human_score,
            "judge_score": j_score,
            "delta": abs(j_score - human_score),
        })

    if len(human_scores) < 2:
        pearson_r = 0.0
        spearman_r = 0.0
    else:
        pearson_r, _ = scipy_stats.pearsonr(human_scores, judge_scores)
        spearman_r, _ = scipy_stats.spearmanr(human_scores, judge_scores)

    mae = sum(e["delta"] for e in per_example) / len(per_example) if per_example else 0.0

    return CalibrationResult(
        pearson_r=float(pearson_r),
        spearman_r=float(spearman_r),
        mean_absolute_error=mae,
        n_examples=len(per_example),
        per_example=per_example,
    )
