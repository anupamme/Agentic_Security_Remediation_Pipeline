import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from multi_agent_security.agents.base import BaseAgent
from multi_agent_security.tools.diff_generator import (
    apply_patch_to_content,
    generate_unified_diff,
    validate_diff,
)
from multi_agent_security.tools.test_runner import run_tests_on_patched_code
from multi_agent_security.types import (
    AgentMessage,
    Patch,
    TriageResult,
    Vulnerability,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are a security patch generator. Your job is to write a minimal, correct code patch that fixes a security vulnerability.

Rules:
1. **Minimal changes only.** Fix the vulnerability without refactoring unrelated code. Maintainers reject PRs that change too much.
2. **Preserve existing style.** Match the indentation, naming conventions, and patterns in the existing code.
3. **Don't break functionality.** The patched code must do the same thing as the original, minus the vulnerability.
4. **Consider the fix strategy.** The triager has recommended a strategy — follow it unless you have a strong reason not to.
5. **Explain your reasoning.** Describe why your fix is correct and what attack vector it closes.
6. **Output the FULL patched file** in the `patched_code` field, not just the changed lines.

Respond with JSON only, matching the provided schema.\
"""

_REVISION_SECTION = """

IMPORTANT: A reviewer has rejected your previous patch with this feedback:
{feedback}

Previous patched code:
```
{previous_patched_code}
```

Address ALL of the reviewer's concerns in this revision.\
"""


# ---------------------------------------------------------------------------
# Input / output models
# ---------------------------------------------------------------------------

class PatcherInput(BaseModel):
    vulnerability: Vulnerability
    triage: TriageResult
    file_content: str           # Full content of the file to patch
    repo_path: str
    language: str
    revision_feedback: Optional[str] = None   # From reviewer, if this is a retry
    previous_patch: Optional[Patch] = None    # Previous rejected patch, if retry


class PatcherOutput(BaseModel):
    patch: Patch
    test_suggestion: Optional[str] = None
    alternative_approaches: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Intermediate LLM response model
# ---------------------------------------------------------------------------

class _LLMPatchResponse(BaseModel):
    patched_code: str       # Full file, snippet, or unified diff — detected automatically
    patch_reasoning: str
    test_suggestion: Optional[str] = None
    alternative_approaches: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PatcherAgent
# ---------------------------------------------------------------------------

class PatcherAgent(BaseAgent):
    name = "patcher"

    async def run(
        self, input_data: BaseModel, context: list[AgentMessage]
    ) -> tuple[PatcherOutput, AgentMessage]:
        assert isinstance(input_data, PatcherInput)
        inp: PatcherInput = input_data
        vuln = inp.vulnerability

        # Build system prompt — append revision section when feedback is present
        system_prompt = _SYSTEM_PROMPT_BASE
        if inp.revision_feedback:
            prev_code = inp.previous_patch.patched_code if inp.previous_patch else ""
            system_prompt += _REVISION_SECTION.format(
                feedback=inp.revision_feedback,
                previous_patched_code=prev_code,
            )

        user_prompt = self._build_user_prompt(inp)

        response = await self.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=_LLMPatchResponse,
        )

        metrics = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
        }

        # Parse LLM response with graceful fallback
        try:
            llm_result = _LLMPatchResponse.model_validate_json(response.content)
        except Exception as exc:
            logger.warning("Failed to parse LLM patch response for %s: %s", vuln.id, exc)
            llm_result = _LLMPatchResponse(
                patched_code=inp.file_content,  # no-op fallback
                patch_reasoning="LLM response parse error — original code returned unchanged.",
            )

        # Strip markdown code fences if the LLM wrapped the code
        raw = _strip_code_fences(llm_result.patched_code)

        # Determine final patched file content and unified diff
        is_unified_diff = "---" in raw and "+++" in raw and "@@" in raw

        if is_unified_diff:
            # Branch 1: LLM returned a unified diff directly
            logger.warning(
                "LLM returned a unified diff for %s; using it directly. "
                "patched_code will equal original as best-effort fallback.",
                vuln.id,
            )
            unified_diff = raw
            final_patched_code = inp.file_content  # best-effort: diff is authoritative

        else:
            ratio = len(raw) / max(len(inp.file_content), 1)
            if 0.8 <= ratio <= 1.2:
                # Branch 2: LLM returned the full patched file
                final_patched_code = raw
            else:
                # Branch 3: LLM returned a snippet — splice into original
                logger.info(
                    "LLM returned a code snippet for %s (length ratio=%.2f); "
                    "applying to original at lines %d-%d",
                    vuln.id, ratio, vuln.line_start, vuln.line_end,
                )
                final_patched_code = apply_patch_to_content(
                    inp.file_content, raw, vuln.line_start, vuln.line_end
                )

            unified_diff = generate_unified_diff(
                inp.file_content, final_patched_code, vuln.file_path
            )

        if not validate_diff(unified_diff):
            logger.warning("Generated diff for %s failed validation", vuln.id)

        patch = Patch(
            vuln_id=vuln.id,
            file_path=vuln.file_path,
            original_code=inp.file_content,
            patched_code=final_patched_code,
            unified_diff=unified_diff,
            patch_reasoning=llm_result.patch_reasoning,
        )

        # Optionally run the test suite on the patched code
        if self.config.agents.patcher.run_tests and _repo_has_tests(inp.repo_path):
            test_result = await run_tests_on_patched_code(
                inp.repo_path, patch, inp.language
            )
            if not test_result.passed:
                logger.warning(
                    "Tests failed after patching %s: %s",
                    vuln.id,
                    test_result.output[:200],
                )

        output = PatcherOutput(
            patch=patch,
            test_suggestion=llm_result.test_suggestion,
            alternative_approaches=llm_result.alternative_approaches,
        )

        message = AgentMessage(
            agent_name=type(self).name,
            timestamp=datetime.now(timezone.utc),
            content=output.model_dump_json(),
            token_count_input=metrics["input_tokens"],
            token_count_output=metrics["output_tokens"],
            latency_ms=metrics["latency_ms"],
            cost_usd=metrics["cost_usd"],
        )

        return output, message

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _build_user_prompt(self, inp: PatcherInput) -> str:
        vuln = inp.vulnerability
        triage = inp.triage
        parts = [
            f"## Vulnerability: {vuln.id}",
            f"- Type: {vuln.vuln_type.value}",
            f"- File: {vuln.file_path}, Lines {vuln.line_start}-{vuln.line_end}",
            f"- Severity: {triage.severity.value}",
            f"- Recommended Fix Strategy: {triage.fix_strategy.value}",
            f"- Estimated Complexity: {triage.estimated_complexity}",
            "",
            "### Vulnerability Description:",
            vuln.description,
            "",
            f"### Vulnerable Code Snippet:",
            f"```{inp.language}",
            vuln.code_snippet,
            "```",
            "",
            f"### Scanner Reasoning:",
            vuln.scanner_reasoning,
            "",
            f"### Triage Reasoning:",
            triage.triage_reasoning,
            "",
            f"### Full File Content ({vuln.file_path}):",
            f"```{inp.language}",
            inp.file_content,
            "```",
        ]

        if inp.previous_patch:
            parts += [
                "",
                "### Previous Patch (rejected):",
                "```diff",
                inp.previous_patch.unified_diff,
                "```",
            ]

        parts += [
            "",
            "Generate a patch that fixes this vulnerability.",
        ]

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(code: str) -> str:
    """Remove leading/trailing markdown code fences (```lang...```) if present."""
    stripped = code.strip()
    # Match opening fence with optional language tag, then content, then closing fence
    m = re.match(r"^```[^\n]*\n(.*?)```\s*$", stripped, re.DOTALL)
    if m:
        return m.group(1)
    return code


def _repo_has_tests(repo_path: str) -> bool:
    """Return True if the repo appears to have a test suite."""
    test_dirs = {"tests", "test", "__tests__", "spec"}
    try:
        for entry in os.listdir(repo_path):
            if entry in test_dirs and os.path.isdir(os.path.join(repo_path, entry)):
                return True
            if entry.startswith("test_") and entry.endswith(".py"):
                return True
    except OSError:
        pass
    return False
