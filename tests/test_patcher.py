"""Tests for PatcherAgent, diff_generator, and test_runner."""
import json
import pathlib
import subprocess
from unittest.mock import AsyncMock, patch

import pytest

from multi_agent_security.agents.patcher import (
    PatcherAgent,
    PatcherInput,
    PatcherOutput,
    _strip_code_fences,
)
from multi_agent_security.config import AgentsConfig, PatcherAgentConfig
from multi_agent_security.llm_client import LLMResponse
from multi_agent_security.tools.diff_generator import (
    apply_patch_to_content,
    generate_unified_diff,
    validate_diff,
)
from multi_agent_security.tools.test_runner import TestResult, run_tests_on_patched_code
from multi_agent_security.types import (
    FixStrategy,
    Patch,
    TriageResult,
    Vulnerability,
    VulnSeverity,
    VulnType,
)

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "vulnerable_repo"

# The SQL injection fixture from tests/fixtures/vulnerable_repo/app.py
_SQL_ORIGINAL = FIXTURES_DIR.joinpath("app.py").read_text()

# A corrected version using a parameterised query
_SQL_PATCHED = """\
import sqlite3


def get_user(username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    return cursor.fetchone()
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vulnerability(
    id: str = "VULN-001",
    file_path: str = "app.py",
    line_start: int = 7,
    line_end: int = 7,
    vuln_type: VulnType = VulnType.SQL_INJECTION,
    description: str = "SQL injection via f-string interpolation",
    code_snippet: str = "cursor.execute(f\"SELECT * FROM users WHERE username = '{username}'\")",
    scanner_reasoning: str = "Unsanitised user input interpolated into SQL query",
) -> Vulnerability:
    return Vulnerability(
        id=id,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        vuln_type=vuln_type,
        description=description,
        confidence=0.95,
        code_snippet=code_snippet,
        scanner_reasoning=scanner_reasoning,
    )


def _make_triage(
    vuln_id: str = "VULN-001",
    severity: VulnSeverity = VulnSeverity.HIGH,
    fix_strategy: FixStrategy = FixStrategy.ONE_LINER,
    complexity: str = "low",
) -> TriageResult:
    return TriageResult(
        vuln_id=vuln_id,
        severity=severity,
        exploitability_score=0.9,
        fix_strategy=fix_strategy,
        estimated_complexity=complexity,
        triage_reasoning="Direct SQL injection; easily exploitable from external input.",
    )


def _llm_patch_response(
    patched_code: str,
    patch_reasoning: str = "Fixed by using parameterised query.",
    test_suggestion: str | None = "Test with malicious username input.",
    alternative_approaches: list[str] | None = None,
) -> LLMResponse:
    payload = json.dumps({
        "patched_code": patched_code,
        "patch_reasoning": patch_reasoning,
        "test_suggestion": test_suggestion,
        "alternative_approaches": alternative_approaches or [],
    })
    return LLMResponse(
        content=payload,
        input_tokens=500,
        output_tokens=150,
        latency_ms=120.0,
        cost_usd=0.0038,
        model="claude-sonnet-4-20250514",
    )


# ---------------------------------------------------------------------------
# TestDiffGenerator
# ---------------------------------------------------------------------------

class TestDiffGenerator:
    def test_generate_unified_diff_contains_hunk_header(self):
        diff = generate_unified_diff("a\nb\nc\n", "a\nB\nc\n", "test.py")
        assert "@@" in diff

    def test_generate_unified_diff_shows_removal(self):
        diff = generate_unified_diff("a\nb\nc\n", "a\nB\nc\n", "test.py")
        removed = [l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
        assert any("b" in l for l in removed)

    def test_generate_unified_diff_shows_addition(self):
        diff = generate_unified_diff("a\nb\nc\n", "a\nB\nc\n", "test.py")
        added = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
        assert any("B" in l for l in added)

    def test_generate_unified_diff_file_headers(self):
        diff = generate_unified_diff("x\n", "y\n", "src/foo.py")
        assert "a/src/foo.py" in diff
        assert "b/src/foo.py" in diff

    def test_generate_unified_diff_identical_input_returns_empty(self):
        diff = generate_unified_diff("same\ncontent\n", "same\ncontent\n", "f.py")
        assert diff == ""

    def test_apply_patch_to_content_single_line_replacement(self):
        original = "line1\nline2\nline3\n"
        result = apply_patch_to_content(original, "LINE2\n", line_start=2, line_end=2)
        assert result == "line1\nLINE2\nline3\n"

    def test_apply_patch_to_content_preserves_surrounding_lines(self):
        original = "a\nb\nc\nd\n"
        result = apply_patch_to_content(original, "B\n", line_start=2, line_end=2)
        assert result.startswith("a\n")
        assert result.endswith("c\nd\n")

    def test_apply_patch_to_content_line_start_1(self):
        original = "first\nsecond\n"
        result = apply_patch_to_content(original, "FIRST\n", line_start=1, line_end=1)
        assert result == "FIRST\nsecond\n"

    def test_apply_patch_to_content_multi_line_snippet(self):
        original = "a\nb\nc\nd\n"
        replacement = "X\nY\n"
        result = apply_patch_to_content(original, replacement, line_start=2, line_end=3)
        assert result == "a\nX\nY\nd\n"

    def test_validate_diff_valid(self):
        diff = generate_unified_diff("old\n", "new\n", "f.py")
        assert validate_diff(diff) is True

    def test_validate_diff_empty_string(self):
        assert validate_diff("") is False

    def test_validate_diff_whitespace_only(self):
        assert validate_diff("   \n  \n") is False

    def test_validate_diff_no_change_lines(self):
        # Only header lines, no +/-
        fake = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n context\n"
        assert validate_diff(fake) is False

    def test_validate_diff_massive_deletion_rejected(self):
        # A diff that removes all 10 original lines and adds nothing
        header = "--- a/f.py\n+++ b/f.py\n@@ -1,10 +0,0 @@\n"
        removals = "".join(f"-line{i}\n" for i in range(10))
        diff = header + removals
        assert validate_diff(diff) is False

    def test_validate_diff_moderate_deletion_accepted(self):
        # Remove 2 of 10 lines — well under 50%
        header = "--- a/f.py\n+++ b/f.py\n@@ -1,10 +1,8 @@\n"
        lines = " context\n" * 8 + "-removed1\n-removed2\n"
        diff = header + lines
        assert validate_diff(diff) is True


# ---------------------------------------------------------------------------
# TestPatcherAgent
# ---------------------------------------------------------------------------

class TestPatcherAgent:
    async def test_sql_injection_fix_produces_valid_diff(
        self, app_config, dry_run_llm_client
    ):
        agent = PatcherAgent(config=app_config, llm_client=dry_run_llm_client)
        inp = PatcherInput(
            vulnerability=_make_vulnerability(),
            triage=_make_triage(),
            file_content=_SQL_ORIGINAL,
            repo_path=str(FIXTURES_DIR),
            language="python",
        )
        mock_resp = _llm_patch_response(_SQL_PATCHED)

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            output, message = await agent.run(inp, context=[])

        assert isinstance(output, PatcherOutput)
        assert validate_diff(output.patch.unified_diff)
        assert output.patch.vuln_id == "VULN-001"

    async def test_agent_message_fields_populated(
        self, app_config, dry_run_llm_client
    ):
        agent = PatcherAgent(config=app_config, llm_client=dry_run_llm_client)
        inp = PatcherInput(
            vulnerability=_make_vulnerability(),
            triage=_make_triage(),
            file_content=_SQL_ORIGINAL,
            repo_path=str(FIXTURES_DIR),
            language="python",
        )
        mock_resp = _llm_patch_response(_SQL_PATCHED)

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            _, message = await agent.run(inp, context=[])

        assert message.agent_name == "patcher"
        assert message.token_count_input == 500
        assert message.token_count_output == 150
        assert message.cost_usd > 0
        assert message.latency_ms >= 0

    async def test_revision_feedback_appears_in_system_prompt(
        self, app_config, dry_run_llm_client
    ):
        agent = PatcherAgent(config=app_config, llm_client=dry_run_llm_client)
        prev_patch = Patch(
            vuln_id="VULN-001",
            file_path="app.py",
            original_code=_SQL_ORIGINAL,
            patched_code=_SQL_ORIGINAL,  # bad patch
            unified_diff="",
            patch_reasoning="Previous attempt",
        )
        inp = PatcherInput(
            vulnerability=_make_vulnerability(),
            triage=_make_triage(),
            file_content=_SQL_ORIGINAL,
            repo_path=str(FIXTURES_DIR),
            language="python",
            revision_feedback="The patch does not use parameterised queries.",
            previous_patch=prev_patch,
        )

        captured: dict = {}

        async def capture(system_prompt, user_prompt, **kwargs):
            captured["system_prompt"] = system_prompt
            return _llm_patch_response(_SQL_PATCHED)

        with patch.object(dry_run_llm_client, "complete", AsyncMock(side_effect=capture)):
            await agent.run(inp, context=[])

        assert "reviewer has rejected" in captured["system_prompt"].lower()
        assert "parameterised queries" in captured["system_prompt"]

    async def test_format_detection_unified_diff_branch(
        self, app_config, dry_run_llm_client
    ):
        """When LLM returns a unified diff directly, it is used as-is."""
        agent = PatcherAgent(config=app_config, llm_client=dry_run_llm_client)
        # Craft a fake unified diff
        fake_diff = (
            "--- a/app.py\n+++ b/app.py\n"
            "@@ -7,1 +7,1 @@\n"
            "-    cursor.execute(f\"SELECT * FROM users WHERE username = '{username}'\")\n"
            '+    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))\n'
        )
        inp = PatcherInput(
            vulnerability=_make_vulnerability(),
            triage=_make_triage(),
            file_content=_SQL_ORIGINAL,
            repo_path=str(FIXTURES_DIR),
            language="python",
        )
        mock_resp = _llm_patch_response(fake_diff)

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            output, _ = await agent.run(inp, context=[])

        assert output.patch.unified_diff == fake_diff

    async def test_format_detection_full_file_branch(
        self, app_config, dry_run_llm_client
    ):
        """When LLM returns the full patched file (ratio ≈ 1), it is used directly."""
        agent = PatcherAgent(config=app_config, llm_client=dry_run_llm_client)
        inp = PatcherInput(
            vulnerability=_make_vulnerability(),
            triage=_make_triage(),
            file_content=_SQL_ORIGINAL,
            repo_path=str(FIXTURES_DIR),
            language="python",
        )
        # _SQL_PATCHED is approximately the same length as _SQL_ORIGINAL
        mock_resp = _llm_patch_response(_SQL_PATCHED)

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            output, _ = await agent.run(inp, context=[])

        assert output.patch.patched_code == _SQL_PATCHED

    async def test_format_detection_snippet_branch(
        self, app_config, dry_run_llm_client
    ):
        """When LLM returns only a short snippet, it is spliced into the original."""
        agent = PatcherAgent(config=app_config, llm_client=dry_run_llm_client)
        # A single-line replacement (much shorter than the full file)
        snippet = '    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))'
        inp = PatcherInput(
            vulnerability=_make_vulnerability(line_start=7, line_end=7),
            triage=_make_triage(),
            file_content=_SQL_ORIGINAL,
            repo_path=str(FIXTURES_DIR),
            language="python",
        )
        mock_resp = _llm_patch_response(snippet)

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            output, _ = await agent.run(inp, context=[])

        # Snippet replaces line 7 — the rest of the original file is preserved
        result_lines = output.patch.patched_code.splitlines()
        assert "?" in result_lines[6]          # line 7 (0-indexed: 6) is the fixed line
        assert result_lines[0] == "import sqlite3"  # line 1 unchanged

    async def test_parse_error_fallback(self, app_config, dry_run_llm_client):
        """Invalid JSON from LLM → fallback patch with original code unchanged."""
        agent = PatcherAgent(config=app_config, llm_client=dry_run_llm_client)
        inp = PatcherInput(
            vulnerability=_make_vulnerability(),
            triage=_make_triage(),
            file_content=_SQL_ORIGINAL,
            repo_path=str(FIXTURES_DIR),
            language="python",
        )
        bad_resp = LLMResponse(
            content="this is not json at all",
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            cost_usd=0.0,
            model="claude-sonnet-4-20250514",
        )

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=bad_resp)):
            output, _ = await agent.run(inp, context=[])

        assert output.patch.original_code == _SQL_ORIGINAL
        assert output.patch.patched_code == _SQL_ORIGINAL

    async def test_no_tests_run_when_config_disabled(
        self, app_config, dry_run_llm_client
    ):
        """When run_tests=False in config, test runner is never called."""
        config_no_tests = app_config.model_copy(
            update={
                "agents": app_config.agents.model_copy(
                    update={"patcher": PatcherAgentConfig(run_tests=False)}
                )
            }
        )
        agent = PatcherAgent(config=config_no_tests, llm_client=dry_run_llm_client)
        inp = PatcherInput(
            vulnerability=_make_vulnerability(),
            triage=_make_triage(),
            file_content=_SQL_ORIGINAL,
            repo_path=str(FIXTURES_DIR),
            language="python",
        )
        mock_resp = _llm_patch_response(_SQL_PATCHED)

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            with patch(
                "multi_agent_security.agents.patcher.run_tests_on_patched_code"
            ) as mock_runner:
                await agent.run(inp, context=[])

        mock_runner.assert_not_called()

    def test_strip_code_fences_removes_fences(self):
        fenced = "```python\nprint('hello')\n```"
        assert _strip_code_fences(fenced) == "print('hello')\n"

    def test_strip_code_fences_no_fences_unchanged(self):
        plain = "print('hello')\n"
        assert _strip_code_fences(plain) == plain


# ---------------------------------------------------------------------------
# TestTestRunner
# ---------------------------------------------------------------------------

class TestTestRunner:
    async def test_passing_tests(self, tmp_path):
        # Create a minimal Python repo with a passing test
        (tmp_path / "mymod.py").write_text("def add(a, b):\n    return a + b\n")
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        (test_dir / "__init__.py").write_text("")
        (test_dir / "test_add.py").write_text(
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))\n"
            "from mymod import add\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
        )
        patch_obj = Patch(
            vuln_id="VULN-001",
            file_path="mymod.py",
            original_code="def add(a, b):\n    return a + b\n",
            patched_code="def add(a, b):\n    return a + b  # patched\n",
            unified_diff="",
            patch_reasoning="test",
        )

        result = await run_tests_on_patched_code(
            str(tmp_path), patch_obj, "python", timeout_seconds=60
        )

        assert result.passed is True
        assert result.error is None
        assert result.tests_run >= 1

    async def test_failing_tests(self, tmp_path):
        # Create a repo where the test always fails
        (tmp_path / "mymod.py").write_text("def add(a, b):\n    return a + b\n")
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        (test_dir / "__init__.py").write_text("")
        (test_dir / "test_fail.py").write_text(
            "def test_always_fails():\n"
            "    assert False, 'intentional failure'\n"
        )
        patch_obj = Patch(
            vuln_id="VULN-001",
            file_path="mymod.py",
            original_code="def add(a, b):\n    return a + b\n",
            patched_code="def add(a, b):\n    return a + b\n",
            unified_diff="",
            patch_reasoning="test",
        )

        result = await run_tests_on_patched_code(
            str(tmp_path), patch_obj, "python", timeout_seconds=60
        )

        assert result.passed is False
        assert result.tests_failed >= 1

    async def test_output_truncated_to_2000_chars(self, tmp_path):
        # Generate a test that produces very long output
        (tmp_path / "mymod.py").write_text("")
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        (test_dir / "__init__.py").write_text("")
        # print a 5000-char string in the test to inflate output
        (test_dir / "test_verbose.py").write_text(
            "def test_verbose():\n"
            "    print('x' * 5000)\n"
        )
        patch_obj = Patch(
            vuln_id="VULN-001",
            file_path="mymod.py",
            original_code="",
            patched_code="",
            unified_diff="",
            patch_reasoning="test",
        )

        result = await run_tests_on_patched_code(
            str(tmp_path), patch_obj, "python", timeout_seconds=60
        )

        assert len(result.output) <= 2000

    async def test_unsupported_language(self, tmp_path):
        patch_obj = Patch(
            vuln_id="VULN-001",
            file_path="main.rb",
            original_code="",
            patched_code="",
            unified_diff="",
            patch_reasoning="test",
        )

        result = await run_tests_on_patched_code(
            str(tmp_path), patch_obj, "ruby", timeout_seconds=10
        )

        assert result.passed is False
        assert result.error is not None
        assert "ruby" in result.error.lower() or "unsupported" in result.error.lower()

    async def test_timeout_returns_error(self, tmp_path):
        patch_obj = Patch(
            vuln_id="VULN-001",
            file_path="mymod.py",
            original_code="",
            patched_code="",
            unified_diff="",
            patch_reasoning="test",
        )

        with patch(
            "multi_agent_security.tools.test_runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["pytest"], timeout=1),
        ):
            result = await run_tests_on_patched_code(
                str(tmp_path), patch_obj, "python", timeout_seconds=1
            )

        assert result.passed is False
        assert result.error is not None
        assert "timed out" in result.error.lower()


# ---------------------------------------------------------------------------
# TestPatcherIntegration
# ---------------------------------------------------------------------------

class TestPatcherIntegration:
    async def test_triager_output_fed_into_patcher(
        self, app_config, dry_run_llm_client
    ):
        """End-to-end: TriageResult for SQL injection fixture → PatcherAgent → valid patch."""
        vuln = _make_vulnerability(
            id="VULN-001",
            file_path="app.py",
            line_start=7,
            line_end=7,
            description="SQL injection via f-string",
            code_snippet=(
                "cursor.execute(f\"SELECT * FROM users WHERE username = '{username}'\")"
            ),
        )
        triage = _make_triage("VULN-001")

        agent = PatcherAgent(config=app_config, llm_client=dry_run_llm_client)
        mock_resp = _llm_patch_response(_SQL_PATCHED)

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_resp)):
            output, message = await agent.run(
                PatcherInput(
                    vulnerability=vuln,
                    triage=triage,
                    file_content=_SQL_ORIGINAL,
                    repo_path=str(FIXTURES_DIR),
                    language="python",
                ),
                context=[],
            )

        assert isinstance(output, PatcherOutput)
        assert output.patch.vuln_id == "VULN-001"
        # Parameterised query should appear in the patched code
        assert "?" in output.patch.patched_code
        # The diff should be non-empty and valid
        assert validate_diff(output.patch.unified_diff)
        # The original f-string should NOT be in the patched code
        assert "f\"SELECT" not in output.patch.patched_code
        # AgentMessage sanity checks
        assert message.agent_name == "patcher"
        assert message.cost_usd > 0
