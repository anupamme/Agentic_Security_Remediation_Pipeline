"""Tests for ScannerAgent, FileChunker, deduplication, and static analysis parsing."""
import json
import pathlib
from unittest.mock import AsyncMock, patch

import pytest

from multi_agent_security.agents.scanner import (
    FileChunker,
    ScanBatch,
    ScannerAgent,
    ScannerInput,
    ScannerOutput,
    _VulnCandidate,
    _deduplicate,
)
from multi_agent_security.llm_client import LLMResponse
from multi_agent_security.tools.static_analysis import (
    StaticAnalysisFinding,
    run_semgrep,
)
from multi_agent_security.types import FileContext, VulnType

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "vulnerable_repo"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file_context(path: str, lines: int, language: str = "python") -> FileContext:
    content = "\n".join(f"line{i}" for i in range(lines))
    return FileContext(path=path, content=content, language=language, line_count=lines)


def _llm_response_with(candidates: list[dict]) -> LLMResponse:
    payload = json.dumps({"vulnerabilities": candidates})
    return LLMResponse(
        content=payload,
        input_tokens=100,
        output_tokens=50,
        latency_ms=10.0,
        cost_usd=0.001,
        model="claude-sonnet-4-20250514",
    )


_SQL_INJECTION_CANDIDATE = {
    "file_path": "app.py",
    "line_start": 6,
    "line_end": 6,
    "vuln_type": "CWE-89",
    "description": "SQL injection via f-string",
    "confidence": 0.95,
    "code_snippet": "cursor.execute(f\"SELECT * FROM users WHERE username = '{username}'\")",
    "scanner_reasoning": "Unsanitised user input interpolated into SQL query",
}

# ---------------------------------------------------------------------------
# FileChunker tests
# ---------------------------------------------------------------------------

class TestFileChunker:
    def test_small_file_single_batch(self):
        fc = _make_file_context("a.py", 50)
        chunker = FileChunker()
        batches = chunker.chunk_files([fc])
        assert len(batches) == 1
        assert batches[0].files[0].path == "a.py"
        assert batches[0].chunk_index is None

    def test_multiple_small_files_batched_together(self):
        files = [_make_file_context(f"file{i}.py", 30) for i in range(3)]
        chunker = FileChunker()
        batches = chunker.chunk_files(files)
        # All three should fit in one batch (30 lines * 3 << 12000 chars)
        assert len(batches) == 1
        assert len(batches[0].files) == 3

    def test_medium_file_single_batch(self):
        fc = _make_file_context("medium.py", 300)
        chunker = FileChunker()
        batches = chunker.chunk_files([fc])
        assert len(batches) == 1
        assert len(batches[0].files) == 1
        assert batches[0].chunk_index is None

    def test_large_file_split_into_chunks(self):
        fc = _make_file_context("big.py", 2500)
        chunker = FileChunker(max_lines_per_chunk=1000, overlap=100)
        batches = chunker.chunk_files([fc])
        # 2500 lines with chunk=1000 overlap=100 → chunks at 0, 900, 1800 → 3 chunks
        assert len(batches) >= 2
        for b in batches:
            assert b.chunk_index is not None
            assert b.total_chunks == len(batches)

    def test_large_file_chunk_overlap(self):
        lines = [f"line{i}" for i in range(2200)]
        chunks = FileChunker._split_lines(lines, chunk_size=1000, overlap=100)
        assert len(chunks) >= 2
        # Second chunk should start at line 900 (1000 - 100 overlap)
        assert chunks[1][0] == "line900"

    def test_small_batch_char_budget_overflow(self):
        # Each file is 90 lines but ~200 chars per line → near budget limit
        big_content = "\n".join("x" * 200 for _ in range(90))
        files = [
            FileContext(path=f"f{i}.py", content=big_content, language="python", line_count=90)
            for i in range(4)
        ]
        chunker = FileChunker()
        batches = chunker.chunk_files(files)
        # 4 * 90*201 chars ≈ 72360 chars >> 12000 limit → should be split across batches
        assert len(batches) > 1


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDeduplication:
    def _candidate(self, file_path, line_start, line_end, confidence) -> _VulnCandidate:
        return _VulnCandidate(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            vuln_type=VulnType.SQL_INJECTION,
            description="test",
            confidence=confidence,
            code_snippet="code",
            scanner_reasoning="reason",
        )

    def test_no_duplicates(self):
        a = self._candidate("a.py", 1, 5, 0.9)
        b = self._candidate("a.py", 10, 15, 0.8)
        result = _deduplicate([a, b])
        assert len(result) == 2

    def test_overlapping_findings_keep_higher_confidence(self):
        high = self._candidate("a.py", 1, 10, 0.9)
        low = self._candidate("a.py", 3, 8, 0.6)
        result = _deduplicate([high, low])
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_overlapping_findings_replace_with_higher_confidence(self):
        low = self._candidate("a.py", 1, 10, 0.6)
        high = self._candidate("a.py", 3, 8, 0.9)
        result = _deduplicate([low, high])
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_different_files_not_deduplicated(self):
        a = self._candidate("a.py", 1, 10, 0.9)
        b = self._candidate("b.py", 1, 10, 0.8)
        result = _deduplicate([a, b])
        assert len(result) == 2

    def test_exactly_50_percent_overlap_not_deduped(self):
        # a: 0-10 (11 lines), b: 5-15 (11 lines)
        # overlap = min(10,15) - max(0,5) = 5 lines; 5/11 ≈ 0.45 < 0.5 → NOT a dup
        a = self._candidate("a.py", 0, 10, 0.9)
        b = self._candidate("a.py", 5, 15, 0.7)
        result = _deduplicate([a, b])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Static analysis parsing tests
# ---------------------------------------------------------------------------

class TestSemgrepParsing:
    @pytest.mark.asyncio
    async def test_valid_semgrep_json(self):
        semgrep_output = json.dumps({
            "results": [
                {
                    "check_id": "python.lang.security.audit.formatted-sql-query",
                    "path": "app.py",
                    "start": {"line": 6},
                    "end": {"line": 6},
                    "extra": {
                        "message": "Detected SQL injection via string formatting",
                        "severity": "ERROR",
                        "metadata": {"cwe": ["CWE-89: SQL Injection"]},
                    },
                }
            ]
        }).encode()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(semgrep_output, b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            findings = await run_semgrep("/fake/repo", "python")

        assert len(findings) == 1
        f = findings[0]
        assert f.tool == "semgrep"
        assert f.rule_id == "python.lang.security.audit.formatted-sql-query"
        assert f.file_path == "app.py"
        assert f.line_start == 6
        assert f.severity == "ERROR"
        assert "CWE-89" in f.cwe

    @pytest.mark.asyncio
    async def test_semgrep_not_installed_returns_empty(self):
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            findings = await run_semgrep("/fake/repo", "python")
        assert findings == []

    @pytest.mark.asyncio
    async def test_semgrep_no_findings(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b'{"results": []}', b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            findings = await run_semgrep("/fake/repo", "python")
        assert findings == []

    @pytest.mark.asyncio
    async def test_semgrep_timeout_returns_empty(self):
        import asyncio as _asyncio

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=_asyncio.TimeoutError)
        mock_proc.kill = AsyncMock()

        async def fake_communicate():
            return b"", b""

        mock_proc.communicate = AsyncMock(side_effect=_asyncio.TimeoutError)

        async def communicate_after_kill():
            return b"", b""

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("asyncio.wait_for", side_effect=_asyncio.TimeoutError):
            findings = await run_semgrep("/fake/repo", "python")

        assert findings == []


# ---------------------------------------------------------------------------
# ScannerAgent tests
# ---------------------------------------------------------------------------

class TestScannerAgent:
    @pytest.mark.asyncio
    async def test_mock_llm_returns_vulnerabilities(self, app_config, dry_run_llm_client):
        agent = ScannerAgent(config=app_config, llm_client=dry_run_llm_client)
        mock_response = _llm_response_with([_SQL_INJECTION_CANDIDATE])

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            output, message = await agent.run(
                ScannerInput(
                    repo_path=str(FIXTURES_DIR),
                    target_files=["app.py"],
                    language="python",
                    static_analysis_results=[],
                ),
                context=[],
            )

        assert isinstance(output, ScannerOutput)
        assert len(output.vulnerabilities) == 1
        assert output.vulnerabilities[0].id == "VULN-001"
        assert output.vulnerabilities[0].vuln_type == VulnType.SQL_INJECTION
        assert message.agent_name == "scanner"

    @pytest.mark.asyncio
    async def test_confidence_filter_removes_low_confidence(self, app_config, dry_run_llm_client):
        low_conf = {**_SQL_INJECTION_CANDIDATE, "confidence": 0.3}
        mock_response = _llm_response_with([low_conf])

        agent = ScannerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            output, _ = await agent.run(
                ScannerInput(
                    repo_path=str(FIXTURES_DIR),
                    target_files=["app.py"],
                    language="python",
                    static_analysis_results=[],
                ),
                context=[],
            )

        assert output.vulnerabilities == []

    @pytest.mark.asyncio
    async def test_confidence_boundary_exactly_05_included(self, app_config, dry_run_llm_client):
        boundary = {**_SQL_INJECTION_CANDIDATE, "confidence": 0.5}
        mock_response = _llm_response_with([boundary])

        agent = ScannerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            output, _ = await agent.run(
                ScannerInput(
                    repo_path=str(FIXTURES_DIR),
                    target_files=["app.py"],
                    language="python",
                    static_analysis_results=[],
                ),
                context=[],
            )

        assert len(output.vulnerabilities) == 1

    @pytest.mark.asyncio
    async def test_ids_assigned_sequentially(self, app_config, dry_run_llm_client):
        cred = {
            "file_path": "config.py",
            "line_start": 1,
            "line_end": 1,
            "vuln_type": "CWE-798",
            "description": "Hardcoded credentials",
            "confidence": 0.85,
            "code_snippet": 'API_KEY = "sk-..."',
            "scanner_reasoning": "Literal credential in source",
        }
        mock_response = _llm_response_with([_SQL_INJECTION_CANDIDATE, cred])

        agent = ScannerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            output, _ = await agent.run(
                ScannerInput(
                    repo_path=str(FIXTURES_DIR),
                    target_files=["app.py"],
                    language="python",
                    static_analysis_results=[],
                ),
                context=[],
            )

        ids = [v.id for v in output.vulnerabilities]
        assert ids == ["VULN-001", "VULN-002"]

    @pytest.mark.asyncio
    async def test_results_sorted_by_confidence_desc(self, app_config, dry_run_llm_client):
        low = {**_SQL_INJECTION_CANDIDATE, "confidence": 0.6}
        high = {**_SQL_INJECTION_CANDIDATE, "file_path": "config.py", "confidence": 0.95}
        mock_response = _llm_response_with([low, high])

        agent = ScannerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            output, _ = await agent.run(
                ScannerInput(
                    repo_path=str(FIXTURES_DIR),
                    target_files=["app.py"],
                    language="python",
                    static_analysis_results=[],
                ),
                context=[],
            )

        confidences = [v.confidence for v in output.vulnerabilities]
        assert confidences == sorted(confidences, reverse=True)

    @pytest.mark.asyncio
    async def test_integration_vulnerable_repo_files_scanned(self, app_config, dry_run_llm_client):
        """Run scanner on all 3 fixture files, confirm files_scanned == 3."""
        mock_response = _llm_response_with([])

        agent = ScannerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)), \
             patch(
                "multi_agent_security.agents.scanner.run_semgrep",
                AsyncMock(return_value=[]),
             ), \
             patch(
                "multi_agent_security.agents.scanner.run_dependency_audit",
                AsyncMock(return_value=[]),
             ):
            output, _ = await agent.run(
                ScannerInput(
                    repo_path=str(FIXTURES_DIR),
                    target_files=[],
                    language="python",
                ),
                context=[],
            )

        assert output.files_scanned == 3

    @pytest.mark.asyncio
    async def test_static_analysis_pre_provided_skips_tools(self, app_config, dry_run_llm_client):
        """If static_analysis_results is pre-provided, run_semgrep should not be called."""
        pre_provided = [
            StaticAnalysisFinding(
                tool="semgrep",
                rule_id="test-rule",
                file_path="app.py",
                line_start=6,
                line_end=6,
                message="SQL injection",
                severity="ERROR",
            )
        ]
        mock_response = _llm_response_with([])
        agent = ScannerAgent(config=app_config, llm_client=dry_run_llm_client)

        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)), \
             patch(
                "multi_agent_security.agents.scanner.run_semgrep",
                AsyncMock(return_value=[]),
             ) as mock_semgrep:
            await agent.run(
                ScannerInput(
                    repo_path=str(FIXTURES_DIR),
                    target_files=["app.py"],
                    language="python",
                    static_analysis_results=pre_provided,
                ),
                context=[],
            )

        mock_semgrep.assert_not_called()
