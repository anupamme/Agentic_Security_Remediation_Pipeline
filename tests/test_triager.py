"""Tests for TriagerAgent, detect_framework, and extract_repo_metadata."""
import json
import pathlib
from unittest.mock import AsyncMock, patch

import pytest

from multi_agent_security.agents.triager import (
    RepoMetadata,
    TriagerAgent,
    TriagerInput,
    TriagerOutput,
    _DEFAULT_COMPLEXITY,
    _DEFAULT_EXPLOITABILITY,
    _DEFAULT_REASONING,
    _DEFAULT_SEVERITY,
    _DEFAULT_STRATEGY,
)
from multi_agent_security.llm_client import LLMResponse
from multi_agent_security.tools.code_parser import detect_framework, extract_repo_metadata
from multi_agent_security.types import (
    FixStrategy,
    TriageResult,
    Vulnerability,
    VulnSeverity,
    VulnType,
)

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "vulnerable_repo"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vulnerability(
    id: str,
    vuln_type: VulnType = VulnType.SQL_INJECTION,
    confidence: float = 0.9,
    file_path: str = "app.py",
) -> Vulnerability:
    return Vulnerability(
        id=id,
        file_path=file_path,
        line_start=1,
        line_end=5,
        vuln_type=vuln_type,
        description=f"Test vulnerability {id}",
        confidence=confidence,
        code_snippet=f"# vulnerable code for {id}",
        scanner_reasoning=f"Reason for {id}",
    )


def _make_repo_metadata(
    repo_name: str = "test-repo",
    language: str = "python",
    framework: str | None = None,
    has_tests: bool = True,
    file_count: int = 5,
) -> RepoMetadata:
    return RepoMetadata(
        repo_name=repo_name,
        language=language,
        framework=framework,
        has_tests=has_tests,
        file_count=file_count,
        dependency_files=["requirements.txt"],
    )


def _llm_triage_response(
    items: list[dict],
    priority_order: list[str],
    summary: str = "Test triage summary",
) -> LLMResponse:
    payload = json.dumps({
        "triage_results": items,
        "priority_order": priority_order,
        "summary": summary,
    })
    return LLMResponse(
        content=payload,
        input_tokens=200,
        output_tokens=100,
        latency_ms=50.0,
        cost_usd=0.002,
        model="claude-sonnet-4-20250514",
    )


def _triage_item(vuln_id: str, severity: str = "high", exploitability: float = 0.8) -> dict:
    return {
        "vuln_id": vuln_id,
        "severity": severity,
        "exploitability_score": exploitability,
        "fix_strategy": "refactor",
        "estimated_complexity": "medium",
        "triage_reasoning": f"Reasoning for {vuln_id}",
    }


# ---------------------------------------------------------------------------
# TriagerAgent tests
# ---------------------------------------------------------------------------

class TestTriagerAgent:
    @pytest.mark.asyncio
    async def test_mock_llm_returns_3_vulns_all_mapped(self, app_config, dry_run_llm_client):
        vulns = [_make_vulnerability(f"VULN-{i:03d}") for i in range(1, 4)]
        ids = [v.id for v in vulns]
        items = [_triage_item(vid) for vid in ids]
        mock_response = _llm_triage_response(items, priority_order=ids)

        agent = TriagerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            output, message = await agent.run(
                TriagerInput(vulnerabilities=vulns, repo_metadata=_make_repo_metadata()),
                context=[],
            )

        assert isinstance(output, TriagerOutput)
        assert len(output.triage_results) == 3
        result_ids = {r.vuln_id for r in output.triage_results}
        assert result_ids == set(ids)
        assert message.agent_name == "triager"

    @pytest.mark.asyncio
    async def test_missing_vuln_in_llm_output_gets_defaults(self, app_config, dry_run_llm_client):
        vulns = [_make_vulnerability(f"VULN-{i:03d}") for i in range(1, 4)]
        # LLM only returns results for VULN-001 and VULN-002, omits VULN-003
        items = [_triage_item("VULN-001"), _triage_item("VULN-002")]
        mock_response = _llm_triage_response(
            items, priority_order=["VULN-001", "VULN-002", "VULN-003"]
        )

        agent = TriagerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            output, _ = await agent.run(
                TriagerInput(vulnerabilities=vulns, repo_metadata=_make_repo_metadata()),
                context=[],
            )

        assert len(output.triage_results) == 3
        missing = next(r for r in output.triage_results if r.vuln_id == "VULN-003")
        assert missing.severity == _DEFAULT_SEVERITY
        assert missing.exploitability_score == _DEFAULT_EXPLOITABILITY
        assert missing.fix_strategy == _DEFAULT_STRATEGY
        assert missing.estimated_complexity == _DEFAULT_COMPLEXITY
        assert missing.triage_reasoning == _DEFAULT_REASONING

    @pytest.mark.asyncio
    async def test_exploitability_score_clamped_above_one(self, app_config, dry_run_llm_client):
        # Pydantic will reject scores >1 in the schema, so we simulate a parse error
        # by providing a valid response with boundary value 1.0
        vulns = [_make_vulnerability("VULN-001")]
        items = [_triage_item("VULN-001", exploitability=1.0)]
        mock_response = _llm_triage_response(items, priority_order=["VULN-001"])

        agent = TriagerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            output, _ = await agent.run(
                TriagerInput(vulnerabilities=vulns, repo_metadata=_make_repo_metadata()),
                context=[],
            )

        result = output.triage_results[0]
        assert 0.0 <= result.exploitability_score <= 1.0

    @pytest.mark.asyncio
    async def test_exploitability_score_out_of_range_clamped(self, app_config, dry_run_llm_client):
        """Test _clamp_exploitability directly when score is out of range."""
        from multi_agent_security.agents.triager import _clamp_exploitability
        assert _clamp_exploitability(1.5, "VULN-001") == 1.0
        assert _clamp_exploitability(-0.2, "VULN-001") == 0.0
        assert _clamp_exploitability(0.7, "VULN-001") == 0.7

    @pytest.mark.asyncio
    async def test_priority_order_contains_all_vuln_ids(self, app_config, dry_run_llm_client):
        vulns = [_make_vulnerability(f"VULN-{i:03d}") for i in range(1, 4)]
        ids = [v.id for v in vulns]
        items = [_triage_item(vid) for vid in ids]
        mock_response = _llm_triage_response(items, priority_order=ids)

        agent = TriagerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            output, _ = await agent.run(
                TriagerInput(vulnerabilities=vulns, repo_metadata=_make_repo_metadata()),
                context=[],
            )

        assert set(output.priority_order) == set(ids)
        assert len(output.priority_order) == len(ids)

    @pytest.mark.asyncio
    async def test_priority_order_extra_ids_removed(self, app_config, dry_run_llm_client):
        vulns = [_make_vulnerability("VULN-001")]
        items = [_triage_item("VULN-001")]
        # LLM adds a spurious "VULN-999" that was never in input
        mock_response = _llm_triage_response(
            items, priority_order=["VULN-001", "VULN-999"]
        )

        agent = TriagerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            output, _ = await agent.run(
                TriagerInput(vulnerabilities=vulns, repo_metadata=_make_repo_metadata()),
                context=[],
            )

        assert output.priority_order == ["VULN-001"]

    @pytest.mark.asyncio
    async def test_priority_order_missing_id_appended(self, app_config, dry_run_llm_client):
        vulns = [_make_vulnerability("VULN-001"), _make_vulnerability("VULN-002")]
        items = [_triage_item("VULN-001"), _triage_item("VULN-002")]
        # LLM omits VULN-002 from priority_order
        mock_response = _llm_triage_response(items, priority_order=["VULN-001"])

        agent = TriagerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            output, _ = await agent.run(
                TriagerInput(vulnerabilities=vulns, repo_metadata=_make_repo_metadata()),
                context=[],
            )

        assert "VULN-001" in output.priority_order
        assert "VULN-002" in output.priority_order
        assert len(output.priority_order) == 2

    @pytest.mark.asyncio
    async def test_agent_message_fields(self, app_config, dry_run_llm_client):
        vulns = [_make_vulnerability("VULN-001")]
        items = [_triage_item("VULN-001")]
        mock_response = _llm_triage_response(items, priority_order=["VULN-001"])

        agent = TriagerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            _, message = await agent.run(
                TriagerInput(vulnerabilities=vulns, repo_metadata=_make_repo_metadata()),
                context=[],
            )

        assert message.agent_name == "triager"
        assert message.timestamp.tzinfo is not None
        assert message.token_count_input == 200
        assert message.token_count_output == 100
        assert message.cost_usd == pytest.approx(0.002)

    @pytest.mark.asyncio
    async def test_empty_vulnerabilities_list(self, app_config, dry_run_llm_client):
        mock_response = _llm_triage_response([], priority_order=[], summary="Nothing to triage.")

        agent = TriagerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=mock_response)):
            output, _ = await agent.run(
                TriagerInput(vulnerabilities=[], repo_metadata=_make_repo_metadata()),
                context=[],
            )

        assert output.triage_results == []
        assert output.priority_order == []

    @pytest.mark.asyncio
    async def test_llm_parse_error_assigns_all_defaults(self, app_config, dry_run_llm_client):
        vulns = [_make_vulnerability("VULN-001"), _make_vulnerability("VULN-002")]
        bad_response = LLMResponse(
            content="this is not valid json {{{",
            input_tokens=10,
            output_tokens=5,
            latency_ms=0.0,
            cost_usd=0.0,
            model="claude-sonnet-4-20250514",
        )

        agent = TriagerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(dry_run_llm_client, "complete", AsyncMock(return_value=bad_response)):
            output, _ = await agent.run(
                TriagerInput(vulnerabilities=vulns, repo_metadata=_make_repo_metadata()),
                context=[],
            )

        assert len(output.triage_results) == 2
        for result in output.triage_results:
            assert result.severity == _DEFAULT_SEVERITY
            assert result.fix_strategy == _DEFAULT_STRATEGY


# ---------------------------------------------------------------------------
# detect_framework tests
# ---------------------------------------------------------------------------

class TestDetectFramework:
    def test_python_django(self, tmp_path):
        (tmp_path / "app.py").write_text("import django\nfrom django.db import models\n")
        assert detect_framework(str(tmp_path), "python") == "django"

    def test_python_flask(self, tmp_path):
        (tmp_path / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\n")
        assert detect_framework(str(tmp_path), "python") == "flask"

    def test_python_fastapi(self, tmp_path):
        (tmp_path / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")
        assert detect_framework(str(tmp_path), "python") == "fastapi"

    def test_python_no_framework(self, tmp_path):
        (tmp_path / "utils.py").write_text("import os\nimport sys\n")
        assert detect_framework(str(tmp_path), "python") is None

    def test_javascript_express(self, tmp_path):
        pkg = {"dependencies": {"express": "^4.18.0", "lodash": "^4.17.21"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        assert detect_framework(str(tmp_path), "javascript") == "express"

    def test_javascript_next(self, tmp_path):
        pkg = {"dependencies": {"next": "^13.0.0", "react": "^18.0.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        # "next" comes before "react" in _JS_FRAMEWORKS
        result = detect_framework(str(tmp_path), "javascript")
        assert result in ("next", "react")

    def test_javascript_no_package_json(self, tmp_path):
        assert detect_framework(str(tmp_path), "javascript") is None

    def test_javascript_no_known_framework(self, tmp_path):
        pkg = {"dependencies": {"lodash": "^4.17.21"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        assert detect_framework(str(tmp_path), "javascript") is None

    def test_unknown_language_returns_none(self, tmp_path):
        assert detect_framework(str(tmp_path), "cobol") is None


# ---------------------------------------------------------------------------
# extract_repo_metadata tests
# ---------------------------------------------------------------------------

class TestExtractRepoMetadata:
    def test_has_tests_true_via_directory(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "app.py").write_text("x = 1\n")
        meta = extract_repo_metadata(str(tmp_path), "python")
        assert meta.has_tests is True

    def test_has_tests_true_via_test_dir(self, tmp_path):
        (tmp_path / "test").mkdir()
        (tmp_path / "app.py").write_text("x = 1\n")
        meta = extract_repo_metadata(str(tmp_path), "python")
        assert meta.has_tests is True

    def test_has_tests_true_via_test_file(self, tmp_path):
        (tmp_path / "test_utils.py").write_text("def test_foo(): pass\n")
        meta = extract_repo_metadata(str(tmp_path), "python")
        assert meta.has_tests is True

    def test_has_tests_false(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\n")
        meta = extract_repo_metadata(str(tmp_path), "python")
        assert meta.has_tests is False

    def test_dependency_files_found(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask==2.0\n")
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\n")
        (tmp_path / "app.py").write_text("x = 1\n")
        meta = extract_repo_metadata(str(tmp_path), "python")
        assert "requirements.txt" in meta.dependency_files
        assert "pyproject.toml" in meta.dependency_files

    def test_dependency_files_empty_when_none(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\n")
        meta = extract_repo_metadata(str(tmp_path), "python")
        assert meta.dependency_files == []

    def test_repo_name_from_directory(self, tmp_path):
        meta = extract_repo_metadata(str(tmp_path), "python")
        assert meta.repo_name == tmp_path.name

    def test_repo_name_from_git_config(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text(
            "[core]\n\trepositoryformatversion = 0\n"
            "[remote \"origin\"]\n\turl = https://github.com/example/my-repo.git\n"
        )
        meta = extract_repo_metadata(str(tmp_path), "python")
        assert meta.repo_name == "my-repo"

    def test_file_count(self, tmp_path):
        for i in range(3):
            (tmp_path / f"mod{i}.py").write_text("x = 1\n")
        meta = extract_repo_metadata(str(tmp_path), "python")
        assert meta.file_count == 3

    def test_language_stored_correctly(self, tmp_path):
        meta = extract_repo_metadata(str(tmp_path), "python")
        assert meta.language == "python"

    def test_framework_detected(self, tmp_path):
        (tmp_path / "app.py").write_text("import flask\n")
        meta = extract_repo_metadata(str(tmp_path), "python")
        assert meta.framework == "flask"


# ---------------------------------------------------------------------------
# Integration test: Scanner → Triager
# ---------------------------------------------------------------------------

class TestScannerToTriagerIntegration:
    @pytest.mark.asyncio
    async def test_scanner_output_fed_into_triager(self, app_config, dry_run_llm_client):
        """Feed scanner fixture vulns into triager; verify TriageResult count matches."""
        from multi_agent_security.agents.scanner import ScannerAgent, ScannerInput
        from multi_agent_security.llm_client import LLMResponse as LLMResp

        # Scanner: return one SQL injection finding
        scanner_vuln = {
            "file_path": "app.py",
            "line_start": 7,
            "line_end": 7,
            "vuln_type": "CWE-89",
            "description": "SQL injection via f-string",
            "confidence": 0.95,
            "code_snippet": "cursor.execute(f\"SELECT * FROM users WHERE username = '{username}'\")",
            "scanner_reasoning": "Unsanitised user input in SQL query",
        }
        scanner_mock = LLMResp(
            content=json.dumps({"vulnerabilities": [scanner_vuln]}),
            input_tokens=100,
            output_tokens=50,
            latency_ms=10.0,
            cost_usd=0.001,
            model="claude-sonnet-4-20250514",
        )

        scanner = ScannerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(
            dry_run_llm_client, "complete", AsyncMock(return_value=scanner_mock)
        ):
            scanner_output, _ = await scanner.run(
                ScannerInput(
                    repo_path=str(FIXTURES_DIR),
                    target_files=["app.py"],
                    language="python",
                    static_analysis_results=[],
                ),
                context=[],
            )

        assert len(scanner_output.vulnerabilities) == 1

        # Triager: return triage for the scanner's finding
        vuln_id = scanner_output.vulnerabilities[0].id
        triage_item = _triage_item(vuln_id, severity="critical", exploitability=0.9)
        triager_mock = _llm_triage_response([triage_item], priority_order=[vuln_id])

        triager = TriagerAgent(config=app_config, llm_client=dry_run_llm_client)
        with patch.object(
            dry_run_llm_client, "complete", AsyncMock(return_value=triager_mock)
        ):
            triager_output, _ = await triager.run(
                TriagerInput(
                    vulnerabilities=scanner_output.vulnerabilities,
                    repo_metadata=_make_repo_metadata(),
                ),
                context=[],
            )

        # Every scanner vulnerability must have a triage result
        assert len(triager_output.triage_results) == len(scanner_output.vulnerabilities)
        assert triager_output.triage_results[0].vuln_id == vuln_id
        assert triager_output.triage_results[0].severity == VulnSeverity.CRITICAL
        assert triager_output.priority_order == [vuln_id]
