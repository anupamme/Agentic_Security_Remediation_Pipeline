"""Unit tests for ablation study components.

Covers:
  - SingleAgent: mock LLM → all 4 outputs parsed into SingleAgentOutput
  - AblationOrchestrator skip_agents=["reviewer"]: auto-accept all patches
  - AblationOrchestrator skip_agents=["triager"]: default triage values used
  - AblationOrchestrator skip_agents=["triager","patcher","reviewer"]: scan only
"""
import pathlib
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multi_agent_security.agents.patcher import PatcherOutput
from multi_agent_security.agents.reviewer import ReviewerOutput
from multi_agent_security.agents.scanner import ScannerInput, ScannerOutput
from multi_agent_security.agents.single_agent import (
    SingleAgent,
    SingleAgentInput,
    _SingleAgentLLMResponse,
    _RawVuln,
    _RawTriage,
    _RawPatch,
    _RawReview,
)
from multi_agent_security.agents.triager import TriagerInput, TriagerOutput
from multi_agent_security.llm_client import LLMResponse
from multi_agent_security.memory.full_context import FullContextMemory
from multi_agent_security.orchestration.ablation import AblationOrchestrator, SingleAgentOrchestrator
from multi_agent_security.types import (
    AgentMessage,
    FixStrategy,
    Patch,
    ReviewResult,
    TaskState,
    TriageResult,
    Vulnerability,
    VulnSeverity,
    VulnType,
)

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "vulnerable_repo"


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_sequential.py)
# ---------------------------------------------------------------------------

def _make_msg(agent_name: str = "test") -> AgentMessage:
    return AgentMessage(
        agent_name=agent_name,
        timestamp=datetime.now(timezone.utc),
        content="{}",
        token_count_input=10,
        token_count_output=10,
        latency_ms=50.0,
        cost_usd=0.0001,
    )


def _make_vuln(vuln_id: str = "VULN-001", file_path: str = "app.py") -> Vulnerability:
    return Vulnerability(
        id=vuln_id,
        file_path=file_path,
        line_start=7,
        line_end=7,
        vuln_type=VulnType.SQL_INJECTION,
        description="SQL injection via f-string",
        confidence=0.95,
        code_snippet='cursor.execute(f"SELECT * FROM users WHERE username = \'{username}\'")',
        scanner_reasoning="f-string interpolation in SQL context",
    )


def _make_triage(vuln_id: str = "VULN-001") -> TriageResult:
    return TriageResult(
        vuln_id=vuln_id,
        severity=VulnSeverity.HIGH,
        exploitability_score=0.9,
        fix_strategy=FixStrategy.ONE_LINER,
        estimated_complexity="low",
        triage_reasoning="Easy to exploit remotely",
    )


def _make_patch(vuln_id: str = "VULN-001") -> Patch:
    original = FIXTURES_DIR.joinpath("app.py").read_text()
    patched = original.replace(
        'cursor.execute(f"SELECT * FROM users WHERE username = \'{username}\'")',
        'cursor.execute("SELECT * FROM users WHERE username = ?", (username,))',
    )
    return Patch(
        vuln_id=vuln_id,
        file_path="app.py",
        original_code=original,
        patched_code=patched,
        unified_diff="--- a/app.py\n+++ b/app.py\n",
        patch_reasoning="Use parameterised query",
    )


def _make_config(max_revision_loops: int = 3, run_tests: bool = False):
    from multi_agent_security.config import AppConfig
    return AppConfig.model_validate({
        "agents": {
            "patcher": {"max_revision_loops": max_revision_loops, "run_tests": run_tests},
            "scanner": {"use_static_analysis": False},
        }
    })


def _make_scanner_agent(vulns: list[Vulnerability]) -> AsyncMock:
    agent = AsyncMock()
    agent.name = "scanner"
    agent.run.return_value = (
        ScannerOutput(
            vulnerabilities=vulns,
            files_scanned=1,
            scan_summary=f"{len(vulns)} vuln(s) found",
        ),
        _make_msg("scanner"),
    )
    return agent


def _make_triager_agent(vulns: list[Vulnerability]) -> AsyncMock:
    triage_results = [_make_triage(v.id) for v in vulns]
    agent = AsyncMock()
    agent.name = "triager"
    agent.run.return_value = (
        TriagerOutput(
            triage_results=triage_results,
            priority_order=[v.id for v in vulns],
            summary="Triaged",
        ),
        _make_msg("triager"),
    )
    return agent


def _make_patcher_agent(patch: Patch | None = None) -> AsyncMock:
    agent = AsyncMock()
    agent.name = "patcher"
    p = patch or _make_patch()
    agent.run.return_value = (
        PatcherOutput(patch=p),
        _make_msg("patcher"),
    )
    return agent


def _make_reviewer_agent(accepted: bool = True) -> AsyncMock:
    from multi_agent_security.agents.reviewer import ReviewerOutput
    agent = AsyncMock()
    agent.name = "reviewer"
    agent.run.return_value = (
        ReviewerOutput(
            review=ReviewResult(
                vuln_id="VULN-001",
                patch_accepted=accepted,
                correctness_score=0.9 if accepted else 0.4,
                security_score=0.95 if accepted else 0.5,
                style_score=0.8 if accepted else 0.6,
                review_reasoning="Looks good" if accepted else "Still vulnerable",
                revision_request=None if accepted else "Use parameterised queries",
            ),
            should_retry=not accepted,
        ),
        _make_msg("reviewer"),
    )
    return agent


def _base_task_state() -> TaskState:
    return TaskState(
        task_id=str(uuid.uuid4()),
        repo_url="local",
        language="python",
    )


# ---------------------------------------------------------------------------
# Test 1: SingleAgent produces all 4 outputs from one LLM call
# ---------------------------------------------------------------------------

class TestSingleAgent:
    @pytest.mark.asyncio
    async def test_single_agent_all_outputs(self, tmp_path):
        """Mock LLM returns all 4 outputs → correctly parsed into SingleAgentOutput."""
        # Create a file for the agent to load
        app_py = tmp_path / "app.py"
        app_py.write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        # Build the expected LLM response
        llm_response_data = _SingleAgentLLMResponse(
            vulnerabilities=[
                _RawVuln(
                    file_path="app.py",
                    line_start=7,
                    line_end=7,
                    vuln_type=VulnType.SQL_INJECTION,
                    description="SQL injection",
                    confidence=0.95,
                    code_snippet="cursor.execute(f\"...\")",
                    scanner_reasoning="f-string in SQL",
                ),
            ],
            triage_results=[
                _RawTriage(
                    vuln_index=0,
                    severity=VulnSeverity.HIGH,
                    exploitability_score=0.9,
                    fix_strategy=FixStrategy.ONE_LINER,
                    estimated_complexity="low",
                    triage_reasoning="Easy to exploit",
                ),
            ],
            patches=[
                _RawPatch(
                    vuln_index=0,
                    file_path="app.py",
                    original_code="cursor.execute(f\"...\")",
                    patched_code="cursor.execute(\"...\", (val,))",
                    unified_diff="--- a/app.py\n+++ b/app.py\n",
                    patch_reasoning="Use parameterised query",
                ),
            ],
            reviews=[
                _RawReview(
                    vuln_index=0,
                    patch_accepted=True,
                    correctness_score=0.9,
                    security_score=0.95,
                    style_score=0.8,
                    review_reasoning="Looks good",
                    revision_request=None,
                ),
            ],
            summary="1 SQL injection found and patched.",
        )

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=LLMResponse(
            content=llm_response_data.model_dump_json(),
            input_tokens=500,
            output_tokens=300,
            latency_ms=1200.0,
            cost_usd=0.005,
            model="claude-sonnet-4-20250514",
        ))

        config = _make_config()
        agent = SingleAgent(config, mock_llm)

        inp = SingleAgentInput(
            repo_path=str(tmp_path),
            target_files=["app.py"],
            language="python",
        )
        output, msg = await agent.run(inp, context=[])

        # All 4 outputs populated
        assert len(output.vulnerabilities) == 1
        assert len(output.triage_results) == 1
        assert len(output.patches) == 1
        assert len(output.reviews) == 1

        # IDs assigned correctly
        assert output.vulnerabilities[0].id == "VULN-001"
        assert output.triage_results[0].vuln_id == "VULN-001"
        assert output.patches[0].vuln_id == "VULN-001"
        assert output.reviews[0].vuln_id == "VULN-001"

        # AgentMessage has cost
        assert msg.cost_usd > 0
        assert msg.token_count_input > 0
        assert msg.agent_name == "single_agent"

    @pytest.mark.asyncio
    async def test_single_agent_no_vulns(self, tmp_path):
        """LLM returns empty vulnerability list → output has empty lists."""
        (tmp_path / "app.py").write_text("def safe(): pass\n")

        empty_response = _SingleAgentLLMResponse(
            vulnerabilities=[],
            triage_results=[],
            patches=[],
            reviews=[],
            summary="No vulnerabilities found.",
        )

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=LLMResponse(
            content=empty_response.model_dump_json(),
            input_tokens=100,
            output_tokens=50,
            latency_ms=400.0,
            cost_usd=0.001,
            model="claude-sonnet-4-20250514",
        ))

        config = _make_config()
        agent = SingleAgent(config, mock_llm)
        output, msg = await agent.run(
            SingleAgentInput(repo_path=str(tmp_path), target_files=["app.py"], language="python"),
            context=[],
        )

        assert output.vulnerabilities == []
        assert output.triage_results == []
        assert output.patches == []
        assert output.reviews == []


# ---------------------------------------------------------------------------
# Test 2: AblationOrchestrator — skip reviewer → auto-accept patches
# ---------------------------------------------------------------------------

class TestAblationSkipReviewer:
    @pytest.mark.asyncio
    async def test_skip_reviewer_auto_accepts(self, tmp_path):
        """Reviewer never called; patch auto-accepted with patch_accepted=True."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config()
        scanner = _make_scanner_agent([vuln])
        triager = _make_triager_agent([vuln])
        patcher = _make_patcher_agent()
        reviewer = _make_reviewer_agent(accepted=True)  # should not be called

        memory = FullContextMemory()
        agents = {
            "scanner": scanner,
            "triager": triager,
            "patcher": patcher,
            "reviewer": reviewer,
        }
        orch = AblationOrchestrator(config, agents, memory, skip_agents=["reviewer"])
        state = await orch.run(str(tmp_path), _base_task_state())

        # Reviewer never called
        reviewer.run.assert_not_called()

        # Patch auto-accepted
        assert len(state.reviews) == 1
        assert state.reviews[0].patch_accepted is True
        assert "skipped" in state.reviews[0].review_reasoning.lower()
        assert state.status == "complete"

    @pytest.mark.asyncio
    async def test_skip_reviewer_patcher_called_once(self, tmp_path):
        """With no reviewer, patcher is only called once (no retry loop)."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config(max_revision_loops=5)
        scanner = _make_scanner_agent([vuln])
        triager = _make_triager_agent([vuln])
        patcher = _make_patcher_agent()
        reviewer = AsyncMock()
        reviewer.name = "reviewer"

        memory = FullContextMemory()
        agents = {"scanner": scanner, "triager": triager, "patcher": patcher, "reviewer": reviewer}
        orch = AblationOrchestrator(config, agents, memory, skip_agents=["reviewer"])
        await orch.run(str(tmp_path), _base_task_state())

        # Patcher called exactly once (no retry since reviewer is skipped)
        assert patcher.run.call_count == 1
        reviewer.run.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: AblationOrchestrator — skip triager → default triage used
# ---------------------------------------------------------------------------

class TestAblationSkipTriager:
    @pytest.mark.asyncio
    async def test_skip_triager_default_values(self, tmp_path):
        """Triager never called; default TriageResult assigned for each vuln."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config()
        scanner = _make_scanner_agent([vuln])
        triager = _make_triager_agent([vuln])  # should not be called
        patcher = _make_patcher_agent()
        reviewer = _make_reviewer_agent(accepted=True)

        memory = FullContextMemory()
        agents = {"scanner": scanner, "triager": triager, "patcher": patcher, "reviewer": reviewer}
        orch = AblationOrchestrator(config, agents, memory, skip_agents=["triager"])
        state = await orch.run(str(tmp_path), _base_task_state())

        # Triager never called
        triager.run.assert_not_called()

        # Default triage values assigned
        assert len(state.triage_results) == 1
        tr = state.triage_results[0]
        assert tr.vuln_id == vuln.id
        assert tr.severity == VulnSeverity.MEDIUM
        assert tr.exploitability_score == 0.5
        assert tr.fix_strategy == FixStrategy.REFACTOR
        assert "skipped" in tr.triage_reasoning.lower()

        # Pipeline still completes normally
        assert len(state.patches) == 1
        assert state.status == "complete"

    @pytest.mark.asyncio
    async def test_skip_triager_priority_order_by_confidence(self, tmp_path):
        """With triager skipped, vulns are processed in scanner confidence order."""
        vuln_hi = Vulnerability(
            id="VULN-001", file_path="app.py", line_start=1, line_end=1,
            vuln_type=VulnType.SQL_INJECTION, description="hi", confidence=0.95,
            code_snippet="...", scanner_reasoning="...",
        )
        vuln_lo = Vulnerability(
            id="VULN-002", file_path="app.py", line_start=5, line_end=5,
            vuln_type=VulnType.XSS, description="lo", confidence=0.6,
            code_snippet="...", scanner_reasoning="...",
        )
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config()
        scanner = _make_scanner_agent([vuln_hi, vuln_lo])
        triager = _make_triager_agent([vuln_hi, vuln_lo])
        patcher = _make_patcher_agent()
        reviewer = _make_reviewer_agent(accepted=True)

        memory = FullContextMemory()
        agents = {"scanner": scanner, "triager": triager, "patcher": patcher, "reviewer": reviewer}
        orch = AblationOrchestrator(config, agents, memory, skip_agents=["triager"])
        state = await orch.run(str(tmp_path), _base_task_state())

        triager.run.assert_not_called()
        # Both vulns should have triage with default values
        assert len(state.triage_results) == 2


# ---------------------------------------------------------------------------
# Test 4: AblationOrchestrator — scan only (skip triager + patcher + reviewer)
# ---------------------------------------------------------------------------

class TestAblationScanOnly:
    @pytest.mark.asyncio
    async def test_scan_only_no_patches(self, tmp_path):
        """A4: Only scanner runs; no patches or reviews generated."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config()
        scanner = _make_scanner_agent([vuln])
        triager = _make_triager_agent([vuln])
        patcher = _make_patcher_agent()
        reviewer = _make_reviewer_agent()

        memory = FullContextMemory()
        agents = {"scanner": scanner, "triager": triager, "patcher": patcher, "reviewer": reviewer}
        orch = AblationOrchestrator(
            config, agents, memory,
            skip_agents=["triager", "patcher", "reviewer"],
        )
        state = await orch.run(str(tmp_path), _base_task_state())

        # Only scanner runs
        scanner.run.assert_called_once()
        triager.run.assert_not_called()
        patcher.run.assert_not_called()
        reviewer.run.assert_not_called()

        # Vulnerabilities detected but no patches or reviews
        assert len(state.vulnerabilities) == 1
        assert state.patches == []
        assert state.reviews == []
        assert state.status == "complete"

    @pytest.mark.asyncio
    async def test_scan_only_empty_scanner(self, tmp_path):
        """A4 with no vulns found → still completes with empty state."""
        config = _make_config()
        scanner = _make_scanner_agent([])
        triager = _make_triager_agent([])
        patcher = _make_patcher_agent()
        reviewer = _make_reviewer_agent()

        memory = FullContextMemory()
        agents = {"scanner": scanner, "triager": triager, "patcher": patcher, "reviewer": reviewer}
        orch = AblationOrchestrator(
            config, agents, memory,
            skip_agents=["triager", "patcher", "reviewer"],
        )
        state = await orch.run(str(tmp_path), _base_task_state())

        assert state.vulnerabilities == []
        assert state.patches == []
        assert state.status == "complete"


# ---------------------------------------------------------------------------
# Test 5: SingleAgentOrchestrator
# ---------------------------------------------------------------------------

class TestSingleAgentOrchestrator:
    @pytest.mark.asyncio
    async def test_orchestrator_maps_outputs_to_task_state(self, tmp_path):
        """SingleAgentOrchestrator maps SingleAgentOutput to TaskState fields."""
        from multi_agent_security.agents.single_agent import SingleAgentOutput

        vuln = _make_vuln()
        patch = _make_patch()
        triage = _make_triage()
        review = ReviewResult(
            vuln_id="VULN-001",
            patch_accepted=True,
            correctness_score=0.9,
            security_score=0.95,
            style_score=0.8,
            review_reasoning="Auto",
        )

        single_agent = AsyncMock()
        single_agent.name = "single_agent"
        single_agent.run.return_value = (
            SingleAgentOutput(
                vulnerabilities=[vuln],
                triage_results=[triage],
                patches=[patch],
                reviews=[review],
                summary="1 vuln found and patched.",
            ),
            _make_msg("single_agent"),
        )

        config = _make_config()
        memory = FullContextMemory()
        agents = {"single_agent": single_agent}
        orch = SingleAgentOrchestrator(config, agents, memory)

        state = await orch.run(str(tmp_path), _base_task_state())

        assert len(state.vulnerabilities) == 1
        assert len(state.triage_results) == 1
        assert len(state.patches) == 1
        assert len(state.reviews) == 1
        assert len(state.messages) == 1
        assert state.messages[0].agent_name == "single_agent"
        assert state.status == "complete"
