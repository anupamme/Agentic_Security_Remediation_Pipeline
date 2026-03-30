"""Unit and integration tests for SequentialOrchestrator and FullContextMemory."""
import pathlib
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multi_agent_security.agents.patcher import PatcherInput, PatcherOutput
from multi_agent_security.agents.reviewer import ReviewerInput, ReviewerOutput
from multi_agent_security.agents.scanner import ScannerInput, ScannerOutput
from multi_agent_security.agents.triager import RepoMetadata, TriagerInput, TriagerOutput
from multi_agent_security.memory.full_context import FullContextMemory
from multi_agent_security.orchestration.sequential import SequentialOrchestrator
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
# Helpers
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


def _make_review(vuln_id: str = "VULN-001", accepted: bool = True) -> ReviewResult:
    return ReviewResult(
        vuln_id=vuln_id,
        patch_accepted=accepted,
        correctness_score=0.9 if accepted else 0.4,
        security_score=0.95 if accepted else 0.5,
        style_score=0.8 if accepted else 0.6,
        review_reasoning="Looks good" if accepted else "Still vulnerable",
        revision_request=None if accepted else "Use parameterised queries",
    )


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


def _make_reviewer_agent(accepted: bool = True, should_retry: bool = False) -> AsyncMock:
    agent = AsyncMock()
    agent.name = "reviewer"
    agent.run.return_value = (
        ReviewerOutput(review=_make_review(accepted=accepted), should_retry=should_retry),
        _make_msg("reviewer"),
    )
    return agent


def _make_config(max_revision_loops: int = 3, run_tests: bool = False):
    from multi_agent_security.config import AppConfig, PatcherAgentConfig, AgentsConfig
    agents = AgentsConfig(
        patcher=PatcherAgentConfig(max_revision_loops=max_revision_loops, run_tests=run_tests)
    )
    # AppConfig is frozen; build via model_validate with a dict
    return AppConfig.model_validate({
        "agents": {
            "patcher": {"max_revision_loops": max_revision_loops, "run_tests": run_tests},
            "scanner": {"use_static_analysis": False},
        }
    })


def _build_orchestrator(config, scanner, triager, patcher, reviewer):
    memory = FullContextMemory()
    agents = {
        "scanner": scanner,
        "triager": triager,
        "patcher": patcher,
        "reviewer": reviewer,
    }
    return SequentialOrchestrator(config, agents, memory)


def _base_task_state(repo_url: str = "local") -> TaskState:
    return TaskState(
        task_id=str(uuid.uuid4()),
        repo_url=repo_url,
        language="python",
    )


# ---------------------------------------------------------------------------
# FullContextMemory tests
# ---------------------------------------------------------------------------

class TestFullContextMemory:
    def test_store_and_retrieve_returns_all(self):
        mem = FullContextMemory()
        msg1 = _make_msg("scanner")
        msg2 = _make_msg("triager")
        mem.store(msg1)
        mem.store(msg2)
        assert mem.retrieve("scanner") == [msg1, msg2]
        assert mem.retrieve("patcher") == [msg1, msg2]

    def test_clear_empties_memory(self):
        mem = FullContextMemory()
        mem.store(_make_msg())
        mem.clear()
        assert mem.retrieve("scanner") == []

    def test_retrieve_returns_copy(self):
        mem = FullContextMemory()
        mem.store(_make_msg())
        result = mem.retrieve("x")
        result.clear()
        assert len(mem.retrieve("x")) == 1  # original unchanged


# ---------------------------------------------------------------------------
# SequentialOrchestrator unit tests (mocked agents)
# ---------------------------------------------------------------------------

class TestSequentialOrchestrator:
    @pytest.mark.asyncio
    async def test_execution_order(self, tmp_path):
        """Scanner → Triager → Patcher → Reviewer called in order."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config()
        scanner = _make_scanner_agent([vuln])
        triager = _make_triager_agent([vuln])
        patcher = _make_patcher_agent()
        reviewer = _make_reviewer_agent(accepted=True)

        call_order = []
        scanner.run.side_effect = lambda *a, **kw: (
            call_order.append("scanner"),
            _make_scanner_agent([vuln]).run.return_value,
        )[1]
        triager.run.side_effect = lambda *a, **kw: (
            call_order.append("triager"),
            _make_triager_agent([vuln]).run.return_value,
        )[1]
        patcher.run.side_effect = lambda *a, **kw: (
            call_order.append("patcher"),
            _make_patcher_agent().run.return_value,
        )[1]
        reviewer.run.side_effect = lambda *a, **kw: (
            call_order.append("reviewer"),
            _make_reviewer_agent(accepted=True).run.return_value,
        )[1]

        orch = _build_orchestrator(config, scanner, triager, patcher, reviewer)
        state = await orch.run(str(tmp_path), _base_task_state())

        assert call_order == ["scanner", "triager", "patcher", "reviewer"]
        assert state.status == "complete"
        assert len(state.patches) == 1
        assert len(state.reviews) == 1

    @pytest.mark.asyncio
    async def test_retry_loop(self, tmp_path):
        """Reviewer rejects → patcher called again with feedback → accepted on 2nd try."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config(max_revision_loops=3)
        scanner = _make_scanner_agent([vuln])
        triager = _make_triager_agent([vuln])
        patcher = _make_patcher_agent()
        # Reject first, accept second
        reviewer = AsyncMock()
        reviewer.name = "reviewer"
        reviewer.run.side_effect = [
            (ReviewerOutput(review=_make_review(accepted=False), should_retry=True), _make_msg("reviewer")),
            (ReviewerOutput(review=_make_review(accepted=True), should_retry=False), _make_msg("reviewer")),
        ]

        orch = _build_orchestrator(config, scanner, triager, patcher, reviewer)
        state = await orch.run(str(tmp_path), _base_task_state())

        assert patcher.run.call_count == 2
        assert reviewer.run.call_count == 2
        assert state.revision_count == 1
        assert state.status == "complete"

        # Second patcher call should have revision_feedback set
        second_patcher_call_input = patcher.run.call_args_list[1][0][0]
        assert second_patcher_call_input.revision_feedback is not None

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, tmp_path):
        """Reviewer always rejects — pipeline moves on after max_revision_loops."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        max_loops = 3
        config = _make_config(max_revision_loops=max_loops)
        scanner = _make_scanner_agent([vuln])
        triager = _make_triager_agent([vuln])
        patcher = _make_patcher_agent()
        reviewer = _make_reviewer_agent(accepted=False, should_retry=True)

        orch = _build_orchestrator(config, scanner, triager, patcher, reviewer)
        state = await orch.run(str(tmp_path), _base_task_state())

        assert patcher.run.call_count == max_loops
        assert state.status == "complete"

    @pytest.mark.asyncio
    async def test_empty_scanner_output(self, tmp_path):
        """No vulns found → triager/patcher/reviewer never called, status=complete."""
        config = _make_config()
        scanner = _make_scanner_agent([])  # no vulns
        triager = _make_triager_agent([])
        patcher = _make_patcher_agent()
        reviewer = _make_reviewer_agent()

        orch = _build_orchestrator(config, scanner, triager, patcher, reviewer)
        state = await orch.run(str(tmp_path), _base_task_state())

        assert state.status == "complete"
        triager.run.assert_not_called()
        patcher.run.assert_not_called()
        reviewer.run.assert_not_called()
        assert state.vulnerabilities == []

    @pytest.mark.asyncio
    async def test_patcher_error_continues(self, tmp_path):
        """Patcher raises — vulnerability marked failed, pipeline continues to next vuln."""
        vuln1 = _make_vuln("VULN-001")
        vuln2 = _make_vuln("VULN-002")
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config()
        scanner = _make_scanner_agent([vuln1, vuln2])
        triager = _make_triager_agent([vuln1, vuln2])

        patcher = AsyncMock()
        patcher.name = "patcher"
        patcher.run.side_effect = [
            RuntimeError("LLM error on vuln1"),
            (_make_patcher_agent(_make_patch("VULN-002")).run.return_value),
        ]

        reviewer = _make_reviewer_agent(accepted=True)

        orch = _build_orchestrator(config, scanner, triager, patcher, reviewer)
        state = await orch.run(str(tmp_path), _base_task_state())

        assert state.status == "complete"
        assert "VULN-001" in state.failed_vulns
        # Reviewer called once for VULN-002 (which succeeded)
        assert reviewer.run.call_count == 1

    @pytest.mark.asyncio
    async def test_messages_tracked_in_task_state(self, tmp_path):
        """All AgentMessages are stored in task_state.messages."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config()
        scanner = _make_scanner_agent([vuln])
        triager = _make_triager_agent([vuln])
        patcher = _make_patcher_agent()
        reviewer = _make_reviewer_agent(accepted=True)

        orch = _build_orchestrator(config, scanner, triager, patcher, reviewer)
        state = await orch.run(str(tmp_path), _base_task_state())

        # scanner + triager + patcher + reviewer = 4 messages
        assert len(state.messages) == 4
        agent_names = [m.agent_name for m in state.messages]
        assert agent_names == ["scanner", "triager", "patcher", "reviewer"]


# ---------------------------------------------------------------------------
# Config inheritance tests
# ---------------------------------------------------------------------------

class TestConfigInheritance:
    def test_arch_sequential_inherits_default(self):
        from multi_agent_security.config import load_config
        import pathlib
        config_path = pathlib.Path(__file__).parent.parent / "config" / "arch_sequential.yaml"
        config = load_config(str(config_path))
        assert config.architecture == "sequential"
        assert config.memory.strategy == "full_context"
        # Inherited from default.yaml
        assert config.llm.model == "claude-sonnet-4-20250514"
        assert config.agents.patcher.max_revision_loops == 3

    def test_deep_merge_override(self, tmp_path):
        from multi_agent_security.config import load_config
        base = tmp_path / "base.yaml"
        base.write_text("llm:\n  model: base-model\n  max_tokens: 1000\n")
        override = tmp_path / "override.yaml"
        override.write_text(f"base: base.yaml\nllm:\n  model: override-model\n")
        config = load_config(str(override))
        assert config.llm.model == "override-model"
        assert config.llm.max_tokens == 1000  # preserved from base


# ---------------------------------------------------------------------------
# Integration test (requires real LLM — skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_real_llm():
    """End-to-end pipeline on tests/fixtures/vulnerable_repo/ with a real LLM."""
    import pathlib
    from multi_agent_security.config import load_config
    from multi_agent_security.llm_client import LLMClient
    from multi_agent_security.agents.scanner import ScannerAgent
    from multi_agent_security.agents.triager import TriagerAgent
    from multi_agent_security.agents.patcher import PatcherAgent
    from multi_agent_security.agents.reviewer import ReviewerAgent

    config_path = pathlib.Path(__file__).parent.parent / "config" / "arch_sequential.yaml"
    config = load_config(str(config_path))
    llm_client = LLMClient(config.llm)

    agents = {
        "scanner": ScannerAgent(config, llm_client),
        "triager": TriagerAgent(config, llm_client),
        "patcher": PatcherAgent(config, llm_client),
        "reviewer": ReviewerAgent(config, llm_client),
    }
    memory = FullContextMemory()
    orch = SequentialOrchestrator(config, agents, memory)

    repo_path = str(pathlib.Path(__file__).parent / "fixtures" / "vulnerable_repo")
    state = TaskState(
        task_id="integration-test-001",
        repo_url="local",
        language="python",
    )
    result = await orch.run(repo_path, state)

    assert result.status == "complete"
    assert len(result.vulnerabilities) > 0
    assert len(result.messages) > 0
    # Every message should have cost tracked
    for msg in result.messages:
        assert msg.cost_usd >= 0
        assert msg.token_count_input > 0
