"""Tests for HubSpokeOrchestrator (Architecture B)."""
import pathlib
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from multi_agent_security.agents.patcher import PatcherOutput
from multi_agent_security.agents.reviewer import ReviewerOutput
from multi_agent_security.agents.scanner import ScannerOutput
from multi_agent_security.agents.triager import TriagerOutput
from multi_agent_security.llm_client import LLMResponse
from multi_agent_security.memory.full_context import FullContextMemory
from multi_agent_security.orchestration.hub_spoke import (
    ContextPackage,
    HubSpokeOrchestrator,
    RoutingDecision,
)
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


def _make_msg(agent_name: str = "test", tokens_in: int = 10, tokens_out: int = 10) -> AgentMessage:
    return AgentMessage(
        agent_name=agent_name,
        timestamp=datetime.now(timezone.utc),
        content='{"result": "ok"}',
        token_count_input=tokens_in,
        token_count_output=tokens_out,
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
    agent.run.return_value = (PatcherOutput(patch=p), _make_msg("patcher"))
    return agent


def _make_reviewer_agent(accepted: bool = True, should_retry: bool = False) -> AsyncMock:
    agent = AsyncMock()
    agent.name = "reviewer"
    agent.run.return_value = (
        ReviewerOutput(review=_make_review(accepted=accepted), should_retry=should_retry),
        _make_msg("reviewer"),
    )
    return agent


def _make_config(orchestrator_type: str = "rule_based", max_revision_loops: int = 3):
    return __import__(
        "multi_agent_security.config", fromlist=["AppConfig"]
    ).AppConfig.model_validate({
        "agents": {
            "patcher": {"max_revision_loops": max_revision_loops, "run_tests": False},
            "scanner": {"use_static_analysis": False},
        },
        "orchestrator": {"type": orchestrator_type},
    })


def _build_orchestrator(
    config, scanner, triager, patcher, reviewer, routing_llm_client=None
) -> HubSpokeOrchestrator:
    memory = FullContextMemory()
    agents = {
        "scanner": scanner,
        "triager": triager,
        "patcher": patcher,
        "reviewer": reviewer,
    }
    return HubSpokeOrchestrator(config, agents, memory, routing_llm_client=routing_llm_client)


def _base_task_state() -> TaskState:
    return TaskState(task_id=str(uuid.uuid4()), repo_url="local", language="python")


def _make_routing_llm(decision: RoutingDecision | None = None) -> AsyncMock:
    """Return a mock LLM client that returns a canned RoutingDecision."""
    if decision is None:
        decision = RoutingDecision(
            context_summary="Relevant context summary",
            included_messages=[0],
            excluded_messages=[],
            routing_reasoning="Include only recent messages",
        )
    mock = AsyncMock()
    mock.complete.return_value = LLMResponse(
        content=decision.model_dump_json(),
        input_tokens=50,
        output_tokens=30,
        latency_ms=100.0,
        cost_usd=0.001,
        model="claude-sonnet-4-20250514",
    )
    return mock


# ---------------------------------------------------------------------------
# TestRuleBasedContext
# ---------------------------------------------------------------------------


class TestRuleBasedContext:
    def _orch(self) -> HubSpokeOrchestrator:
        config = _make_config("rule_based")
        return HubSpokeOrchestrator(config, {}, FullContextMemory(), routing_llm_client=None)

    def test_scanner_gets_empty_context(self):
        orch = self._orch()
        messages = [_make_msg("scanner"), _make_msg("triager")]
        result = orch._rule_based_context("scanner", messages)
        assert result == []

    def test_triager_gets_only_scanner_messages(self):
        orch = self._orch()
        scanner_msg = _make_msg("scanner")
        triager_msg = _make_msg("triager")
        messages = [scanner_msg, triager_msg]
        result = orch._rule_based_context("triager", messages)
        assert result == [scanner_msg]

    def test_patcher_gets_triager_and_patcher_messages(self):
        orch = self._orch()
        scanner_msg = _make_msg("scanner")
        triager_msg = _make_msg("triager")
        patcher_msg = _make_msg("patcher")
        messages = [scanner_msg, triager_msg, patcher_msg]
        result = orch._rule_based_context("patcher", messages)
        assert scanner_msg not in result
        assert triager_msg in result
        assert patcher_msg in result

    def test_reviewer_gets_triager_and_patcher_messages(self):
        orch = self._orch()
        scanner_msg = _make_msg("scanner")
        triager_msg = _make_msg("triager")
        patcher_msg = _make_msg("patcher")
        messages = [scanner_msg, triager_msg, patcher_msg]
        result = orch._rule_based_context("reviewer", messages)
        assert scanner_msg not in result
        assert triager_msg in result
        assert patcher_msg in result

    def test_unknown_agent_gets_empty_context(self):
        orch = self._orch()
        messages = [_make_msg("scanner"), _make_msg("triager")]
        result = orch._rule_based_context("unknown_agent", messages)
        assert result == []


# ---------------------------------------------------------------------------
# TestLLMBasedContext
# ---------------------------------------------------------------------------


class TestLLMBasedContext:
    @pytest.mark.asyncio
    async def test_llm_routing_call_made(self):
        config = _make_config("llm_based")
        mock_llm = _make_routing_llm()
        orch = HubSpokeOrchestrator(config, {}, FullContextMemory(), routing_llm_client=mock_llm)
        messages = [_make_msg("scanner")]
        await orch._llm_based_context("triager", messages)
        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args
        # Second positional arg is user_prompt; should mention the agent name
        assert "triager" in call_kwargs[0][0] or "triager" in call_kwargs[0][1]

    @pytest.mark.asyncio
    async def test_context_package_built_from_decision(self):
        config = _make_config("llm_based")
        messages = [_make_msg("scanner"), _make_msg("triager")]
        decision = RoutingDecision(
            context_summary="Only scanner output matters",
            included_messages=[0],
            excluded_messages=[1],
            routing_reasoning="Triager only needs scanner data",
        )
        mock_llm = _make_routing_llm(decision)
        orch = HubSpokeOrchestrator(config, {}, FullContextMemory(), routing_llm_client=mock_llm)
        result = await orch._llm_based_context("triager", messages)
        # First item is the orchestrator summary message
        assert result[0].agent_name == "orchestrator"
        assert result[0].content == "Only scanner output matters"
        # Second item is messages[0] (index 0 = scanner)
        assert result[1] == messages[0]
        # messages[1] (triager) excluded
        assert messages[1] not in result

    @pytest.mark.asyncio
    async def test_orchestrator_tokens_tracked(self):
        config = _make_config("llm_based")
        mock_llm = _make_routing_llm()
        orch = HubSpokeOrchestrator(config, {}, FullContextMemory(), routing_llm_client=mock_llm)
        messages = [_make_msg("scanner")]
        await orch._llm_based_context("triager", messages)
        assert orch._orch_tokens_in > 0
        assert orch._orch_tokens_out > 0
        assert orch._orch_llm_calls == 1

    @pytest.mark.asyncio
    async def test_empty_messages_returns_empty_without_llm_call(self):
        config = _make_config("llm_based")
        mock_llm = _make_routing_llm()
        orch = HubSpokeOrchestrator(config, {}, FullContextMemory(), routing_llm_client=mock_llm)
        result = await orch._llm_based_context("scanner", [])
        assert result == []
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_rule_based(self):
        config = _make_config("llm_based")
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = RuntimeError("LLM unavailable")
        orch = HubSpokeOrchestrator(config, {}, FullContextMemory(), routing_llm_client=mock_llm)
        scanner_msg = _make_msg("scanner")
        messages = [scanner_msg]
        result = await orch._llm_based_context("triager", messages)
        # Falls back to rule-based: triager gets scanner messages
        assert scanner_msg in result


# ---------------------------------------------------------------------------
# TestContextCompressionRatio
# ---------------------------------------------------------------------------


class TestContextCompressionRatio:
    @pytest.mark.asyncio
    async def test_compression_ratio_less_than_one(self, tmp_path):
        """With rule-based routing, agents get less than the full history."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config("rule_based")
        scanner = _make_scanner_agent([vuln])
        triager = _make_triager_agent([vuln])
        patcher = _make_patcher_agent()
        reviewer = _make_reviewer_agent(accepted=True)

        orch = _build_orchestrator(config, scanner, triager, patcher, reviewer)
        await orch.run(str(tmp_path), _base_task_state())

        metrics = orch.metrics
        # There are 4 agents; each gets a filtered subset so ratio should be < 1
        # (scanner gets 0, triager gets 1 msg, patcher gets 1, reviewer gets 2)
        assert 0.0 <= metrics["context_compression_ratio"] < 1.0

    def test_compression_ratio_zero_when_no_tokens_available(self):
        config = _make_config("rule_based")
        orch = HubSpokeOrchestrator(config, {}, FullContextMemory(), routing_llm_client=None)
        assert orch.metrics["context_compression_ratio"] == 0.0

    @pytest.mark.asyncio
    async def test_compression_ratio_calculated_correctly(self):
        """Manually check the ratio calculation."""
        config = _make_config("rule_based")
        orch = HubSpokeOrchestrator(config, {}, FullContextMemory(), routing_llm_client=None)

        # Simulate two calls: 100 tokens available, 30 tokens passed
        messages_100 = [_make_msg("scanner", tokens_in=50, tokens_out=50)]
        messages_30 = [_make_msg("triager", tokens_in=15, tokens_out=15)]

        # Manually call _get_context_for_agent to update stats
        await orch._get_context_for_agent("scanner", messages_100)  # scanner gets 0 → 0 passed
        await orch._get_context_for_agent("triager", messages_30)  # triager gets nothing (no scanner)

        metrics = orch.metrics
        assert metrics["context_compression_ratio"] == 0.0  # 0 passed / 130 available


# ---------------------------------------------------------------------------
# TestHubSpokePipelineIntegration
# ---------------------------------------------------------------------------


class TestHubSpokePipelineIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_rule_based(self, tmp_path):
        """Full run with mock agents in rule-based mode: status=complete, 4 messages."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config("rule_based")
        orch = _build_orchestrator(
            config,
            _make_scanner_agent([vuln]),
            _make_triager_agent([vuln]),
            _make_patcher_agent(),
            _make_reviewer_agent(accepted=True),
        )
        state = await orch.run(str(tmp_path), _base_task_state())

        assert state.status == "complete"
        assert len(state.messages) == 4
        assert [m.agent_name for m in state.messages] == ["scanner", "triager", "patcher", "reviewer"]
        assert orch.metrics["orchestrator_llm_calls"] == 0  # no LLM routing in rule-based

    @pytest.mark.asyncio
    async def test_full_pipeline_llm_based(self, tmp_path):
        """Full run with mock agents + mock routing LLM; orch_llm_calls >= 4."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config("llm_based")
        # Routing LLM always returns empty included_messages (minimal context)
        decision = RoutingDecision(
            context_summary="Minimal context",
            included_messages=[],
            excluded_messages=[],
            routing_reasoning="Keep it lean",
        )
        mock_llm = _make_routing_llm(decision)
        orch = _build_orchestrator(
            config,
            _make_scanner_agent([vuln]),
            _make_triager_agent([vuln]),
            _make_patcher_agent(),
            _make_reviewer_agent(accepted=True),
            routing_llm_client=mock_llm,
        )
        state = await orch.run(str(tmp_path), _base_task_state())

        assert state.status == "complete"
        # triager + patcher + reviewer = 3 routing calls
        # (scanner has no prior messages so the LLM call is skipped for it)
        assert orch.metrics["orchestrator_llm_calls"] >= 3

    @pytest.mark.asyncio
    async def test_empty_scanner_output(self, tmp_path):
        """No vulns found → triager/patcher/reviewer never called."""
        config = _make_config("rule_based")
        triager = _make_triager_agent([])
        patcher = _make_patcher_agent()
        reviewer = _make_reviewer_agent()
        orch = _build_orchestrator(
            config,
            _make_scanner_agent([]),
            triager,
            patcher,
            reviewer,
        )
        state = await orch.run(str(tmp_path), _base_task_state())

        assert state.status == "complete"
        triager.run.assert_not_called()
        patcher.run.assert_not_called()
        reviewer.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_loop(self, tmp_path):
        """Reviewer rejects first patch, accepts on second attempt."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config("rule_based", max_revision_loops=3)
        reviewer = AsyncMock()
        reviewer.name = "reviewer"
        reviewer.run.side_effect = [
            (ReviewerOutput(review=_make_review(accepted=False), should_retry=True), _make_msg("reviewer")),
            (ReviewerOutput(review=_make_review(accepted=True), should_retry=False), _make_msg("reviewer")),
        ]
        patcher = _make_patcher_agent()

        orch = _build_orchestrator(
            config,
            _make_scanner_agent([vuln]),
            _make_triager_agent([vuln]),
            patcher,
            reviewer,
        )
        state = await orch.run(str(tmp_path), _base_task_state())

        assert state.status == "complete"
        assert patcher.run.call_count == 2
        assert reviewer.run.call_count == 2
        assert state.revision_count == 1

    @pytest.mark.asyncio
    async def test_patcher_context_includes_previous_patcher_messages(self, tmp_path):
        """On retry, patcher's own previous message is included in rule-based context."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config("rule_based", max_revision_loops=2)
        reviewer = AsyncMock()
        reviewer.name = "reviewer"
        reviewer.run.side_effect = [
            (ReviewerOutput(review=_make_review(accepted=False), should_retry=True), _make_msg("reviewer")),
            (ReviewerOutput(review=_make_review(accepted=True), should_retry=False), _make_msg("reviewer")),
        ]
        patcher = _make_patcher_agent()

        orch = _build_orchestrator(
            config,
            _make_scanner_agent([vuln]),
            _make_triager_agent([vuln]),
            patcher,
            reviewer,
        )
        state = await orch.run(str(tmp_path), _base_task_state())

        # Second patcher call should have received context containing patcher's own first message
        second_call_context = patcher.run.call_args_list[1][0][1]
        context_agent_names = [m.agent_name for m in second_call_context]
        assert "patcher" in context_agent_names

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self, tmp_path):
        """After run(), metrics dict has all expected keys with valid values."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config("rule_based")
        orch = _build_orchestrator(
            config,
            _make_scanner_agent([vuln]),
            _make_triager_agent([vuln]),
            _make_patcher_agent(),
            _make_reviewer_agent(accepted=True),
        )
        await orch.run(str(tmp_path), _base_task_state())

        metrics = orch.metrics
        assert set(metrics.keys()) == {
            "orchestrator_llm_calls",
            "orchestrator_tokens_in",
            "orchestrator_tokens_out",
            "context_compression_ratio",
            "information_loss_events",
        }
        assert metrics["context_compression_ratio"] >= 0.0
        assert metrics["orchestrator_llm_calls"] == 0
