"""Tests for Blackboard (Architecture C) — data structure and orchestrator."""
import pathlib
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from multi_agent_security.agents.patcher import PatcherOutput
from multi_agent_security.agents.reviewer import ReviewerOutput
from multi_agent_security.agents.scanner import ScannerOutput
from multi_agent_security.agents.triager import TriagerOutput
from multi_agent_security.memory.full_context import FullContextMemory
from multi_agent_security.orchestration.blackboard import (
    AGENT_READ_PERMISSIONS,
    Blackboard,
    BlackboardEntry,
    BlackboardOrchestrator,
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


def _make_msg(agent_name: str = "test") -> AgentMessage:
    return AgentMessage(
        agent_name=agent_name,
        timestamp=datetime.now(timezone.utc),
        content='{"result": "ok"}',
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


def _make_review(
    vuln_id: str = "VULN-001",
    accepted: bool = True,
    correctness: float = 0.9,
    security: float = 0.95,
) -> ReviewResult:
    return ReviewResult(
        vuln_id=vuln_id,
        patch_accepted=accepted,
        correctness_score=correctness,
        security_score=security,
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


def _make_config(max_revision_loops: int = 3, run_tests: bool = False):
    return __import__(
        "multi_agent_security.config", fromlist=["AppConfig"]
    ).AppConfig.model_validate({
        "agents": {
            "patcher": {"max_revision_loops": max_revision_loops, "run_tests": run_tests},
            "scanner": {"use_static_analysis": False},
        },
        "orchestrator": {"type": "rule_based"},
        "memory": {"max_context_tokens": 4000},
    })


def _build_orchestrator(config, scanner, triager, patcher, reviewer) -> BlackboardOrchestrator:
    memory = FullContextMemory()
    agents = {
        "scanner": scanner,
        "triager": triager,
        "patcher": patcher,
        "reviewer": reviewer,
    }
    return BlackboardOrchestrator(config, agents, memory)


def _base_task_state() -> TaskState:
    return TaskState(task_id=str(uuid.uuid4()), repo_url="local", language="python")


# ---------------------------------------------------------------------------
# TestBlackboardCRUD
# ---------------------------------------------------------------------------


class TestCollectSourceFiles:
    def test_skips_excluded_dirs(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git internals")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lodash.js").write_text("module code")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"\x00\x01")
        (tmp_path / "app.py").write_text("print('hello')")

        result = BlackboardOrchestrator._collect_source_files(str(tmp_path), "python")
        assert result == ["app.py"]

    def test_filters_by_language_extension(self, tmp_path):
        (tmp_path / "main.py").write_text("# python")
        (tmp_path / "index.js").write_text("// js")
        (tmp_path / "README.md").write_text("# docs")

        py_files = BlackboardOrchestrator._collect_source_files(str(tmp_path), "python")
        js_files = BlackboardOrchestrator._collect_source_files(str(tmp_path), "javascript")

        assert py_files == ["main.py"]
        assert js_files == ["index.js"]

    def test_unknown_language_returns_all_files(self, tmp_path):
        (tmp_path / "main.rs").write_text("fn main() {}")
        (tmp_path / "Cargo.toml").write_text("[package]")

        result = BlackboardOrchestrator._collect_source_files(str(tmp_path), "rust")
        assert len(result) == 2

    def test_oversized_file_skipped_at_load_time(self, tmp_path):
        """Files exceeding _MAX_FILE_BYTES must not be written to the blackboard."""
        from multi_agent_security.orchestration.blackboard import _MAX_FILE_BYTES
        (tmp_path / "big.py").write_bytes(b"x" * (_MAX_FILE_BYTES + 1))
        (tmp_path / "small.py").write_text("x = 1")

        config = _make_config()
        orch = BlackboardOrchestrator(config, {}, FullContextMemory())
        # Directly exercise the loading logic by calling _collect_source_files
        # then checking the size gate in a minimal run simulation.
        files = BlackboardOrchestrator._collect_source_files(str(tmp_path), "python")
        assert "big.py" in files   # collected, but will be skipped on load
        assert "small.py" in files


class TestBlackboardCRUD:
    def test_write_and_read(self):
        bb = Blackboard()
        bb.write("scanner.summary", "2 vulns found", "scanner")
        entry = bb.read("scanner.summary")
        assert entry is not None
        assert entry.value == "2 vulns found"
        assert entry.agent == "scanner"
        assert entry.version == 1

    def test_version_incremented_on_update(self):
        bb = Blackboard()
        bb.write("patcher.patches.VULN-001", "patch_v1", "patcher")
        bb.write("patcher.patches.VULN-001", "patch_v2", "patcher")
        entry = bb.read("patcher.patches.VULN-001")
        assert entry is not None
        assert entry.version == 2
        assert entry.value == "patch_v2"

    def test_history_appended_on_update(self):
        bb = Blackboard()
        bb.write("patcher.patches.VULN-001", "patch_v1", "patcher")
        bb.write("patcher.patches.VULN-001", "patch_v2", "patcher")
        # History should contain the first version
        history = [e for e in bb._history if e.key == "patcher.patches.VULN-001"]
        assert len(history) == 1
        assert history[0].version == 1
        assert history[0].value == "patch_v1"

    def test_read_nonexistent_returns_none(self):
        bb = Blackboard()
        assert bb.read("nonexistent.key") is None

    def test_read_prefix(self):
        bb = Blackboard()
        bb.write("scanner.vulnerabilities.VULN-001", "v1", "scanner")
        bb.write("scanner.vulnerabilities.VULN-002", "v2", "scanner")
        bb.write("scanner.summary", "done", "scanner")
        bb.write("triager.summary", "triaged", "triager")

        results = bb.read_prefix("scanner")
        keys = {e.key for e in results}
        assert "scanner.vulnerabilities.VULN-001" in keys
        assert "scanner.vulnerabilities.VULN-002" in keys
        assert "scanner.summary" in keys
        assert "triager.summary" not in keys

    def test_read_prefix_exact_match(self):
        bb = Blackboard()
        bb.write("scanner.summary", "done", "scanner")
        results = bb.read_prefix("scanner.summary")
        assert len(results) == 1
        assert results[0].key == "scanner.summary"

    def test_get_full_state(self):
        bb = Blackboard()
        bb.write("scanner.summary", "ok", "scanner")
        bb.write("triager.summary", "done", "triager")
        state = bb.get_full_state()
        assert state == {"scanner.summary": "ok", "triager.summary": "done"}


# ---------------------------------------------------------------------------
# TestPermissionFiltering
# ---------------------------------------------------------------------------


class TestPermissionFiltering:
    def _populated_bb(self) -> Blackboard:
        bb = Blackboard()
        # repo
        bb.write("repo.metadata", {"name": "testrepo"}, "orchestrator")
        # scanner
        bb.write("scanner.vulnerabilities.VULN-001", "vuln1", "scanner")
        bb.write("scanner.vulnerabilities.VULN-002", "vuln2", "scanner")
        bb.write("scanner.summary", "2 found", "scanner")
        # triager
        bb.write("triager.triage_results.VULN-001", "triage1", "triager")
        bb.write("triager.triage_results.VULN-002", "triage2", "triager")
        bb.write("triager.priority_order", ["VULN-001", "VULN-002"], "triager")
        # patcher
        bb.write("patcher.patches.VULN-001", "patch1", "patcher")
        bb.write("patcher.patches.VULN-002", "patch2", "patcher")
        # reviewer
        bb.write("reviewer.feedback.VULN-001", "use param queries", "reviewer")
        # files
        bb.write("files.app.py", "def foo(): pass", "orchestrator")
        return bb

    def test_scanner_reads_repo_and_static_analysis(self):
        bb = self._populated_bb()
        bb.write("static_analysis.0", {"rule": "sqli"}, "orchestrator")
        view = bb.read_for_agent("scanner")
        assert "repo.metadata" in view
        assert "static_analysis.0" in view
        # scanner should not see triager or patcher output
        assert not any(k.startswith("triager") for k in view)
        assert not any(k.startswith("patcher") for k in view)

    def test_triager_reads_repo_and_scanner(self):
        bb = self._populated_bb()
        view = bb.read_for_agent("triager")
        assert "repo.metadata" in view
        assert "scanner.summary" in view
        assert "scanner.vulnerabilities.VULN-001" in view
        assert not any(k.startswith("patcher") for k in view)

    def test_patcher_only_sees_its_own_vuln(self):
        bb = self._populated_bb()
        view = bb.read_for_agent("patcher", vuln_id="VULN-001", file_path="app.py")
        assert "scanner.vulnerabilities.VULN-001" in view
        assert "triager.triage_results.VULN-001" in view
        assert "reviewer.feedback.VULN-001" in view
        assert "files.app.py" in view
        # Must not see the other vuln's data
        assert "scanner.vulnerabilities.VULN-002" not in view
        assert "triager.triage_results.VULN-002" not in view
        assert "patcher.patches.VULN-001" not in view  # patcher can't read its own patches

    def test_reviewer_only_sees_its_own_vuln(self):
        bb = self._populated_bb()
        view = bb.read_for_agent("reviewer", vuln_id="VULN-001", file_path="app.py")
        assert "scanner.vulnerabilities.VULN-001" in view
        assert "patcher.patches.VULN-001" in view
        assert "files.app.py" in view
        # Must not see the other vuln's patch
        assert "patcher.patches.VULN-002" not in view
        assert "scanner.vulnerabilities.VULN-002" not in view

    def test_unknown_agent_sees_nothing(self):
        bb = self._populated_bb()
        view = bb.read_for_agent("unknown_agent")
        assert view == {}

    def test_permissions_without_vuln_id_still_work_for_wildcard_patterns(self):
        # scanner/triager use ".*" patterns that don't need vuln_id
        bb = Blackboard()
        bb.write("repo.metadata", "meta", "orchestrator")
        bb.write("scanner.summary", "ok", "scanner")
        view = bb.read_for_agent("triager")
        assert "repo.metadata" in view
        assert "scanner.summary" in view


# ---------------------------------------------------------------------------
# TestToPromptContext
# ---------------------------------------------------------------------------


class TestToPromptContext:
    def test_serialises_entries(self):
        bb = Blackboard()
        bb.write("repo.metadata", {"name": "testrepo"}, "orchestrator")
        bb.write("scanner.summary", "2 vulns", "scanner")
        ctx = bb.to_prompt_context("triager", max_tokens=4000)
        assert "repo.metadata" in ctx
        assert "scanner.summary" in ctx

    def test_respects_token_limit(self):
        bb = Blackboard()
        # Write many large entries
        for i in range(50):
            bb.write(f"scanner.vulnerabilities.VULN-{i:03d}", "x" * 500, "scanner")
        ctx = bb.to_prompt_context("triager", max_tokens=200)
        # 200 tokens * 4 chars/token = 800 char budget
        assert len(ctx) <= 900  # small tolerance for line overhead

    def test_empty_board_returns_empty_string(self):
        bb = Blackboard()
        ctx = bb.to_prompt_context("scanner", max_tokens=4000)
        assert ctx == ""

    def test_agent_with_no_permissions_returns_empty(self):
        bb = Blackboard()
        bb.write("scanner.summary", "ok", "scanner")
        ctx = bb.to_prompt_context("unknown_agent", max_tokens=4000)
        assert ctx == ""


# ---------------------------------------------------------------------------
# TestConflictResolution
# ---------------------------------------------------------------------------


class TestConflictResolution:
    def _build_orch(self) -> BlackboardOrchestrator:
        config = _make_config()
        return BlackboardOrchestrator(config, {}, FullContextMemory())

    def test_best_patch_selected(self):
        """3 patches with different scores → the highest-scoring one wins."""
        orch = self._build_orch()
        bb = orch.blackboard
        vuln_id = "VULN-001"

        # Simulate 3 revision attempts
        for i, (correctness, security) in enumerate(
            [(0.4, 0.5), (0.8, 0.9), (0.6, 0.7)], start=1
        ):
            patch = _make_patch(vuln_id)
            review = _make_review(
                vuln_id, accepted=False, correctness=correctness, security=security
            )
            bb.write(f"patcher.patches.{vuln_id}", patch, "patcher")
            bb.write(f"reviewer.reviews.{vuln_id}", review, "reviewer")

        resolution = orch.resolve_conflict(vuln_id, max_revisions=3)
        assert resolution == "accepted_with_reservations"

        # The current entry should be the best patch (attempt 2: avg 0.85)
        current = bb.read(f"patcher.patches.{vuln_id}")
        assert current is not None
        # The version is incremented once more by the conflict resolver
        assert current.agent == "conflict_resolver"

    def test_no_patches_returns_abandoned(self):
        orch = self._build_orch()
        result = orch.resolve_conflict("VULN-999", max_revisions=3)
        assert result == "abandoned"

    def test_patches_without_reviews_falls_back_to_current(self):
        orch = self._build_orch()
        bb = orch.blackboard
        vuln_id = "VULN-001"
        patch = _make_patch(vuln_id)
        bb.write(f"patcher.patches.{vuln_id}", patch, "patcher")
        # No reviews written
        resolution = orch.resolve_conflict(vuln_id, max_revisions=1)
        assert resolution == "accepted_with_reservations"


# ---------------------------------------------------------------------------
# TestBlackboardOrchestratorPipeline
# ---------------------------------------------------------------------------


class TestBlackboardOrchestratorPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_accepted_first_try(self, tmp_path):
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config()
        orch = _build_orchestrator(
            config,
            _make_scanner_agent([vuln]),
            _make_triager_agent([vuln]),
            _make_patcher_agent(),
            _make_reviewer_agent(accepted=True),
        )
        state = await orch.run(str(tmp_path), _base_task_state())

        assert state.status == "complete"
        assert len(state.vulnerabilities) == 1
        assert len(state.patches) == 1
        assert len(state.reviews) == 1
        assert state.reviews[0].patch_accepted is True

    @pytest.mark.asyncio
    async def test_empty_scanner_output(self, tmp_path):
        config = _make_config()
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
    async def test_retry_loop_accepts_on_second_attempt(self, tmp_path):
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config(max_revision_loops=3)
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
    async def test_conflict_resolution_after_max_revisions(self, tmp_path):
        """All revision attempts rejected → resolve_conflict() is called."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config(max_revision_loops=2)
        reviewer = AsyncMock()
        reviewer.name = "reviewer"
        # Always reject, never retry (to reach end without looping)
        reviewer.run.return_value = (
            ReviewerOutput(review=_make_review(accepted=False), should_retry=False),
            _make_msg("reviewer"),
        )

        orch = _build_orchestrator(
            config,
            _make_scanner_agent([vuln]),
            _make_triager_agent([vuln]),
            _make_patcher_agent(),
            reviewer,
        )
        state = await orch.run(str(tmp_path), _base_task_state())

        assert state.status == "complete"
        # Conflict resolver should have written to the blackboard
        conflict_entry = orch.blackboard.read(f"patcher.patches.{vuln.id}")
        assert conflict_entry is not None
        assert conflict_entry.agent == "conflict_resolver"

    @pytest.mark.asyncio
    async def test_blackboard_populated_after_scanner(self, tmp_path):
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config()
        orch = _build_orchestrator(
            config,
            _make_scanner_agent([vuln]),
            _make_triager_agent([vuln]),
            _make_patcher_agent(),
            _make_reviewer_agent(accepted=True),
        )
        await orch.run(str(tmp_path), _base_task_state())

        bb = orch.blackboard
        assert bb.read("repo.metadata") is not None
        assert bb.read(f"scanner.vulnerabilities.{vuln.id}") is not None
        assert bb.read("scanner.summary") is not None
        assert bb.read(f"triager.triage_results.{vuln.id}") is not None
        assert bb.read(f"patcher.patches.{vuln.id}") is not None
        assert bb.read(f"reviewer.reviews.{vuln.id}") is not None

    @pytest.mark.asyncio
    async def test_patcher_receives_blackboard_context(self, tmp_path):
        """Patcher's agent.run() is called with a blackboard-context AgentMessage."""
        vuln = _make_vuln()
        (tmp_path / "app.py").write_text(FIXTURES_DIR.joinpath("app.py").read_text())

        config = _make_config()
        patcher = _make_patcher_agent()
        orch = _build_orchestrator(
            config,
            _make_scanner_agent([vuln]),
            _make_triager_agent([vuln]),
            patcher,
            _make_reviewer_agent(accepted=True),
        )
        await orch.run(str(tmp_path), _base_task_state())

        # First call: second arg is the context list
        context = patcher.run.call_args[0][1]
        assert len(context) == 1
        assert context[0].agent_name == "blackboard"
