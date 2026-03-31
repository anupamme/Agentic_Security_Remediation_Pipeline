import asyncio
import fnmatch
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import anthropic
import pydantic
from pydantic import BaseModel

from multi_agent_security.agents.patcher import PatcherInput, PatcherOutput
from multi_agent_security.agents.reviewer import ReviewerInput, ReviewerOutput
from multi_agent_security.agents.scanner import ScannerInput, ScannerOutput
from multi_agent_security.agents.triager import TriagerInput, TriagerOutput
from multi_agent_security.orchestration.base import BaseOrchestrator
from multi_agent_security.tools.code_parser import extract_repo_metadata
from multi_agent_security.tools.static_analysis import run_dependency_audit, run_semgrep
from multi_agent_security.tools.test_runner import run_tests_on_patched_code
from multi_agent_security.types import AgentMessage, TaskState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-loading constants
# ---------------------------------------------------------------------------

# Directories that are never useful to load into the blackboard.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", ".npm",
    "vendor", "third_party",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "target", "build", "dist", "out", "bin", "obj",
    ".venv", "venv", "env", ".env",
    ".idea", ".vscode",
    "coverage", ".nyc_output",
})

# Source-file extensions per language (mirrors scanner._LANG_EXTENSIONS).
_LANG_EXTENSIONS: dict[str, list[str]] = {
    "python": ["*.py"],
    "javascript": ["*.js", "*.ts", "*.jsx", "*.tsx"],
    "go": ["*.go"],
    "java": ["*.java"],
}

# Hard cap: skip any file larger than this (bytes).
_MAX_FILE_BYTES: int = 256 * 1024  # 256 KB

# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------

AGENT_READ_PERMISSIONS: dict[str, list[str]] = {
    "scanner": [
        "repo.*",
        "static_analysis.*",
    ],
    "triager": [
        "repo.*",
        "scanner.*",
    ],
    "patcher": [
        "repo.*",
        "scanner.vulnerabilities.{vuln_id}",
        "triager.triage_results.{vuln_id}",
        "reviewer.feedback.{vuln_id}",
        "files.{file_path}",
    ],
    "reviewer": [
        "scanner.vulnerabilities.{vuln_id}",
        "triager.triage_results.{vuln_id}",
        "patcher.patches.{vuln_id}",
        "files.{file_path}",
        "tests.{vuln_id}",
    ],
}

AGENT_WRITE_KEYS: dict[str, list[str]] = {
    "scanner": ["scanner.vulnerabilities", "scanner.summary"],
    "triager": ["triager.triage_results", "triager.priority_order", "triager.summary"],
    "patcher": ["patcher.patches.{vuln_id}"],
    "reviewer": ["reviewer.reviews.{vuln_id}", "reviewer.feedback.{vuln_id}"],
}

# ---------------------------------------------------------------------------
# Blackboard data structures
# ---------------------------------------------------------------------------


class BlackboardEntry(BaseModel):
    key: str
    value: Any
    agent: str
    timestamp: datetime
    version: int = 1


class Blackboard:
    """
    Shared state object that all agents can read from and write to.
    Organised as a hierarchical key-value store.
    """

    def __init__(self) -> None:
        self._entries: dict[str, BlackboardEntry] = {}
        self._history: list[BlackboardEntry] = []  # Append-only log of every version

    # ------------------------------------------------------------------
    # Write / Read
    # ------------------------------------------------------------------

    def write(self, key: str, value: Any, agent: str) -> None:
        """Write or update a key. Increments version if the key already exists."""
        existing = self._entries.get(key)
        version = (existing.version + 1) if existing is not None else 1
        if existing is not None:
            self._history.append(existing)
        entry = BlackboardEntry(
            key=key,
            value=value,
            agent=agent,
            timestamp=datetime.now(timezone.utc),
            version=version,
        )
        self._entries[key] = entry

    def read(self, key: str) -> Optional[BlackboardEntry]:
        """Read a single key."""
        return self._entries.get(key)

    def read_prefix(self, prefix: str) -> list[BlackboardEntry]:
        """Return all entries whose key starts with `prefix.` or equals `prefix`."""
        needle = prefix if prefix.endswith(".") else prefix + "."
        return [
            entry
            for key, entry in self._entries.items()
            if key == prefix or key.startswith(needle)
        ]

    def read_for_agent(
        self,
        agent_name: str,
        vuln_id: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Return a filtered view of the blackboard for the given agent.
        Uses AGENT_READ_PERMISSIONS; resolves {vuln_id}/{file_path} templates.
        """
        permissions = AGENT_READ_PERMISSIONS.get(agent_name, [])
        result: dict[str, Any] = {}
        for key, entry in self._entries.items():
            if self._key_matches_permissions(key, permissions, vuln_id, file_path):
                result[key] = entry.value
        return result

    def to_prompt_context(
        self,
        agent_name: str,
        vuln_id: Optional[str] = None,
        file_path: Optional[str] = None,
        max_tokens: int = 4000,
    ) -> str:
        """
        Serialise the agent's readable blackboard entries into a text format
        suitable for inclusion in an LLM prompt, truncated to max_tokens.
        Rough estimate: 1 token ≈ 4 characters.
        """
        view = self.read_for_agent(agent_name, vuln_id=vuln_id, file_path=file_path)
        if not view:
            return ""

        budget = max_tokens * 4  # character budget
        lines: list[str] = []
        used = 0
        for key, value in view.items():
            try:
                serialised = json.dumps(value, default=str)
            except Exception:
                serialised = str(value)
            line = f"{key}: {serialised}"
            if used + len(line) + 1 > budget:
                # Truncate this line to fit remaining budget
                remaining = budget - used - 1
                if remaining > 20:
                    lines.append(line[:remaining] + "…")
                break
            lines.append(line)
            used += len(line) + 1  # +1 for newline

        return "\n".join(lines)

    def get_full_state(self) -> dict[str, Any]:
        """Return entire blackboard as a plain dict."""
        return {k: e.value for k, e in self._entries.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key_matches_permissions(
        self,
        key: str,
        permissions: list[str],
        vuln_id: Optional[str],
        file_path: Optional[str],
    ) -> bool:
        for pattern in permissions:
            resolved = pattern
            if vuln_id:
                resolved = resolved.replace("{vuln_id}", vuln_id)
            if file_path:
                resolved = resolved.replace("{file_path}", file_path)

            if resolved.endswith(".*"):
                prefix = resolved[:-1]  # keeps the trailing dot
                if key == resolved[:-2] or key.startswith(prefix):
                    return True
            elif key == resolved or key.startswith(resolved + "."):
                return True
        return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class BlackboardOrchestrator(BaseOrchestrator):
    """
    Architecture C: Shared state.
    All agents read/write to a shared blackboard.
    Orchestrator controls turn order and resolves conflicts.
    """

    def __init__(self, config, agents: dict, memory) -> None:
        super().__init__(config, agents, memory)
        self.blackboard = Blackboard()

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def run(self, repo_path: str, task_state: TaskState) -> TaskState:
        """Full pipeline execution for one repository."""
        task_state.status = "scanning"
        repo_metadata = extract_repo_metadata(repo_path, task_state.language)

        # Initialise blackboard with repo metadata
        self.blackboard.write("repo.metadata", repo_metadata, "orchestrator")

        # Load target files into blackboard
        files_to_load = task_state.target_files or []
        if not files_to_load:
            files_to_load = self._collect_source_files(repo_path, repo_metadata.language)

        loaded = skipped = 0
        for rel_path in files_to_load:
            abs_path = os.path.join(repo_path, rel_path)
            try:
                size = os.path.getsize(abs_path)
                if size > _MAX_FILE_BYTES:
                    logger.debug("Skipping oversized file %s (%d bytes)", rel_path, size)
                    skipped += 1
                    continue
                with open(abs_path, encoding="utf-8", errors="replace") as fh:
                    self.blackboard.write(f"files.{rel_path}", fh.read(), "orchestrator")
                loaded += 1
            except OSError:
                pass
        logger.info("Loaded %d source file(s) into blackboard (%d skipped).", loaded, skipped)

        # Static analysis (pre-scanner)
        static_findings = []
        if self.config.agents.scanner.use_static_analysis:
            try:
                static_findings = await run_semgrep(repo_path, repo_metadata.language)
                static_findings += await run_dependency_audit(repo_path, repo_metadata.language)
                for i, finding in enumerate(static_findings):
                    self.blackboard.write(f"static_analysis.{i}", finding, "orchestrator")
            except Exception as exc:
                logger.warning("Static analysis failed, continuing without it: %s", exc)

        # --- Scanner ---
        scanner_input = ScannerInput(
            repo_path=repo_path,
            target_files=task_state.target_files,
            language=repo_metadata.language,
            static_analysis_results=static_findings or None,
        )
        try:
            scanner_output, _ = await self._run_agent_with_blackboard(
                self.agents["scanner"], scanner_input, task_state
            )
        except Exception as exc:
            logger.error("Scanner failed fatally: %s", exc, exc_info=True)
            task_state.status = "failed"
            return task_state

        task_state.vulnerabilities = scanner_output.vulnerabilities

        if not scanner_output.vulnerabilities:
            logger.info("No vulnerabilities found — pipeline complete.")
            task_state.status = "complete"
            return task_state

        # --- Triager ---
        task_state.status = "triaging"
        triager_input = TriagerInput(
            vulnerabilities=scanner_output.vulnerabilities,
            repo_metadata=repo_metadata,
        )
        try:
            triager_output, _ = await self._run_agent_with_blackboard(
                self.agents["triager"], triager_input, task_state
            )
        except Exception as exc:
            logger.error("Triager failed fatally: %s", exc, exc_info=True)
            task_state.status = "failed"
            return task_state

        task_state.triage_results = triager_output.triage_results

        # --- Patch + Review loop ---
        task_state.status = "patching"
        max_loops = self.config.agents.patcher.max_revision_loops

        priority_order_entry = self.blackboard.read("triager.priority_order")
        priority_order: list[str] = (
            priority_order_entry.value if priority_order_entry else []
        )

        for vuln_id in priority_order:
            vuln_entry = self.blackboard.read(f"scanner.vulnerabilities.{vuln_id}")
            triage_entry = self.blackboard.read(f"triager.triage_results.{vuln_id}")
            if vuln_entry is None or triage_entry is None:
                logger.warning("Skipping vuln %s: not found on blackboard", vuln_id)
                continue

            vuln = vuln_entry.value
            triage = triage_entry.value

            file_entry = self.blackboard.read(f"files.{vuln.file_path}")
            if file_entry is not None:
                file_content = file_entry.value
            else:
                file_path = os.path.join(repo_path, vuln.file_path)
                try:
                    with open(file_path, encoding="utf-8", errors="replace") as fh:
                        file_content = fh.read()
                except OSError as exc:
                    logger.error(
                        "Cannot read file %s for vuln %s: %s", file_path, vuln_id, exc
                    )
                    task_state.failed_vulns.append(vuln_id)
                    continue

            revision_feedback: Optional[str] = None
            previous_patch = None
            accepted = False

            for attempt in range(1, max_loops + 1):
                # --- Patcher ---
                patcher_input = PatcherInput(
                    vulnerability=vuln,
                    triage=triage,
                    file_content=file_content,
                    repo_path=repo_path,
                    language=repo_metadata.language,
                    revision_feedback=revision_feedback,
                    previous_patch=previous_patch,
                )
                try:
                    patcher_output, _ = await self._run_agent_with_blackboard(
                        self.agents["patcher"], patcher_input, task_state, vuln_id=vuln_id
                    )
                except Exception as exc:
                    logger.error(
                        "Patcher failed for vuln %s attempt %d: %s",
                        vuln_id, attempt, exc, exc_info=True,
                    )
                    task_state.failed_vulns.append(vuln_id)
                    break

                # Optional: run tests on patched code
                test_result = None
                if self.config.agents.patcher.run_tests and repo_metadata.has_tests:
                    try:
                        test_result = await run_tests_on_patched_code(
                            repo_path, patcher_output.patch, repo_metadata.language
                        )
                        if test_result is not None:
                            self.blackboard.write(
                                f"tests.{vuln_id}", test_result, "orchestrator"
                            )
                    except Exception as exc:
                        logger.warning("Test run failed for vuln %s: %s", vuln_id, exc)

                # --- Reviewer ---
                task_state.status = "reviewing"
                patched_content = patcher_output.patch.patched_code
                reviewer_input = ReviewerInput(
                    vulnerability=vuln,
                    triage=triage,
                    patch=patcher_output.patch,
                    original_file_content=file_content,
                    patched_file_content=patched_content,
                    language=repo_metadata.language,
                    test_result=test_result,
                    revision_number=attempt,
                )
                try:
                    reviewer_output, _ = await self._run_agent_with_blackboard(
                        self.agents["reviewer"], reviewer_input, task_state, vuln_id=vuln_id
                    )
                except Exception as exc:
                    logger.error(
                        "Reviewer failed for vuln %s attempt %d: %s",
                        vuln_id, attempt, exc, exc_info=True,
                    )
                    task_state.failed_vulns.append(vuln_id)
                    break

                task_state.patches.append(patcher_output.patch)
                task_state.reviews.append(reviewer_output.review)

                if reviewer_output.review.patch_accepted:
                    logger.info("Vuln %s accepted on attempt %d", vuln_id, attempt)
                    accepted = True
                    break

                if reviewer_output.should_retry and attempt < max_loops:
                    revision_feedback = reviewer_output.review.revision_request
                    previous_patch = patcher_output.patch
                    task_state.revision_count += 1
                    task_state.status = "patching"
                    logger.info(
                        "Vuln %s rejected on attempt %d — retrying with feedback",
                        vuln_id, attempt,
                    )
                else:
                    break

            if not accepted:
                resolution = self.resolve_conflict(vuln_id, max_loops)
                logger.info(
                    "Conflict resolution for %s: %s", vuln_id, resolution
                )

        task_state.status = "complete"
        return task_state

    # ------------------------------------------------------------------
    # Blackboard agent runner
    # ------------------------------------------------------------------

    async def _run_agent_with_blackboard(
        self,
        agent,
        input_data: BaseModel,
        task_state: TaskState,
        vuln_id: Optional[str] = None,
    ) -> tuple[BaseModel, AgentMessage]:
        """
        1. Get blackboard context for this agent (filtered by permissions).
        2. Wrap as a synthetic AgentMessage.
        3. Run agent.
        4. Write output back to blackboard.
        5. Record message in task_state.
        """
        context_str = self.blackboard.to_prompt_context(
            agent.name,
            vuln_id=vuln_id,
            max_tokens=self.config.memory.max_context_tokens,
        )
        bb_context = AgentMessage(
            agent_name="blackboard",
            timestamp=datetime.now(timezone.utc),
            content=context_str,
            token_count_input=0,
            token_count_output=0,
            latency_ms=0.0,
            cost_usd=0.0,
        )

        output, msg = await self._run_agent(agent, input_data, [bb_context])
        self._write_output_to_blackboard(agent.name, output, vuln_id)
        self._record(task_state, msg)
        return output, msg

    def _write_output_to_blackboard(
        self, agent_name: str, output: BaseModel, vuln_id: Optional[str]
    ) -> None:
        """Write structured agent output to the blackboard."""
        if agent_name == "scanner":
            assert isinstance(output, ScannerOutput)
            for vuln in output.vulnerabilities:
                self.blackboard.write(
                    f"scanner.vulnerabilities.{vuln.id}", vuln, "scanner"
                )
            self.blackboard.write("scanner.summary", output.scan_summary, "scanner")

        elif agent_name == "triager":
            assert isinstance(output, TriagerOutput)
            for tr in output.triage_results:
                self.blackboard.write(
                    f"triager.triage_results.{tr.vuln_id}", tr, "triager"
                )
            self.blackboard.write(
                "triager.priority_order", output.priority_order, "triager"
            )
            self.blackboard.write("triager.summary", output.summary, "triager")

        elif agent_name == "patcher" and vuln_id:
            assert isinstance(output, PatcherOutput)
            self.blackboard.write(
                f"patcher.patches.{vuln_id}", output.patch, "patcher"
            )

        elif agent_name == "reviewer" and vuln_id:
            assert isinstance(output, ReviewerOutput)
            self.blackboard.write(
                f"reviewer.reviews.{vuln_id}", output.review, "reviewer"
            )
            if output.review.revision_request:
                self.blackboard.write(
                    f"reviewer.feedback.{vuln_id}",
                    output.review.revision_request,
                    "reviewer",
                )

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    def resolve_conflict(self, vuln_id: str, max_revisions: int) -> str:
        """
        After max_revisions with no acceptance, pick the patch with the highest
        average (correctness + security) score and mark it as the current value.

        Returns: "accepted", "accepted_with_reservations", or "abandoned".
        """
        patch_key = f"patcher.patches.{vuln_id}"
        review_key = f"reviewer.reviews.{vuln_id}"

        # Gather all historical patch entries (plus current) sorted by version
        all_patch_entries = sorted(
            [e for e in self.blackboard._history if e.key == patch_key],
            key=lambda e: e.version,
        )
        current_patch = self.blackboard.read(patch_key)
        if current_patch:
            all_patch_entries.append(current_patch)

        all_review_entries = sorted(
            [e for e in self.blackboard._history if e.key == review_key],
            key=lambda e: e.version,
        )
        current_review = self.blackboard.read(review_key)
        if current_review:
            all_review_entries.append(current_review)

        if not all_patch_entries:
            return "abandoned"

        # Build a version → review map
        review_by_version: dict[int, Any] = {
            e.version: e.value for e in all_review_entries
        }

        best_score = -1.0
        best_patch = None
        for patch_entry in all_patch_entries:
            review = review_by_version.get(patch_entry.version)
            if review is None:
                continue
            avg = (review.correctness_score + review.security_score) / 2.0
            if avg > best_score:
                best_score = avg
                best_patch = patch_entry.value

        if best_patch is None:
            # No reviewed patches — fall back to current
            if current_patch:
                best_patch = current_patch.value
                best_score = 0.0
            else:
                return "abandoned"

        # Write the best patch as the current entry
        self.blackboard.write(patch_key, best_patch, "conflict_resolver")
        logger.warning(
            "Conflict resolved for %s: accepted_with_reservations (avg_score=%.2f) "
            "— flagged for human review",
            vuln_id, best_score,
        )
        return "accepted_with_reservations"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_source_files(repo_path: str, language: str) -> list[str]:
        """
        Walk the repo and return relative paths of source files, skipping:
        - directories in _SKIP_DIRS (e.g. .git, node_modules, __pycache__)
        - files that don't match the language's extension patterns
        """
        patterns = _LANG_EXTENSIONS.get(language, ["*.*"])
        result: list[str] = []
        for dirpath, dirnames, filenames in os.walk(repo_path):
            # Prune skip-dirs in-place so os.walk won't descend into them.
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fname in filenames:
                if any(fnmatch.fnmatch(fname, pat) for pat in patterns):
                    abs_path = os.path.join(dirpath, fname)
                    result.append(os.path.relpath(abs_path, repo_path))
        return result

    async def _run_agent(self, agent, input_data, context: list[AgentMessage]):
        """Run an agent with rate-limit back-off and parse-error logging."""
        try:
            return await agent.run(input_data, context)
        except anthropic.RateLimitError as exc:
            retry_after = getattr(exc, "retry_after", None) or 30
            logger.warning("Rate limit hit for %s — sleeping %ds", agent.name, retry_after)
            await asyncio.sleep(retry_after)
            return await agent.run(input_data, context)
        except pydantic.ValidationError as exc:
            logger.error(
                "Agent %s returned unparseable output: %s", agent.name, exc, exc_info=True
            )
            raise
        except Exception:
            raise

    def _record(self, task_state: TaskState, msg: AgentMessage) -> None:
        """Store a message in both memory and task_state."""
        self.memory.store(msg)
        task_state.messages.append(msg)
