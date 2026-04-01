import asyncio
import logging
import os
from typing import Optional

import anthropic
import pydantic

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


class SequentialOrchestrator(BaseOrchestrator):
    """
    Architecture A: Linear pipeline.
    Scanner → Triager → Patcher → Reviewer → (retry or done)
    Each agent sees the full message history (no filtering, no summarization).
    """

    async def run(self, repo_path: str, task_state: TaskState) -> TaskState:
        """Full pipeline execution for one repository."""
        task_state.status = "scanning"
        repo_metadata = extract_repo_metadata(repo_path, task_state.language)

        # Static analysis (pre-scanner)
        static_findings = []
        if self.config.agents.scanner.use_static_analysis:
            try:
                static_findings = await run_semgrep(repo_path, repo_metadata.language)
                static_findings += await run_dependency_audit(repo_path, repo_metadata.language)
            except Exception as exc:
                logger.warning("Static analysis failed, continuing without it: %s", exc)

        # --- Scanner ---
        scanner_input = ScannerInput(
            repo_path=repo_path,
            target_files=task_state.target_files,  # Empty = scan all files
            language=repo_metadata.language,
            static_analysis_results=static_findings or None,
        )
        try:
            scanner_output, scanner_msg = await self._run_agent(
                self.agents["scanner"], scanner_input, self.memory.retrieve("scanner")
            )
        except Exception as exc:
            logger.error("Scanner failed fatally: %s", exc, exc_info=True)
            task_state.status = "failed"
            return task_state

        await self._record(task_state, scanner_msg)
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
            triager_output, triager_msg = await self._run_agent(
                self.agents["triager"], triager_input, self.memory.retrieve("triager")
            )
        except Exception as exc:
            logger.error("Triager failed fatally: %s", exc, exc_info=True)
            task_state.status = "failed"
            return task_state

        await self._record(task_state, triager_msg)
        task_state.triage_results = triager_output.triage_results

        # --- Patch + Review loop ---
        task_state.status = "patching"
        max_loops = self.config.agents.patcher.max_revision_loops

        for vuln_id in triager_output.priority_order:
            vuln = next((v for v in task_state.vulnerabilities if v.id == vuln_id), None)
            triage = next((t for t in task_state.triage_results if t.vuln_id == vuln_id), None)
            if vuln is None or triage is None:
                logger.warning("Skipping vuln %s: not found in triage results", vuln_id)
                continue

            file_path = os.path.join(repo_path, vuln.file_path)
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    file_content = f.read()
            except OSError as exc:
                logger.error("Cannot read file %s for vuln %s: %s", file_path, vuln_id, exc)
                task_state.failed_vulns.append(vuln_id)
                continue

            revision_feedback: Optional[str] = None
            previous_patch = None

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
                    patcher_output, patcher_msg = await self._run_agent(
                        self.agents["patcher"], patcher_input, self.memory.retrieve("patcher")
                    )
                except Exception as exc:
                    logger.error(
                        "Patcher failed for vuln %s attempt %d: %s", vuln_id, attempt, exc,
                        exc_info=True,
                    )
                    task_state.failed_vulns.append(vuln_id)
                    break

                await self._record(task_state, patcher_msg)

                # Optional: run tests on patched code
                test_result = None
                if self.config.agents.patcher.run_tests and repo_metadata.has_tests:
                    try:
                        test_result = await run_tests_on_patched_code(
                            repo_path, patcher_output.patch, repo_metadata.language
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
                    reviewer_output, reviewer_msg = await self._run_agent(
                        self.agents["reviewer"], reviewer_input, self.memory.retrieve("reviewer")
                    )
                except Exception as exc:
                    logger.error(
                        "Reviewer failed for vuln %s attempt %d: %s", vuln_id, attempt, exc,
                        exc_info=True,
                    )
                    task_state.failed_vulns.append(vuln_id)
                    break

                await self._record(task_state, reviewer_msg)
                task_state.patches.append(patcher_output.patch)
                task_state.reviews.append(reviewer_output.review)

                if reviewer_output.review.patch_accepted:
                    logger.info("Vuln %s accepted on attempt %d", vuln_id, attempt)
                    break

                if reviewer_output.should_retry and attempt < max_loops:
                    revision_feedback = reviewer_output.review.revision_request
                    previous_patch = patcher_output.patch
                    task_state.revision_count += 1
                    task_state.status = "patching"
                    logger.info(
                        "Vuln %s rejected on attempt %d — retrying with feedback", vuln_id, attempt
                    )
                else:
                    logger.info(
                        "Vuln %s not accepted after %d attempt(s) — moving on", vuln_id, attempt
                    )
                    break

        task_state.status = "complete"
        return task_state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _run_agent(self, agent, input_data, context: list[AgentMessage]):
        """Run an agent with rate-limit back-off and parse-error logging."""
        try:
            return await agent.run(input_data, context)
        except anthropic.RateLimitError as exc:
            retry_after = getattr(exc, "retry_after", None) or 30
            logger.warning(
                "Rate limit hit for %s — sleeping %ds", agent.name, retry_after
            )
            await asyncio.sleep(retry_after)
            return await agent.run(input_data, context)
        except pydantic.ValidationError as exc:
            logger.error(
                "Agent %s returned unparseable output: %s", agent.name, exc, exc_info=True
            )
            raise
        except Exception:
            raise

    async def _record(self, task_state: TaskState, msg: AgentMessage) -> None:
        """Store a message in both memory and task_state."""
        await self.memory.store(msg)
        task_state.messages.append(msg)
