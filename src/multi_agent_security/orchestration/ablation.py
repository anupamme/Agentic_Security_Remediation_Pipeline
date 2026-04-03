"""
Ablation orchestrators for measuring the marginal value of each agent.

AblationOrchestrator  — Sequential pipeline with configurable agent skipping.
SingleAgentOrchestrator — Wraps SingleAgent as a drop-in pipeline replacement.
"""
import logging
import os
from typing import Optional

from multi_agent_security.agents.patcher import PatcherInput
from multi_agent_security.agents.reviewer import ReviewerInput
from multi_agent_security.agents.scanner import ScannerInput
from multi_agent_security.agents.single_agent import SingleAgentInput
from multi_agent_security.agents.triager import TriagerInput
from multi_agent_security.orchestration.base import BaseOrchestrator
from multi_agent_security.orchestration.sequential import SequentialOrchestrator
from multi_agent_security.tools.code_parser import extract_repo_metadata
from multi_agent_security.tools.static_analysis import run_dependency_audit, run_semgrep
from multi_agent_security.tools.test_runner import run_tests_on_patched_code
from multi_agent_security.types import (
    AgentMessage,
    FixStrategy,
    ReviewResult,
    TaskState,
    TriageResult,
    VulnSeverity,
)

logger = logging.getLogger(__name__)


class AblationOrchestrator(SequentialOrchestrator):
    """
    Configurable orchestrator that can skip specific agents.
    Used for ablation studies (A1–A4).

    skip_agents options:
      "triager"  — skip triage; assign default severity/strategy, order by confidence
      "reviewer" — skip review; auto-accept all patches
      "patcher"  — skip patching (and reviewing); scanner-only run (A4)
    """

    def __init__(self, config, agents: dict, memory, skip_agents: list[str] = None):
        super().__init__(config, agents, memory)
        self.skip_agents: set[str] = set(skip_agents or [])

    async def run(self, repo_path: str, task_state: TaskState) -> TaskState:
        """Pipeline with selected agents optionally skipped."""
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

        # --- Scanner (always runs) ---
        scanner_input = ScannerInput(
            repo_path=repo_path,
            target_files=task_state.target_files,
            language=repo_metadata.language,
            static_analysis_results=static_findings or None,
        )
        try:
            scanner_context = await self.memory.retrieve("scanner")
            scanner_output, scanner_msg = await self._run_agent(
                self.agents["scanner"], scanner_input, scanner_context
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

        # A4 — scan only: stop here
        if "patcher" in self.skip_agents:
            logger.info("Patcher skipped (scan-only ablation) — pipeline complete.")
            task_state.status = "complete"
            return task_state

        # --- Triager (or defaults) ---
        task_state.status = "triaging"
        if "triager" in self.skip_agents:
            logger.info("Triager skipped — assigning default triage values.")
            task_state.triage_results = [
                TriageResult(
                    vuln_id=v.id,
                    severity=VulnSeverity.MEDIUM,
                    exploitability_score=0.5,
                    fix_strategy=FixStrategy.REFACTOR,
                    estimated_complexity="medium",
                    triage_reasoning="Triager skipped — using default values",
                )
                for v in scanner_output.vulnerabilities
            ]
            # Order by scanner confidence (already sorted descending by scanner)
            priority_order = [v.id for v in scanner_output.vulnerabilities]
        else:
            triager_input = TriagerInput(
                vulnerabilities=scanner_output.vulnerabilities,
                repo_metadata=repo_metadata,
            )
            try:
                triager_context = await self.memory.retrieve("triager")
                triager_output, triager_msg = await self._run_agent(
                    self.agents["triager"], triager_input, triager_context
                )
            except Exception as exc:
                logger.error("Triager failed fatally: %s", exc, exc_info=True)
                task_state.status = "failed"
                return task_state

            await self._record(task_state, triager_msg)
            task_state.triage_results = triager_output.triage_results
            priority_order = triager_output.priority_order

        # --- Patch loop ---
        task_state.status = "patching"
        max_loops = self.config.agents.patcher.max_revision_loops

        for vuln_id in priority_order:
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
                    patcher_context = await self.memory.retrieve("patcher")
                    patcher_output, patcher_msg = await self._run_agent(
                        self.agents["patcher"], patcher_input, patcher_context
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

                task_state.patches.append(patcher_output.patch)

                # --- Reviewer (or auto-accept) ---
                if "reviewer" in self.skip_agents:
                    logger.debug("Reviewer skipped — auto-accepting patch for vuln %s", vuln_id)
                    auto_review = ReviewResult(
                        vuln_id=vuln_id,
                        patch_accepted=True,
                        correctness_score=1.0,
                        security_score=1.0,
                        style_score=1.0,
                        review_reasoning="Reviewer skipped — auto-accepted",
                        revision_request=None,
                    )
                    task_state.reviews.append(auto_review)
                    break  # Accept on first attempt, no retry loop needed

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
                    reviewer_context = await self.memory.retrieve("reviewer")
                    reviewer_output, reviewer_msg = await self._run_agent(
                        self.agents["reviewer"], reviewer_input, reviewer_context
                    )
                except Exception as exc:
                    logger.error(
                        "Reviewer failed for vuln %s attempt %d: %s", vuln_id, attempt, exc,
                        exc_info=True,
                    )
                    task_state.failed_vulns.append(vuln_id)
                    break

                await self._record(task_state, reviewer_msg)
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


class SingleAgentOrchestrator(BaseOrchestrator):
    """
    Runs the single-agent baseline — one LLM call per file batch.
    Expects agents["single_agent"] to be a SingleAgent instance.
    """

    async def run(self, repo_path: str, task_state: TaskState) -> TaskState:
        task_state.status = "scanning"

        single_agent_input = SingleAgentInput(
            repo_path=repo_path,
            target_files=task_state.target_files,
            language=task_state.language,
        )

        try:
            output, msg = await self.agents["single_agent"].run(
                single_agent_input, context=[]
            )
        except Exception as exc:
            logger.error("SingleAgent failed fatally: %s", exc, exc_info=True)
            task_state.status = "failed"
            return task_state

        await self.memory.store(msg)
        task_state.messages.append(msg)

        task_state.vulnerabilities = output.vulnerabilities
        task_state.triage_results = output.triage_results
        task_state.patches = output.patches
        task_state.reviews = output.reviews
        task_state.status = "complete"

        return task_state
