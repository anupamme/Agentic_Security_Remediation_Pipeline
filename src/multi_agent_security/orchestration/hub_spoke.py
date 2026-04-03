import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import anthropic
import pydantic
from pydantic import BaseModel

from multi_agent_security.agents.patcher import PatcherInput
from multi_agent_security.agents.reviewer import ReviewerInput
from multi_agent_security.agents.scanner import ScannerInput
from multi_agent_security.agents.triager import TriagerInput
from multi_agent_security.llm_client import LLMClient
from multi_agent_security.orchestration.base import BaseOrchestrator
from multi_agent_security.tools.code_parser import extract_repo_metadata
from multi_agent_security.tools.static_analysis import run_dependency_audit, run_semgrep
from multi_agent_security.tools.test_runner import run_tests_on_patched_code
from multi_agent_security.types import AgentMessage, TaskState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

_ROUTING_SYSTEM_PROMPT = """\
You are an orchestration controller managing a team of security agents. \
Before invoking the next agent, you must decide what context they need.

You have access to the full message history. The next agent to run is: {next_agent_name}

Your job:
1. Summarize the relevant context from previous agents' outputs
2. Decide if any information should be withheld (to save context space) or highlighted
3. Format the context package for the next agent

Rules:
- Scanner needs: file contents + static analysis findings only
- Triager needs: vulnerability list + repo metadata (NOT full file contents)
- Patcher needs: specific vulnerability details + triage + full file content for THAT file only (not other files)
- Reviewer needs: vulnerability + triage + original code + patch + test results (NOT scanner raw output)

Respond with JSON:
{{
  "context_summary": "A concise summary of relevant information for the next agent",
  "included_messages": [list of message indices to include in full],
  "excluded_messages": [list of message indices to exclude],
  "routing_reasoning": "Why you're routing this way"
}}"""


class RoutingDecision(BaseModel):
    context_summary: str
    included_messages: list[int]
    excluded_messages: list[int]
    routing_reasoning: str


class ContextPackage(BaseModel):
    summary: str
    full_messages: list[AgentMessage]
    agent_name: str


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class HubSpokeOrchestrator(BaseOrchestrator):
    """
    Architecture B: All communication routed through orchestrator.
    Agents never see each other's raw output.
    Orchestrator summarizes and filters context for each agent call.
    """

    def __init__(self, config, agents: dict, memory, routing_llm_client=None):
        super().__init__(config, agents, memory)
        self._orch_llm_calls: int = 0
        self._orch_tokens_in: int = 0
        self._orch_tokens_out: int = 0
        self._total_tokens_available: int = 0
        self._total_tokens_passed: int = 0
        self._information_loss_events: int = 0

        if routing_llm_client is not None:
            self._routing_llm: Optional[LLMClient] = routing_llm_client
        elif config.orchestrator.type == "llm_based":
            # Build a routing LLM client using the routing_model from config.
            # We create a new LLMConfig with the routing model, reusing provider/auth.
            from multi_agent_security.config import LLMConfig
            routing_llm_config = LLMConfig(
                provider=config.llm.provider,
                model=config.orchestrator.routing_model,
                max_tokens=config.orchestrator.max_summary_tokens,
                temperature=0.0,
                api_key_env=config.llm.api_key_env,
                aws_region=config.llm.aws_region,
                aws_profile=config.llm.aws_profile,
            )
            self._routing_llm = LLMClient(routing_llm_config)
        else:
            self._routing_llm = None

    @property
    def metrics(self) -> dict:
        ratio = (
            self._total_tokens_passed / self._total_tokens_available
            if self._total_tokens_available > 0
            else 0.0
        )
        return {
            "orchestrator_llm_calls": self._orch_llm_calls,
            "orchestrator_tokens_in": self._orch_tokens_in,
            "orchestrator_tokens_out": self._orch_tokens_out,
            "context_compression_ratio": ratio,
            "information_loss_events": self._information_loss_events,
        }

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

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
        scanner_context = await self._get_context_for_agent("scanner", task_state.messages)
        scanner_input = ScannerInput(
            repo_path=repo_path,
            target_files=task_state.target_files,
            language=repo_metadata.language,
            static_analysis_results=static_findings or None,
        )
        try:
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
            self._log_metrics()
            return task_state

        # --- Triager ---
        task_state.status = "triaging"
        triager_context = await self._get_context_for_agent("triager", task_state.messages)
        triager_input = TriagerInput(
            vulnerabilities=scanner_output.vulnerabilities,
            repo_metadata=repo_metadata,
        )
        try:
            triager_output, triager_msg = await self._run_agent(
                self.agents["triager"], triager_input, triager_context
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
                patcher_context = await self._get_context_for_agent("patcher", task_state.messages)
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

                # --- Reviewer ---
                task_state.status = "reviewing"
                reviewer_context = await self._get_context_for_agent("reviewer", task_state.messages)
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
        self._log_metrics()
        return task_state

    # ------------------------------------------------------------------
    # Context routing
    # ------------------------------------------------------------------

    async def _get_context_for_agent(
        self, agent_name: str, messages: list[AgentMessage]
    ) -> list[AgentMessage]:
        """Return a filtered context list for the given agent."""
        total_available = sum(
            m.token_count_input + m.token_count_output for m in messages
        )
        self._total_tokens_available += total_available

        if self.config.orchestrator.type == "llm_based":
            filtered = await self._llm_based_context(agent_name, messages)
        else:
            filtered = self._rule_based_context(agent_name, messages)

        tokens_passed = sum(m.token_count_input + m.token_count_output for m in filtered)
        self._total_tokens_passed += tokens_passed

        return filtered

    def _rule_based_context(
        self, agent_name: str, messages: list[AgentMessage]
    ) -> list[AgentMessage]:
        """Deterministic context filtering — no LLM call."""
        if agent_name == "scanner":
            return []
        if agent_name == "triager":
            return [m for m in messages if m.agent_name == "scanner"]
        if agent_name in ("patcher", "reviewer"):
            return [m for m in messages if m.agent_name in ("triager", "patcher")]
        # Unknown agent: pass nothing
        return []

    async def _llm_based_context(
        self, agent_name: str, messages: list[AgentMessage]
    ) -> list[AgentMessage]:
        """LLM-based routing: call the routing LLM to decide context."""
        if not messages:
            return []

        history_lines = "\n".join(
            f"[{i}] {m.agent_name}: {m.content[:200]}..."
            if len(m.content) > 200 else f"[{i}] {m.agent_name}: {m.content}"
            for i, m in enumerate(messages)
        )
        system_prompt = _ROUTING_SYSTEM_PROMPT.format(next_agent_name=agent_name)
        user_prompt = (
            f"Message history ({len(messages)} messages):\n{history_lines}\n\n"
            f"Decide what context to pass to the next agent: {agent_name}"
        )

        try:
            response = await self._routing_llm.complete(
                system_prompt, user_prompt, response_format=RoutingDecision
            )
        except Exception as exc:
            logger.warning(
                "Routing LLM call failed for %s, falling back to rule-based: %s",
                agent_name, exc,
            )
            return self._rule_based_context(agent_name, messages)

        self._orch_llm_calls += 1
        self._orch_tokens_in += response.input_tokens
        self._orch_tokens_out += response.output_tokens

        decision = RoutingDecision.model_validate_json(response.content)
        return self._package_context(decision, messages)

    def _package_context(
        self, decision: RoutingDecision, messages: list[AgentMessage]
    ) -> list[AgentMessage]:
        """Build the filtered message list from a routing decision."""
        summary_msg = AgentMessage(
            agent_name="orchestrator",
            timestamp=datetime.now(timezone.utc),
            content=decision.context_summary,
            token_count_input=0,
            token_count_output=max(1, len(decision.context_summary) // 4),
            latency_ms=0.0,
            cost_usd=0.0,
        )
        included = [
            messages[i]
            for i in decision.included_messages
            if 0 <= i < len(messages)
        ]
        return [summary_msg] + included

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    async def _record(self, task_state: TaskState, msg: AgentMessage) -> None:
        """Store a message in both memory and task_state."""
        await self.memory.store(msg)
        task_state.messages.append(msg)

    def _log_metrics(self) -> None:
        m = self.metrics
        logger.info(
            "Hub-spoke metrics: orch_llm_calls=%d orch_tokens_in=%d orch_tokens_out=%d "
            "context_compression_ratio=%.3f information_loss_events=%d",
            m["orchestrator_llm_calls"],
            m["orchestrator_tokens_in"],
            m["orchestrator_tokens_out"],
            m["context_compression_ratio"],
            m["information_loss_events"],
        )
