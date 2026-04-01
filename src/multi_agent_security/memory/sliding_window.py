import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from multi_agent_security.memory.base import BaseMemory
from multi_agent_security.types import AgentMessage

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM_PROMPT = """\
You are summarizing the work history of a multi-agent security remediation pipeline.
Compress the following agent messages into a concise summary that preserves:
1. Key findings (vulnerabilities detected, their types and severities)
2. Decisions made (triage priorities, fix strategies chosen)
3. Actions taken (patches generated, reviews given)
4. Any unresolved issues or pending retries

Be concise but preserve all information that downstream agents might need.
Maximum length: 500 words."""


class SlidingWindowMemory(BaseMemory):
    """Keep the last N messages in full. Summarize everything older into
    a compressed summary that's prepended to the context."""

    def __init__(
        self,
        window_size: int = 5,
        summary_model: str = "claude-sonnet-4-20250514",
        llm_client=None,
    ):
        self.window_size = window_size
        self.summary_model = summary_model
        self.llm_client = llm_client
        self._messages: list[AgentMessage] = []
        self._summary: Optional[str] = None
        self._summary_covers_up_to: int = 0  # index (exclusive) of last summarized range

    def store(self, message: AgentMessage) -> None:
        """Append message. If messages exceed window, trigger re-summarization."""
        self._messages.append(message)
        messages_outside_window = len(self._messages) - self.window_size
        if messages_outside_window > self._summary_covers_up_to:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._update_summary())
                else:
                    loop.run_until_complete(self._update_summary())
            except RuntimeError:
                # No event loop — run synchronous fallback immediately
                self._update_summary_sync()

    def retrieve(self, agent_name: str, query: Optional[str] = None) -> list[AgentMessage]:
        """Return [summary_as_message] + last N messages."""
        context: list[AgentMessage] = []
        if self._summary:
            context.append(
                AgentMessage(
                    agent_name="memory_summary",
                    timestamp=datetime.now(timezone.utc),
                    content=f"SUMMARY OF EARLIER CONTEXT:\n{self._summary}",
                    token_count_input=0,
                    token_count_output=0,
                    latency_ms=0.0,
                    cost_usd=0.0,
                )
            )
        context.extend(self._messages[-self.window_size :])
        return context

    async def _update_summary(self) -> None:
        """Summarize messages[0 : len-window_size] via LLM."""
        cutoff = len(self._messages) - self.window_size
        if cutoff <= 0 or cutoff <= self._summary_covers_up_to:
            return

        to_summarize = self._messages[:cutoff]

        if self.llm_client is None:
            self._apply_extractive_summary(to_summarize, cutoff)
            return

        formatted = "\n\n".join(
            f"[{m.agent_name} @ {m.timestamp.isoformat()}]\n{m.content}"
            for m in to_summarize
        )
        try:
            response = await self.llm_client.complete(
                system_prompt=_SUMMARY_SYSTEM_PROMPT,
                user_prompt=formatted,
                agent_name="memory_summarizer",
            )
            self._summary = response.content
            self._summary_covers_up_to = cutoff
        except Exception as exc:
            logger.warning("Summarization LLM call failed, using extractive fallback: %s", exc)
            self._apply_extractive_summary(to_summarize, cutoff)

    def _update_summary_sync(self) -> None:
        """Synchronous extractive fallback used when no event loop is available."""
        cutoff = len(self._messages) - self.window_size
        if cutoff <= 0 or cutoff <= self._summary_covers_up_to:
            return
        self._apply_extractive_summary(self._messages[:cutoff], cutoff)

    def _apply_extractive_summary(self, messages: list[AgentMessage], cutoff: int) -> None:
        parts = [f"[{m.agent_name}]: {m.content[:100]}" for m in messages]
        self._summary = "\n".join(parts)
        self._summary_covers_up_to = cutoff

    def clear(self) -> None:
        self._messages = []
        self._summary = None
        self._summary_covers_up_to = 0
