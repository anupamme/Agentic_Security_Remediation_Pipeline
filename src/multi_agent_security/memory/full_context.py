from typing import Optional

from multi_agent_security.memory.base import BaseMemory
from multi_agent_security.types import AgentMessage


class FullContextMemory(BaseMemory):
    """Simplest memory strategy: every agent receives the complete message history."""

    def __init__(self):
        self._messages: list[AgentMessage] = []

    def store(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def retrieve(self, agent_name: str, query: Optional[str] = None) -> list[AgentMessage]:
        """Return ALL messages. No filtering."""
        return list(self._messages)

    def clear(self) -> None:
        self._messages = []
