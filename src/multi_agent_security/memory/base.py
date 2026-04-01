from abc import ABC, abstractmethod
from typing import Optional

from multi_agent_security.types import AgentMessage


class BaseMemory(ABC):
    @abstractmethod
    async def store(self, message: AgentMessage) -> None:
        """Store a new agent message."""
        ...

    @abstractmethod
    async def retrieve(
        self, agent_name: str, query: Optional[str] = None
    ) -> list[AgentMessage]:
        """Retrieve relevant context for the given agent."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Reset memory for a new task."""
        ...
