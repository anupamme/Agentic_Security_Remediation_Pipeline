from abc import ABC, abstractmethod
from typing import Optional

from src.types import AgentMessage


class BaseMemory(ABC):
    @abstractmethod
    def store(self, message: AgentMessage) -> None:
        """Store a new agent message."""
        ...

    @abstractmethod
    def retrieve(
        self, agent_name: str, query: Optional[str] = None
    ) -> list[AgentMessage]:
        """Retrieve relevant context for the given agent."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Reset memory for a new task."""
        ...
