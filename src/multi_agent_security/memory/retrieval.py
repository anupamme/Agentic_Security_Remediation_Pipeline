from typing import Optional

from multi_agent_security.memory.base import BaseMemory
from multi_agent_security.types import AgentMessage


class RetrievalMemory(BaseMemory):
    def store(self, message: AgentMessage) -> None:
        raise NotImplementedError

    def retrieve(self, agent_name: str, query: Optional[str] = None) -> list[AgentMessage]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError
