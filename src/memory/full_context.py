from typing import Optional

from src.memory.base import BaseMemory
from src.types import AgentMessage


class FullContextMemory(BaseMemory):
    def store(self, message: AgentMessage) -> None:
        raise NotImplementedError

    def retrieve(self, agent_name: str, query: Optional[str] = None) -> list[AgentMessage]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError
