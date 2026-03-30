from abc import ABC, abstractmethod

from pydantic import BaseModel

from src.config import AppConfig
from src.llm_client import LLMClient
from src.types import AgentMessage


class BaseAgent(ABC):
    def __init__(self, config: AppConfig, llm_client: LLMClient):
        self.config = config
        self.llm = llm_client
        self.name: str = self.__class__.__name__

    @abstractmethod
    async def run(
        self, input_data: BaseModel, context: list[AgentMessage]
    ) -> tuple[BaseModel, AgentMessage]:
        """Execute the agent's task. Return (output, message_log_entry)."""
        ...

    def build_prompt(
        self, input_data: BaseModel, context: list[AgentMessage]
    ) -> tuple[str, str]:
        """Build system and user prompts. Subclasses override this."""
        raise NotImplementedError
