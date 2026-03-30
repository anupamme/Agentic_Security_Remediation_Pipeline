from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.types import AgentMessage


class TriagerAgent(BaseAgent):
    name = "triager"

    async def run(
        self, input_data: BaseModel, context: list[AgentMessage]
    ) -> tuple[BaseModel, AgentMessage]:
        raise NotImplementedError
