from pydantic import BaseModel

from multi_agent_security.agents.base import BaseAgent
from multi_agent_security.types import AgentMessage


class TriagerAgent(BaseAgent):
    name = "triager"

    async def run(
        self, input_data: BaseModel, context: list[AgentMessage]
    ) -> tuple[BaseModel, AgentMessage]:
        raise NotImplementedError
