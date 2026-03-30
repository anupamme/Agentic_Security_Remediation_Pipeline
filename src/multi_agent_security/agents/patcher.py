from pydantic import BaseModel

from multi_agent_security.agents.base import BaseAgent
from multi_agent_security.types import AgentMessage


class PatcherAgent(BaseAgent):
    name = "patcher"

    async def run(
        self, input_data: BaseModel, context: list[AgentMessage]
    ) -> tuple[BaseModel, AgentMessage]:
        raise NotImplementedError
