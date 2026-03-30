from multi_agent_security.orchestration.base import BaseOrchestrator
from multi_agent_security.types import TaskState


class SequentialOrchestrator(BaseOrchestrator):
    async def run(self, repo_path: str, task_state: TaskState) -> TaskState:
        raise NotImplementedError
