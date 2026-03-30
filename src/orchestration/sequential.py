from src.orchestration.base import BaseOrchestrator
from src.types import TaskState


class SequentialOrchestrator(BaseOrchestrator):
    async def run(self, repo_path: str, task_state: TaskState) -> TaskState:
        raise NotImplementedError
