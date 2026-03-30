from abc import ABC, abstractmethod

from src.config import AppConfig
from src.types import TaskState


class BaseOrchestrator(ABC):
    def __init__(self, config: AppConfig, agents: dict, memory):
        self.config = config
        self.agents = agents
        self.memory = memory

    @abstractmethod
    async def run(self, repo_path: str, task_state: TaskState) -> TaskState:
        """Run the full pipeline on a repo. Return updated task state."""
        ...
