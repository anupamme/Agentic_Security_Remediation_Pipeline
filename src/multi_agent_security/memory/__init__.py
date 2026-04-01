from typing import Optional

from multi_agent_security.config import AppConfig
from multi_agent_security.memory.full_context import FullContextMemory
from multi_agent_security.memory.retrieval import RetrievalMemory
from multi_agent_security.memory.sliding_window import SlidingWindowMemory


def create_memory(config: AppConfig, llm_client=None):
    """Factory: return the correct memory strategy from config."""
    strategy = config.memory.strategy
    if strategy == "full_context":
        return FullContextMemory()
    elif strategy == "sliding_window":
        return SlidingWindowMemory(
            window_size=config.memory.sliding_window_size,
            summary_model=config.memory.summary_model,
            llm_client=llm_client,
        )
    elif strategy == "retrieval":
        return RetrievalMemory(
            top_k=config.memory.retrieval_top_k,
            embedding_model=config.memory.embedding_model,
            embedding_provider=config.memory.embedding_provider,
        )
    else:
        raise ValueError(f"Unknown memory strategy: {strategy}")


__all__ = [
    "create_memory",
    "FullContextMemory",
    "SlidingWindowMemory",
    "RetrievalMemory",
]
