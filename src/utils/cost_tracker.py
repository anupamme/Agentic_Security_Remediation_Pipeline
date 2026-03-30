from collections import defaultdict

from src.types import AgentMessage


class CostTracker:
    """Accumulates token counts and costs across a full pipeline run."""

    def __init__(self):
        self.reset()

    def record(self, message: AgentMessage) -> None:
        self._total_input_tokens += message.token_count_input
        self._total_output_tokens += message.token_count_output
        self._total_cost_usd += message.cost_usd
        self._total_latency_ms += message.latency_ms
        agent = message.agent_name
        self._per_agent[agent]["input_tokens"] += message.token_count_input
        self._per_agent[agent]["output_tokens"] += message.token_count_output
        self._per_agent[agent]["cost_usd"] += message.cost_usd
        self._per_agent[agent]["latency_ms"] += message.latency_ms
        self._per_agent[agent]["calls"] += 1

    def get_summary(self) -> dict:
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": round(self._total_cost_usd, 6),
            "total_latency_ms": round(self._total_latency_ms, 2),
            "per_agent": {k: dict(v) for k, v in self._per_agent.items()},
        }

    def reset(self) -> None:
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._total_latency_ms: float = 0.0
        self._per_agent: dict = defaultdict(
            lambda: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": 0.0,
                "calls": 0,
            }
        )
