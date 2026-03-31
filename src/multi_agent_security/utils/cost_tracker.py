from datetime import datetime, timezone

from pydantic import BaseModel

# Per-token pricing (input/output) keyed by model name.
# Rates are in USD per token (i.e. listed $/M divided by 1_000_000).
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-haiku-3-5-20241022": {"input": 0.80 / 1_000_000, "output": 4.0 / 1_000_000},
    # Alias for the model ID used in config/default.yaml
    "claude-3-5-sonnet-20241022": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    # Bedrock cross-region inference profiles use a different ID format;
    # add entries here as needed.
    "default": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
}


class CostRecord(BaseModel):
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: datetime


class AgentCostSummary(BaseModel):
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    total_latency_ms: float
    avg_latency_ms: float


class RunCostSummary(BaseModel):
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    total_latency_ms: float
    per_agent: dict[str, AgentCostSummary]


class CostTracker:
    """Accumulates token counts and costs across a full pipeline run."""

    def __init__(self):
        self._records: list[CostRecord] = []

    def record(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> CostRecord:
        """Record a single LLM call. Cost is computed from PRICING table."""
        pricing = PRICING.get(model, PRICING["default"])
        cost_usd = (
            input_tokens * pricing["input"] + output_tokens * pricing["output"]
        )
        rec = CostRecord(
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc),
        )
        self._records.append(rec)
        return rec

    def get_records(self) -> list[CostRecord]:
        return list(self._records)

    def get_per_agent_summary(self) -> dict[str, AgentCostSummary]:
        buckets: dict[str, list[CostRecord]] = {}
        for rec in self._records:
            buckets.setdefault(rec.agent_name, []).append(rec)

        result: dict[str, AgentCostSummary] = {}
        for agent, recs in buckets.items():
            calls = len(recs)
            total_lat = sum(r.latency_ms for r in recs)
            result[agent] = AgentCostSummary(
                calls=calls,
                input_tokens=sum(r.input_tokens for r in recs),
                output_tokens=sum(r.output_tokens for r in recs),
                cost_usd=sum(r.cost_usd for r in recs),
                total_latency_ms=total_lat,
                avg_latency_ms=total_lat / calls,
            )
        return result

    def get_total_summary(self) -> RunCostSummary:
        per_agent = self.get_per_agent_summary()
        return RunCostSummary(
            total_calls=len(self._records),
            total_input_tokens=sum(r.input_tokens for r in self._records),
            total_output_tokens=sum(r.output_tokens for r in self._records),
            total_cost_usd=sum(r.cost_usd for r in self._records),
            total_latency_ms=sum(r.latency_ms for r in self._records),
            per_agent=per_agent,
        )

    def get_summary(self) -> dict:
        """Backward-compatible wrapper returning a plain dict."""
        return self.get_total_summary().model_dump()

    def reset(self) -> None:
        self._records = []
