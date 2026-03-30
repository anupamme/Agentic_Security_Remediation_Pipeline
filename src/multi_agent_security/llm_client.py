import json
import os
import time
from typing import Optional

import anthropic
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_security.config import LLMConfig

# Pricing constants per million tokens.
# Bedrock pricing mirrors Anthropic list price for Claude Sonnet;
# update per https://aws.amazon.com/bedrock/pricing/ if needed.
_PRICING: dict[str, dict[str, float]] = {
    "anthropic": {"input": 3.0, "output": 15.0},
    "bedrock": {"input": 3.0, "output": 15.0},
}

_DRY_RUN_CONTENT = '{"result": "dry_run_mock_response"}'


class LLMResponse(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    model: str


class LLMClient:
    def __init__(self, config: LLMConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        if not dry_run:
            if config.provider == "bedrock":
                self._client = anthropic.AsyncAnthropicBedrock(
                    aws_region=config.aws_region,
                    aws_profile=config.aws_profile,
                )
            else:
                api_key = os.environ.get(config.api_key_env)
                self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[type[BaseModel]] = None,
    ) -> LLMResponse:
        """Make an LLM call.

        If response_format is provided, appends JSON schema instructions to
        the system prompt, parses the response into the Pydantic model, and
        retries once on parse failure.
        """
        if response_format is not None:
            schema = json.dumps(response_format.model_json_schema(), indent=2)
            system_prompt = (
                f"{system_prompt}\n\n"
                f"Respond with valid JSON that matches this schema:\n{schema}\n"
                "Output only the JSON object, no additional text."
            )

        if self.dry_run:
            return self._mock_response(system_prompt, user_prompt, response_format)

        return await self._call_with_retry(system_prompt, user_prompt, response_format)

    def _mock_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[type[BaseModel]],
    ) -> LLMResponse:
        if response_format is not None:
            # Return minimal valid JSON for the schema
            content = _DRY_RUN_CONTENT
        else:
            content = "This is a dry-run mock response."
        input_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)
        output_tokens = max(1, len(content) // 4)
        cost = _compute_cost(input_tokens, output_tokens, self.config.provider)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=0.0,
            cost_usd=cost,
            model=self.config.model,
        )

    async def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[type[BaseModel]],
    ) -> LLMResponse:
        response = await self._raw_call(system_prompt, user_prompt)

        if response_format is not None:
            try:
                response_format.model_validate_json(response.content)
            except Exception:
                # Retry once
                response = await self._raw_call(system_prompt, user_prompt)
                response_format.model_validate_json(response.content)

        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _raw_call(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        start = time.monotonic()
        message = await self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        latency_ms = (time.monotonic() - start) * 1000

        content = message.content[0].text if message.content else ""
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        cost = _compute_cost(input_tokens, output_tokens, self.config.provider)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            model=self.config.model,
        )


def _compute_cost(input_tokens: int, output_tokens: int, provider: str = "anthropic") -> float:
    pricing = _PRICING.get(provider, _PRICING["anthropic"])
    return (input_tokens / 1_000_000) * pricing["input"] + (
        output_tokens / 1_000_000
    ) * pricing["output"]
