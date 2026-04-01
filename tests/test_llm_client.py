import pytest
from pydantic import BaseModel

from multi_agent_security.llm_client import LLMClient, LLMResponse, _compute_cost
from multi_agent_security.config import LLMConfig


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig()


@pytest.fixture
def dry_client(llm_config) -> LLMClient:
    return LLMClient(config=llm_config, dry_run=True)


async def test_dry_run_returns_llm_response(dry_client):
    response = await dry_client.complete(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say hello.",
    )
    assert isinstance(response, LLMResponse)
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms == 0.0
    assert response.cost_usd >= 0.0
    assert response.model == LLMConfig().model


async def test_dry_run_without_response_format(dry_client):
    response = await dry_client.complete(
        system_prompt="sys",
        user_prompt="user",
    )
    assert "mock" in response.content.lower() or len(response.content) > 0


async def test_dry_run_with_response_format(dry_client):
    class SampleOutput(BaseModel):
        result: str

    # dry_run mode returns mock JSON content; we just confirm LLMResponse is returned
    response = await dry_client.complete(
        system_prompt="Extract result.",
        user_prompt="The result is success.",
        response_format=SampleOutput,
    )
    assert isinstance(response, LLMResponse)
    assert response.content is not None


def test_llm_client_dry_run_flag(llm_config):
    client = LLMClient(config=llm_config, dry_run=True)
    assert client.dry_run is True


def test_llm_client_has_no_anthropic_client_in_dry_run(llm_config):
    client = LLMClient(config=llm_config, dry_run=True)
    assert not hasattr(client, "_client")


# --- Bedrock ---

@pytest.fixture
def bedrock_config() -> LLMConfig:
    return LLMConfig(
        provider="bedrock",
        model="anthropic.claude-sonnet-4-20250514-v1:0",
        aws_region="us-east-1",
    )


@pytest.fixture
def dry_bedrock_client(bedrock_config) -> LLMClient:
    return LLMClient(config=bedrock_config, dry_run=True)


async def test_bedrock_dry_run_returns_llm_response(dry_bedrock_client, bedrock_config):
    response = await dry_bedrock_client.complete(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say hello.",
    )
    assert isinstance(response, LLMResponse)
    assert response.model == bedrock_config.model
    assert response.input_tokens > 0
    assert response.cost_usd >= 0.0


def test_bedrock_dry_run_no_client_instantiated(bedrock_config):
    client = LLMClient(config=bedrock_config, dry_run=True)
    assert not hasattr(client, "_client")


def test_compute_cost_bedrock_same_as_anthropic():
    assert _compute_cost(1_000_000, 0, "bedrock") == _compute_cost(1_000_000, 0, "anthropic")
    assert _compute_cost(0, 1_000_000, "bedrock") == _compute_cost(0, 1_000_000, "anthropic")
