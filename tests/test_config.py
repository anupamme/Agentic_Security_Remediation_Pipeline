import pathlib
import shutil

import pytest
from pydantic import ValidationError

from multi_agent_security.config import AppConfig, LLMConfig, load_config

CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "default.yaml"


def test_load_default_config():
    config = load_config(str(CONFIG_PATH))
    assert isinstance(config, AppConfig)
    assert config.project.name == "multi-agent-security"
    assert config.llm.provider == "anthropic"
    assert config.architecture == "sequential"
    assert config.memory.strategy == "full_context"
    assert config.agents.scanner.enabled is True
    assert config.agents.patcher.max_revision_loops == 3
    assert config.orchestrator.type == "rule_based"
    assert config.eval.num_runs == 3
    assert config.logging.level == "INFO"


def test_config_is_frozen():
    config = load_config(str(CONFIG_PATH))
    with pytest.raises(Exception):
        config.architecture = "blackboard"  # type: ignore[misc]


def test_env_var_override_llm_model(monkeypatch, tmp_path):
    shutil.copy(CONFIG_PATH, tmp_path / "default.yaml")
    monkeypatch.setenv("MASR_LLM_MODEL", "claude-opus-4-20250514")
    config = load_config(str(tmp_path / "default.yaml"))
    assert config.llm.model == "claude-opus-4-20250514"


def test_env_var_override_architecture(monkeypatch, tmp_path):
    shutil.copy(CONFIG_PATH, tmp_path / "default.yaml")
    monkeypatch.setenv("MASR_ARCHITECTURE", "hub_spoke")
    config = load_config(str(tmp_path / "default.yaml"))
    assert config.architecture == "hub_spoke"


def test_env_var_override_memory_strategy(monkeypatch, tmp_path):
    shutil.copy(CONFIG_PATH, tmp_path / "default.yaml")
    monkeypatch.setenv("MASR_MEMORY_STRATEGY", "sliding_window")
    config = load_config(str(tmp_path / "default.yaml"))
    assert config.memory.strategy == "sliding_window"


def test_env_var_override_log_level(monkeypatch, tmp_path):
    shutil.copy(CONFIG_PATH, tmp_path / "default.yaml")
    monkeypatch.setenv("MASR_LOG_LEVEL", "DEBUG")
    config = load_config(str(tmp_path / "default.yaml"))
    assert config.logging.level == "DEBUG"


# --- Bedrock config ---

def test_llm_config_bedrock_valid():
    cfg = LLMConfig(
        provider="bedrock",
        model="anthropic.claude-sonnet-4-20250514-v1:0",
        aws_region="us-east-1",
    )
    assert cfg.provider == "bedrock"
    assert cfg.aws_region == "us-east-1"


def test_llm_config_bedrock_requires_region():
    with pytest.raises(ValidationError, match="aws_region"):
        LLMConfig(provider="bedrock", model="anthropic.claude-sonnet-4-20250514-v1:0")


def test_llm_config_bedrock_optional_profile():
    cfg = LLMConfig(
        provider="bedrock",
        model="anthropic.claude-sonnet-4-20250514-v1:0",
        aws_region="eu-west-1",
        aws_profile="my-profile",
    )
    assert cfg.aws_profile == "my-profile"


def test_env_var_override_aws_region(monkeypatch, tmp_path):
    shutil.copy(CONFIG_PATH, tmp_path / "default.yaml")
    monkeypatch.setenv("MASR_LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("MASR_AWS_REGION", "us-west-2")
    monkeypatch.setenv("MASR_LLM_MODEL", "anthropic.claude-sonnet-4-20250514-v1:0")
    config = load_config(str(tmp_path / "default.yaml"))
    assert config.llm.provider == "bedrock"
    assert config.llm.aws_region == "us-west-2"
