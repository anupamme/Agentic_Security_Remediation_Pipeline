import os
import yaml
from typing import Optional
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "multi-agent-security"
    version: str = "0.1.0"


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.0
    api_key_env: str = "ANTHROPIC_API_KEY"


class MemoryConfig(BaseModel):
    strategy: str = "full_context"
    sliding_window_size: int = 10
    summary_model: str = "claude-sonnet-4-20250514"
    retrieval_top_k: int = 5
    embedding_model: str = "voyage-3"


class ScannerAgentConfig(BaseModel):
    enabled: bool = True
    max_files_per_call: int = 5
    use_static_analysis: bool = True
    static_analysis_tools: list[str] = Field(default_factory=lambda: ["semgrep"])


class TriagerAgentConfig(BaseModel):
    enabled: bool = True


class PatcherAgentConfig(BaseModel):
    enabled: bool = True
    max_revision_loops: int = 3
    run_tests: bool = True


class ReviewerAgentConfig(BaseModel):
    enabled: bool = True


class AgentsConfig(BaseModel):
    scanner: ScannerAgentConfig = Field(default_factory=ScannerAgentConfig)
    triager: TriagerAgentConfig = Field(default_factory=TriagerAgentConfig)
    patcher: PatcherAgentConfig = Field(default_factory=PatcherAgentConfig)
    reviewer: ReviewerAgentConfig = Field(default_factory=ReviewerAgentConfig)


class OrchestratorConfig(BaseModel):
    type: str = "rule_based"
    max_retries: int = 2
    timeout_seconds: int = 300


class EvalConfig(BaseModel):
    judge_model: str = "claude-sonnet-4-20250514"
    num_runs: int = 3
    parallel_workers: int = 4


class LoggingConfig(BaseModel):
    level: str = "INFO"
    structured: bool = True
    output_file: str = "data/results/run.jsonl"


class AppConfig(BaseModel):
    model_config = {"frozen": True}

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    architecture: str = "sequential"
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _apply_env_overrides(data: dict) -> dict:
    """Apply MASR_* environment variable overrides to the config dict.

    Supported overrides (case-insensitive env var names):
      MASR_LLM_MODEL       -> data["llm"]["model"]
      MASR_LLM_PROVIDER    -> data["llm"]["provider"]
      MASR_ARCHITECTURE    -> data["architecture"]
      MASR_MEMORY_STRATEGY -> data["memory"]["strategy"]
      MASR_LOG_LEVEL       -> data["logging"]["level"]
    """
    mapping = {
        "MASR_LLM_MODEL": ("llm", "model"),
        "MASR_LLM_PROVIDER": ("llm", "provider"),
        "MASR_ARCHITECTURE": ("architecture",),
        "MASR_MEMORY_STRATEGY": ("memory", "strategy"),
        "MASR_LOG_LEVEL": ("logging", "level"),
    }
    for env_var, path in mapping.items():
        value = os.environ.get(env_var)
        if value is None:
            continue
        if len(path) == 1:
            data[path[0]] = value
        else:
            section = data.setdefault(path[0], {})
            section[path[1]] = value
    return data


def load_config(path: str) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Environment variables prefixed with MASR_ override YAML values.
    Returns a frozen, validated AppConfig.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw = _apply_env_overrides(raw)
    return AppConfig.model_validate(raw)
