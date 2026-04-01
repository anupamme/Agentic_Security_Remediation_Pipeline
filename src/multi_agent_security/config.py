import os
import yaml
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class ProjectConfig(BaseModel):
    name: str = "multi-agent-security"
    version: str = "0.1.0"


class LLMConfig(BaseModel):
    provider: str = "anthropic"  # "anthropic" | "bedrock"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.0
    api_key_env: str = "ANTHROPIC_API_KEY"
    # Bedrock-specific (ignored when provider="anthropic")
    aws_region: Optional[str] = None
    aws_profile: Optional[str] = None

    @model_validator(mode="after")
    def _check_bedrock_fields(self) -> "LLMConfig":
        if self.provider == "bedrock" and not self.aws_region:
            raise ValueError("aws_region is required when provider='bedrock'")
        return self


class MemoryConfig(BaseModel):
    strategy: str = "full_context"
    sliding_window_size: int = 10
    summary_model: str = "claude-sonnet-4-20250514"
    retrieval_top_k: int = 5
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    embedding_provider: str = "local"  # "local" | "api" | "bedrock"
    max_context_tokens: int = 4000  # Max tokens per agent from blackboard (Architecture C)


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
    routing_model: str = "claude-sonnet-4-20250514"
    max_summary_tokens: int = 500


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
        "MASR_AWS_REGION": ("llm", "aws_region"),
        "MASR_AWS_PROFILE": ("llm", "aws_profile"),
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


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. override wins on conflicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> AppConfig:
    """Load and validate configuration from a YAML file.

    If the YAML contains a ``base:`` key, the referenced file is loaded first
    and the current file's values are deep-merged on top (current wins).
    The ``base:`` path is resolved relative to the directory of ``path``.

    Environment variables prefixed with MASR_ override all YAML values.
    Returns a frozen, validated AppConfig.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    base_path = raw.pop("base", None)
    if base_path:
        base_dir = os.path.dirname(os.path.abspath(path))
        full_base_path = os.path.join(base_dir, base_path)
        with open(full_base_path) as f:
            base_raw = yaml.safe_load(f) or {}
        raw = _deep_merge(base_raw, raw)

    raw = _apply_env_overrides(raw)
    return AppConfig.model_validate(raw)
