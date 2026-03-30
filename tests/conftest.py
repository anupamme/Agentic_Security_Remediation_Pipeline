import pytest
from datetime import datetime, timezone

from multi_agent_security.config import AppConfig, LLMConfig
from multi_agent_security.llm_client import LLMClient
from multi_agent_security.types import (
    AgentMessage,
    FixStrategy,
    TaskState,
    TriageResult,
    Vulnerability,
    VulnSeverity,
    VulnType,
)


@pytest.fixture
def sample_vulnerability() -> Vulnerability:
    return Vulnerability(
        id="VULN-001",
        file_path="app/views.py",
        line_start=42,
        line_end=44,
        vuln_type=VulnType.SQL_INJECTION,
        description="Unsanitised input passed to raw SQL query",
        confidence=0.95,
        code_snippet='cursor.execute(f"SELECT * FROM users WHERE id={user_id}")',
        scanner_reasoning="f-string interpolation in SQL context",
    )


@pytest.fixture
def sample_task_state(sample_vulnerability: Vulnerability) -> TaskState:
    return TaskState(
        task_id="task-001",
        repo_url="https://github.com/example/repo",
        vulnerabilities=[sample_vulnerability],
    )


@pytest.fixture
def app_config(tmp_path) -> AppConfig:
    config_path = tmp_path / "default.yaml"
    import shutil, pathlib
    shutil.copy(
        pathlib.Path(__file__).parent.parent / "config" / "default.yaml",
        config_path,
    )
    from multi_agent_security.config import load_config
    return load_config(str(config_path))


@pytest.fixture
def dry_run_llm_client(app_config: AppConfig) -> LLMClient:
    return LLMClient(config=app_config.llm, dry_run=True)


@pytest.fixture
def sample_agent_message() -> AgentMessage:
    return AgentMessage(
        agent_name="scanner",
        timestamp=datetime.now(timezone.utc),
        content='{"result": "test"}',
        token_count_input=100,
        token_count_output=50,
        latency_ms=250.0,
        cost_usd=0.001,
    )
