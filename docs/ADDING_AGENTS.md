# Adding a New Agent

This guide walks through implementing a custom agent and integrating it into all three
orchestration architectures.

---

## Step 1: Define Input and Output Types

All agent I/O is typed via Pydantic models. Add your types to
`src/multi_agent_security/types.py` or a dedicated module:

```python
from pydantic import BaseModel
from multi_agent_security.types import VulnType, VulnSeverity

class SanitizerInput(BaseModel):
    """Input to the Sanitizer agent."""
    file_path: str
    code: str
    vuln_type: VulnType

class SanitizerOutput(BaseModel):
    """Output from the Sanitizer agent."""
    sanitized_code: str
    changes_made: list[str]
    confidence: float
```

---

## Step 2: Implement the Agent

Create `src/multi_agent_security/agents/sanitizer.py`. Extend `BaseAgent`:

```python
from multi_agent_security.agents.base import BaseAgent
from multi_agent_security.llm_client import LLMClient
from multi_agent_security.config import AppConfig
from .sanitizer_types import SanitizerInput, SanitizerOutput


class SanitizerAgent(BaseAgent):
    """Applies input sanitization as a post-processing step after patching."""

    def __init__(self, config: AppConfig, llm_client: LLMClient) -> None:
        self.config = config
        self.llm = llm_client

    async def run(self, input_data: SanitizerInput) -> SanitizerOutput:
        prompt = self._build_prompt(input_data)
        response = await self.llm.message(
            model=self.config.llm.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.llm.max_tokens,
        )
        return self._parse_response(response)

    def _build_prompt(self, inp: SanitizerInput) -> str:
        return f"""You are a security sanitization expert.

Review this code for input sanitization opportunities:

File: {inp.file_path}
Vulnerability type: {inp.vuln_type}

```python
{inp.code}
```

Return JSON:
{{
  "sanitized_code": "...",
  "changes_made": ["...", "..."],
  "confidence": 0.0
}}"""

    def _parse_response(self, response: str) -> SanitizerOutput:
        import json
        data = json.loads(response)
        return SanitizerOutput(**data)
```

See `src/multi_agent_security/agents/base.py` for the `BaseAgent` interface.
See `src/multi_agent_security/agents/patcher.py` for a full production example.

---

## Step 3: Add Configuration

Add an `agents.sanitizer` section to `config/default.yaml`:

```yaml
agents:
  # ... existing agents ...
  sanitizer:
    enabled: true
    confidence_threshold: 0.7
```

Extend `AgentConfig` in `src/multi_agent_security/config.py`:

```python
class SanitizerAgentConfig(BaseModel):
    enabled: bool = True
    confidence_threshold: float = 0.7

class AgentsConfig(BaseModel):
    scanner: ScannerAgentConfig = ScannerAgentConfig()
    triager: TriagerAgentConfig = TriagerAgentConfig()
    patcher: PatcherAgentConfig = PatcherAgentConfig()
    reviewer: ReviewerAgentConfig = ReviewerAgentConfig()
    sanitizer: SanitizerAgentConfig = SanitizerAgentConfig()  # ← add this
```

---

## Step 4: Wire into the Sequential Orchestrator

Edit `src/multi_agent_security/orchestration/sequential.py` to call your agent
after the existing Reviewer step:

```python
from multi_agent_security.agents.sanitizer import SanitizerAgent
from multi_agent_security.agents.sanitizer_types import SanitizerInput

class SequentialOrchestrator(BaseOrchestrator):
    def __init__(self, config: AppConfig, llm_client: LLMClient) -> None:
        # ... existing agents ...
        if config.agents.sanitizer.enabled:
            self.sanitizer = SanitizerAgent(config, llm_client)

    async def run(self, repo_path: str) -> PipelineResult:
        # ... existing pipeline steps ...

        # After reviewer accepts the patch:
        if review_result.accepted and config.agents.sanitizer.enabled:
            sanitizer_input = SanitizerInput(
                file_path=patch_result.file_path,
                code=patch_result.patched_code,
                vuln_type=triage_result.vuln_type,
            )
            sanitizer_output = await self.sanitizer.run(sanitizer_input)
```

---

## Step 5: Wire into Hub-and-Spoke and Blackboard

**Hub-and-Spoke** (`src/multi_agent_security/orchestration/hub_spoke.py`):
Add the Sanitizer as a spoke that the Hub dispatches after the Reviewer:

```python
if config.agents.sanitizer.enabled:
    self.spokes["sanitizer"] = SanitizerAgent(config, llm_client)

# In the Hub dispatch loop:
sanitizer_result = await self.spokes["sanitizer"].run(sanitizer_input)
```

**Blackboard** (`src/multi_agent_security/orchestration/blackboard.py`):
Post a `sanitize_needed` task to the blackboard after the Reviewer accepts a patch,
and have the Sanitizer agent poll for it:

```python
# After Reviewer posts acceptance:
blackboard.post("sanitize_needed", {"patch": patch_result, "vuln_type": ...})

# In SanitizerAgent.poll():
if task := blackboard.get("sanitize_needed"):
    result = await self.run(SanitizerInput(**task))
    blackboard.post("sanitize_done", result.model_dump())
```

---

## Step 6: Add Tests

Create `tests/test_sanitizer.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from multi_agent_security.agents.sanitizer import SanitizerAgent

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.llm.model = "claude-sonnet-4-20250514"
    cfg.llm.max_tokens = 4096
    return cfg

@pytest.mark.asyncio
async def test_sanitizer_returns_output(mock_config):
    llm = AsyncMock()
    llm.message.return_value = '{"sanitized_code": "safe_code()", "changes_made": ["added validation"], "confidence": 0.9}'

    agent = SanitizerAgent(mock_config, llm)
    result = await agent.run(SanitizerInput(
        file_path="app.py",
        code="unsafe_input()",
        vuln_type="CWE-89",
    ))

    assert result.confidence == 0.9
    assert len(result.changes_made) == 1
```

---

## Step 7: Verify

```bash
# Run tests
pytest tests/test_sanitizer.py -v

# Run ruff
ruff check src/multi_agent_security/agents/sanitizer.py

# Smoke test through the pipeline
MASR_ARCHITECTURE=sequential python scripts/run_pipeline.py \
    --repo data/raw/clones/example-repo/
```
