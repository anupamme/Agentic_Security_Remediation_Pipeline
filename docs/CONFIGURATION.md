# Configuration Reference

All configuration lives in YAML files under `config/`. Values can be overridden at runtime
via `MASR_*` environment variables (see below).

---

## Quick Start

```bash
# Run with defaults (Sequential + Full Context)
python scripts/run_pipeline.py --repo /path/to/repo

# Switch to hub-and-spoke + retrieval memory
MASR_ARCHITECTURE=hub_spoke MASR_MEMORY_STRATEGY=retrieval \
    python scripts/run_pipeline.py --repo /path/to/repo

# Use AWS Bedrock instead of Anthropic
python scripts/run_pipeline.py --repo /path/to/repo --config config/arch_sequential_bedrock.yaml
```

---

## Configuration Files

| File | Description |
|------|-------------|
| `config/default.yaml` | Base configuration; all other files extend this |
| `config/arch_sequential.yaml` | Sequential pipeline orchestrator |
| `config/arch_hub_spoke.yaml` | Hub-and-spoke orchestrator |
| `config/arch_blackboard.yaml` | Blackboard orchestrator |
| `config/arch_sequential_bedrock.yaml` | Sequential + AWS Bedrock LLM |
| `config/arch_hub_spoke_bedrock.yaml` | Hub-and-spoke + AWS Bedrock LLM |
| `config/arch_blackboard_bedrock.yaml` | Blackboard + AWS Bedrock LLM |

---

## Full Reference: `config/default.yaml`

```yaml
project:
  name: "multi-agent-security"
  version: "0.1.0"

llm:
  provider: "anthropic"              # "anthropic" | "bedrock"
  model: "claude-sonnet-4-20250514"  # Model ID passed to the provider
  max_tokens: 4096                   # Max output tokens per LLM call
  temperature: 0.0                   # Sampling temperature (0 = deterministic)
  api_key_env: "ANTHROPIC_API_KEY"   # Env var name for Anthropic API key

architecture: "sequential"           # "sequential" | "hub_spoke" | "blackboard"

memory:
  strategy: "full_context"           # "full_context" | "sliding_window" | "retrieval"
  sliding_window_size: 10            # Number of recent messages to keep (sliding_window only)
  summary_model: "claude-sonnet-4-20250514"  # Model used to generate summaries
  retrieval_top_k: 5                 # Number of messages to retrieve (retrieval only)
  embedding_model: "voyage-3"        # Embedding model for retrieval similarity
  embedding_provider: "anthropic"    # "anthropic" | "bedrock" (uses Titan for bedrock)

agents:
  scanner:
    enabled: true
    max_files_per_call: 5            # Files batched into a single Scanner LLM call
    use_static_analysis: true        # Run Semgrep before LLM analysis
    static_analysis_tools:
      - "semgrep"
  triager:
    enabled: true
  patcher:
    enabled: true
    max_revision_loops: 3            # Max patch revision iterations
    run_tests: true                  # Run test suite after each patch attempt
  reviewer:
    enabled: true

orchestrator:
  type: "rule_based"                 # "rule_based" | "llm_based"
  max_retries: 2                     # Retries on agent call failure
  timeout_seconds: 300               # Per-vulnerability timeout

eval:
  judge_model: "claude-sonnet-4-20250514"
  num_runs: 3                        # Repeated runs per config for variance estimation
  parallel_workers: 4                # Concurrent pipeline instances during benchmarking

logging:
  level: "INFO"                      # "DEBUG" | "INFO" | "WARNING" | "ERROR"
  structured: true                   # Emit JSON log lines (for log aggregation)
  output_file: "data/results/run.jsonl"
```

---

## Environment Variable Overrides

All YAML values can be overridden via `MASR_*` environment variables. The variable name
maps to the YAML path using `_` as a separator.

| Environment Variable | YAML Path | Example |
|---------------------|-----------|---------|
| `MASR_LLM_MODEL` | `llm.model` | `claude-sonnet-4-20250514` |
| `MASR_LLM_PROVIDER` | `llm.provider` | `bedrock` |
| `MASR_ARCHITECTURE` | `architecture` | `hub_spoke` |
| `MASR_MEMORY_STRATEGY` | `memory.strategy` | `retrieval` |
| `MASR_LOG_LEVEL` | `logging.level` | `DEBUG` |
| `MASR_AWS_REGION` | `llm.aws_region` | `us-east-1` |
| `MASR_AWS_PROFILE` | `llm.aws_profile` | `my-profile` |
| `ANTHROPIC_API_KEY` | *(read by LLMClient)* | `sk-ant-...` |
| `AWS_ACCESS_KEY_ID` | *(read by boto3)* | |
| `AWS_SECRET_ACCESS_KEY` | *(read by boto3)* | |
| `AWS_REGION` | *(read by boto3)* | `us-east-1` |

---

## AWS Bedrock Configuration

To use AWS Bedrock as the LLM provider:

```bash
# Option 1: Use a Bedrock config file directly
python scripts/run_pipeline.py --config config/arch_sequential_bedrock.yaml --repo /path/to/repo

# Option 2: Override via env var
MASR_LLM_PROVIDER=bedrock MASR_AWS_REGION=us-east-1 \
    python scripts/run_pipeline.py --repo /path/to/repo
```

AWS credentials are read from the standard boto3 credential chain:
environment variables → `~/.aws/credentials` → IAM instance role.

---

## Programmatic Configuration

```python
from multi_agent_security.config import load_config

# Load from a file
config = load_config("config/arch_hub_spoke.yaml")

# Access fields
print(config.llm.model)           # "claude-sonnet-4-20250514"
print(config.architecture)        # "hub_spoke"
print(config.memory.strategy)     # "full_context"
print(config.agents.scanner.enabled)  # True
```

The `load_config()` function applies `MASR_*` environment variable overrides automatically
after loading the YAML file.
