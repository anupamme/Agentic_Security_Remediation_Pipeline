# Multi-Agent Security Remediation Pipeline

A research system for evaluating multi-agent architectures on automated vulnerability detection and remediation tasks.

## Overview

The pipeline uses four specialized agents (Scanner, Triager, Patcher, Reviewer) that collaborate to detect vulnerabilities in code repositories and produce reviewed, tested patches. Three orchestration architectures (Sequential, Hub-and-Spoke, Blackboard) and three memory strategies (Full Context, Sliding Window, Retrieval) are compared on a curated benchmark dataset.

## Setup

**Requirements:** Python 3.11+

```bash
# Install in editable mode with all dependencies
pip install -e .

# For development tools (pytest, ruff)
pip install -e ".[dev]"
```

Copy `.env.example` and fill in your API key:

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=<your key>
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
src/
  types.py            # Shared Pydantic data models
  config.py           # YAML config loading with env var overrides
  llm_client.py       # Unified LLM client (Anthropic) with cost tracking
  agents/             # Scanner, Triager, Patcher, Reviewer
  orchestration/      # Sequential, Hub-Spoke, Blackboard orchestrators
  memory/             # Full-context, Sliding-window, Retrieval memory
  tools/              # File reader, AST parser, static analysis, test runner
  eval/               # Metrics, LLM judge, eval harness
  utils/              # Structured logging, cost tracker
config/               # YAML configuration files
data/benchmark/       # Dev / test / hard split benchmark examples
scripts/              # Dataset curation and eval entry points
tests/                # Unit tests
```

## Configuration

The system is controlled entirely by YAML config. Switch architectures or memory strategies:

```bash
# Via env var override (no file change needed)
MASR_ARCHITECTURE=hub_spoke MASR_MEMORY_STRATEGY=sliding_window pytest tests/
```

See `config/default.yaml` for all available settings.
