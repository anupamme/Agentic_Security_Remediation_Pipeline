# Multi-Agent Security Remediation Pipeline

A research system comparing multi-agent architectures for automated security vulnerability
detection and patching, with a rigorous evaluation framework and real-world benchmark dataset.

---

## Key Results

We evaluate **9 configurations** (3 architectures × 3 memory strategies) against a benchmark
of 129 real security-fix pull requests from GitHub. The full results are in
`data/results/` and can be reproduced with `scripts/reproduce.sh`.

See [`docs/writeup.md`](docs/writeup.md) for the full technical report and analysis.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/anupamme/Multi_Agent_Security_Remediation_Pipeline
cd Multi_Agent_Security_Remediation_Pipeline

# 2. Install dependencies (Python 3.11+)
pip install -e ".[dev,viz]"

# 3. Configure your API key
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the pipeline on a single repository
python scripts/run_pipeline.py --repo /path/to/your/repo

# 5. Run the benchmark smoke test (3 configs, dev split)
python scripts/run_full_benchmark.py --split dev --configs C1,C4,C7 --runs 1
```

---

## Architecture

The pipeline coordinates four specialist agents:

| Agent | Role |
|-------|------|
| **Scanner** | Detects vulnerable files using Semgrep + LLM analysis |
| **Triager** | Classifies severity and fix strategy; ranks vulnerabilities |
| **Patcher** | Generates a unified-diff patch; runs tests; revises up to 3× |
| **Reviewer** | Accepts or rejects the patch; provides improvement feedback |

Three communication architectures are supported:

- **Sequential** — Linear pipeline (simplest, lowest overhead)
- **Hub-and-Spoke** — Parallel patch generation for multi-vulnerability repos
- **Blackboard** — Shared state coordination (most flexible)

Three memory strategies control context passing between agents:

- **Full Context** — All prior messages (highest quality, highest cost)
- **Sliding Window + Summary** — Rolling window with LLM summarization
- **Retrieval-Augmented** — Semantic retrieval of relevant context (lowest cost)

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for detailed diagrams and trade-offs.

---

## Benchmark

The benchmark comprises **129 real security-fix PRs** from public GitHub repositories,
annotated with CWE types, complexity tags, and ground-truth diffs.

| Split | Size | Purpose |
|-------|------|---------|
| `dev` | small | Quick smoke tests |
| `test` | 129 | Primary evaluation |
| `hard` | small | Multi-file, complex fixes |

CWE types covered: SSRF, Deserialization, Weak Crypto, Hardcoded Credentials,
Input Validation, OS Command Injection, XSS, Path Traversal, SQL Injection, and more.

See [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for the full dataset description and how to
add new examples.

---

## Reproducing Results

```bash
# Full reproduction (requires ANTHROPIC_API_KEY, ~30–60 min)
bash scripts/reproduce.sh
```

This runs:
1. Dataset validation
2. Dev-set smoke test
3. Full benchmark (all 9 configs × 3 runs × 129 examples)
4. Ablation studies (FULL, A0–A4)
5. Comparison reports and failure analysis
6. Figure generation → `docs/figures/`

---

## Project Structure

```
Multi_Agent_Security_Remediation_Pipeline/
├── src/multi_agent_security/
│   ├── agents/           # Scanner, Triager, Patcher, Reviewer, SingleAgent
│   ├── orchestration/    # Sequential, Hub-Spoke, Blackboard orchestrators
│   ├── memory/           # Full-context, Sliding-window, Retrieval strategies
│   ├── eval/             # Metrics, LLM judge, eval runner, failure analysis
│   ├── tools/            # File reader, code parser, Semgrep, test runner, diff gen
│   ├── utils/            # Structured logging, cost tracker, similarity
│   ├── config.py         # YAML config loading with MASR_* env overrides
│   ├── llm_client.py     # Unified LLM client (Anthropic + AWS Bedrock)
│   └── types.py          # Shared Pydantic data models
├── config/               # YAML configs (default + per-architecture)
├── data/
│   ├── benchmark/        # dev / test / hard / local splits (JSON examples)
│   └── results/          # Benchmark run outputs (gitignored)
├── scripts/
│   ├── run_pipeline.py         # Single-repo pipeline run
│   ├── run_full_benchmark.py   # Full 9-config benchmark
│   ├── run_ablations.py        # Ablation study runner
│   ├── generate_comparison.py  # Comparison table generation
│   ├── generate_failure_report.py  # Failure analysis report
│   ├── generate_figures.py     # Publication figures → docs/figures/
│   ├── reproduce.sh            # End-to-end reproduction script
│   ├── curate_dataset.py       # Benchmark dataset curation
│   └── validate_dataset.py     # Dataset integrity validation
├── tests/                # 364 unit tests across all modules
├── docs/
│   ├── writeup.md        # Full technical writeup (10–15 pages)
│   ├── ARCHITECTURE.md   # Architecture diagrams and trade-offs
│   ├── CONFIGURATION.md  # All config options explained
│   ├── BENCHMARK.md      # Dataset format and curation guide
│   ├── ADDING_AGENTS.md  # How to implement a new agent
│   ├── demo_script.md    # 5-minute demo walkthrough
│   └── figures/          # Generated figures (run generate_figures.py)
├── .env.example          # Environment variable template
├── pyproject.toml        # Python package config and dependencies
└── LICENSE               # MIT License
```

---

## Configuration

Switch architectures and memory strategies without changing code:

```bash
# Hub-and-spoke + retrieval memory
MASR_ARCHITECTURE=hub_spoke MASR_MEMORY_STRATEGY=retrieval \
    python scripts/run_pipeline.py --repo /path/to/repo

# AWS Bedrock instead of Anthropic
python scripts/run_pipeline.py --config config/arch_sequential_bedrock.yaml --repo /path/to/repo
```

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for the full configuration reference.

---

## Development

```bash
# Run tests (integration tests excluded by default)
pytest tests/ -v

# Run linter
ruff check src/ scripts/

# Run with real LLM integration tests (requires API key)
pytest tests/ -v -m integration
```

---

## Contributing

To add a new agent, see [`docs/ADDING_AGENTS.md`](docs/ADDING_AGENTS.md) for a step-by-step guide.

To add new benchmark examples, see [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{multiagentsecurity2026,
  title  = {Multi-Agent Security Remediation: Architecture Comparison \& Evaluation Framework},
  author = {anupamme},
  year   = {2026},
  url    = {https://github.com/anupamme/Multi_Agent_Security_Remediation_Pipeline}
}
```

---

## License

[MIT](LICENSE) © 2026 anupamme
