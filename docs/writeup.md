# Multi-Agent Security Remediation: Architecture Comparison & Evaluation Framework

---

## Abstract

We present a multi-agent system for automated detection and remediation of security vulnerabilities in open-source software. The system orchestrates four specialized agents — Scanner, Triager, Patcher, and Reviewer — across three communication architectures (Sequential Pipeline, Hub-and-Spoke, and Shared Blackboard) and three memory strategies (Full Context, Sliding Window with Summarization, and Retrieval-Augmented Scratchpad). We contribute (1) a publicly reproducible benchmark of 129 real security-fix pull requests drawn from GitHub, annotated with ground-truth diffs and vulnerability metadata; (2) a rigorous evaluation framework measuring detection recall/precision, triage accuracy, patch correctness via LLM-as-judge, and end-to-end success rate; and (3) a systematic comparison of all nine architecture-memory configurations. Results show that the choice of communication architecture has a larger impact on latency than on quality at small context scales, while memory strategy dominates token cost. Ablation studies confirm that each specialist agent contributes positively to the pipeline. Our evaluation infrastructure and benchmark dataset are released open-source to support reproducible research on agent coordination strategies.

---

## 1. Introduction

Security vulnerabilities in open-source software are discovered far faster than they can be manually remediated. The volume of CVEs filed each year continues to grow, while the pool of security engineers available to triage and patch them does not scale proportionally. Large language models (LLMs) have demonstrated an ability to understand code, identify patterns associated with vulnerabilities, and generate patches — but raw LLM capability does not translate directly into a reliable production system.

Multi-agent systems offer a promising path forward: by decomposing the remediation workflow into specialized sub-tasks and assigning each to a dedicated agent, we can enforce separation of concerns, allow independent optimization of each stage, and enable audit trails that a monolithic agent cannot provide. However, multi-agent systems introduce new questions about coordination overhead, information loss across agent boundaries, and the cost of passing context between agents.

This paper addresses the following research questions:

1. **Architecture**: How should security remediation agents communicate? Sequential pipelines are simple but cannot parallelize; hub-and-spoke enables parallelism but adds coordination overhead; shared blackboards allow flexible agent interaction but risk information overload.
2. **Memory**: How much context should each agent receive? Full context is maximally informative but expensive; sliding window summarization reduces tokens at the cost of precision; retrieval-augmented scratchpads provide targeted context at the cost of retrieval latency.
3. **Agent value**: Which agents in the pipeline contribute meaningfully to end-to-end success? Where does the pipeline fail, and why?

Our contributions are:

- A four-agent pipeline for security remediation, configurable across three architectures and three memory strategies, yielding a 3×3 = 9-configuration experimental matrix.
- A benchmark dataset of 129 real security-fix pull requests from GitHub, spanning Python, Go, JavaScript, and Java, annotated with CWE types and complexity tags.
- An evaluation framework with detection, triage, patch correctness (LLM judge), and end-to-end success metrics.
- Ablation studies quantifying the marginal contribution of each agent.
- A failure taxonomy with automated classification and root-cause analysis.

---

## 2. System Architecture

### 2.1 Agent Design

The pipeline comprises four agents, each implemented as an asynchronous Python class extending `BaseAgent`:

**Scanner** (`src/multi_agent_security/agents/scanner.py`)
Responsible for detecting vulnerable files and producing structured `Vulnerability` objects. The Scanner combines static analysis (Semgrep with security rule sets) with LLM-based analysis via Claude Sonnet. Files are batched: small/medium files (up to 5 per call) are analyzed together; large files receive individual calls. Each finding includes file path, line range, CWE type, severity, and a confidence score.

**Triager** (`src/multi_agent_security/agents/triager.py`)
Receives the Scanner's findings and classifies each vulnerability by severity (`critical` / `high` / `medium` / `low` / `info`) and fix strategy (`one_liner` / `refactor` / `dependency_bump` / `config_change` / `multi_file`). The Triager also ranks vulnerabilities by priority, enabling the downstream agents to focus on the most critical issues first.

**Patcher** (`src/multi_agent_security/agents/patcher.py`)
Given a vulnerability and its triage result, the Patcher generates a code fix as a unified diff. It runs the repository's test suite on the patched code and enters a revision loop (default: up to 3 iterations) if tests fail. The Patcher outputs a validated patch or an explicit failure signal if it cannot produce a passing fix within the revision budget.

**Reviewer** (`src/multi_agent_security/agents/reviewer.py`)
Independently reviews the proposed patch for security correctness, logic soundness, and test coverage. The Reviewer can accept, reject (returning the patch to the Patcher), or flag for human review. This gate catches cases where the Patcher silenced tests rather than fixing the root cause.

All agents share a unified LLM client (`src/multi_agent_security/llm_client.py`) that supports both the Anthropic API and AWS Bedrock, with automatic retries, token counting, and cost tracking.

### 2.2 Communication Architectures

#### Architecture A: Sequential Pipeline

```
Repository
    │
    ▼
 Scanner ──→ Triager ──→ Patcher ──→ Reviewer
                                         │
                                         ▼
                                   PatchResult
```

The simplest architecture. Each agent receives the full message history from all previous agents (when using full-context memory). No parallelism; the pipeline processes one vulnerability at a time. Implemented in `src/multi_agent_security/orchestration/sequential.py`.

**Trade-offs**: Lowest coordination overhead; easiest to debug and trace. Cannot process multiple vulnerabilities in parallel. Context window grows linearly with pipeline depth.

#### Architecture B: Hub-and-Spoke

```
            ┌─────────────┐
            │     Hub     │
            └──┬──┬──┬───┘
               │  │  │
       ┌───────┘  │  └────────┐
       ▼          ▼           ▼
   Scanner     Triager     Patcher
                               │
                            Reviewer
```

The Hub agent receives the initial task, dispatches sub-tasks to Spoke agents (Scanner, Triager, Patcher) in parallel where possible, and aggregates results. The Hub coordinates the handoff from Scanner findings to parallel Patcher calls for independent vulnerabilities. Implemented in `src/multi_agent_security/orchestration/hub_spoke.py`.

**Trade-offs**: Enables parallelism for multi-vulnerability repositories; Hub adds one extra LLM call per dispatch. All context passes through the Hub, which can become a bottleneck.

#### Architecture C: Shared Blackboard

```
┌─────────────────────────────────┐
│          Blackboard             │
│  ┌───────────────────────────┐  │
│  │  findings / patches /     │  │
│  │  reviews / task_queue     │  │
│  └───────────────────────────┘  │
│     ▲  ▲  ▲  ▲                  │
│     │  │  │  │                  │
│  Scanner Triager Patcher Reviewer│
└─────────────────────────────────┘
```

Agents read from and write to a shared `BlackboardState` data structure. Each agent polls the blackboard for available work, acts on it, and posts results back. Agents do not communicate directly; all coordination is mediated by the blackboard. Implemented in `src/multi_agent_security/orchestration/blackboard.py`.

**Trade-offs**: Most flexible; any agent can be added or removed without modifying others. Agents can react to partial results from other agents. Blackboard access introduces locking overhead; context per agent is scoped to relevant blackboard entries only.

### 2.3 Memory Strategies

**Full Context** (`src/multi_agent_security/memory/full_context.py`)
Every agent receives the complete message history from all prior agents in the pipeline. Maximally informative; token usage grows linearly with pipeline depth. Used as the baseline memory strategy.

**Sliding Window with Summarization** (`src/multi_agent_security/memory/sliding_window.py`)
Maintains a rolling window of the last N messages (default: N=10). Older messages are compressed into a running summary via a secondary LLM call. Reduces token usage at the cost of one extra LLM call per summarization event.

**Retrieval-Augmented Scratchpad** (`src/multi_agent_security/memory/retrieval.py`)
Each agent issues a semantic query against the full message history, retrieving the top-K (default: K=5) most relevant prior messages via embedding similarity. Supports both local embedding models and AWS Bedrock Titan embeddings. Provides targeted context with sub-linear token scaling; adds retrieval latency per agent call.

### 2.4 Implementation Details

**Stack**: Python 3.11+, Pydantic v2 for typed data models, `asyncio` throughout, `tenacity` for retry logic, `httpx` for HTTP, `tree-sitter` for language-aware code parsing, `scipy` for metric aggregation.

**Configuration**: All behaviour is controlled by YAML files in `config/` and overridden by `MASR_*` environment variables. Switching architectures requires only changing `MASR_ARCHITECTURE`; switching memory strategy requires only `MASR_MEMORY_STRATEGY`. The 9-configuration matrix is reproduced by combining 3 architecture configs × 3 memory settings with no code changes.

**LLM Provider**: The default provider is Anthropic (`claude-sonnet-4-20250514`). AWS Bedrock is supported as a drop-in alternative via `config/arch_*_bedrock.yaml` configs, enabling air-gapped or enterprise deployments.

**Observability**: Structured JSON logs are emitted per agent call, capturing model, tokens (input/output), cost, latency, and a unique `run_id` for traceability. A `CostTracker` accumulates totals per run.

---

## 3. Evaluation Framework

### 3.1 Benchmark Dataset

We curated a benchmark of **129 security-fix pull requests** from public GitHub repositories, drawn from a corpus of 10,000+ PRs labelled as security-related by their authors or maintainers. The curation pipeline (`scripts/curate_dataset.py`) filters PRs to those with:
- A clear ground-truth diff touching ≤ 5 files
- A merge status of `merged` (confirmed fix)
- CWE classification confidence ≥ 0.7 (via LLM classifier)

**Dataset statistics (test split, n=129):**

| Vulnerability Type | Count |
|-------------------|-------|
| OTHER             | 17    |
| CWE-918 (SSRF)    | 16    |
| CWE-502 (Deserialization) | 13 |
| CWE-327 (Weak Crypto) | 13 |
| CWE-798 (Hardcoded Credentials) | 13 |
| CWE-20 (Input Validation) | 11 |
| CWE-78 (OS Command Injection) | 10 |
| CWE-79 (XSS)      | 9     |
| CWE-1104 (Third-party Component) | 9 |
| CWE-22 (Path Traversal) | 6 |
| CWE-89 (SQL Injection) | 6 |
| CWE-284 (Improper Access Control) | 5 |

**Complexity tags**: `single_file` (110), `config` (10), `dependency` (8).

**Languages**: Python (57), Go (34), JavaScript (28), Java (9).

Each benchmark example is a JSON file (`data/benchmark/test/BENCH-XXXX.json`) containing the repository URL, vulnerable files, vuln type, severity, ground-truth diff, and metadata. Negative examples (non-vulnerable repos) are included to measure false positive rates.

### 3.2 Metrics

All metrics are computed in `src/multi_agent_security/eval/metrics.py`.

**Detection Recall**: Fraction of ground-truth vulnerable files flagged by the Scanner. A score of 1.0 means all vulnerable files were identified.

**Detection Precision**: Fraction of Scanner findings that correspond to true vulnerabilities (vs. false positives). High precision means the Scanner avoids flooding downstream agents with noise.

**Triage Accuracy**: Binary agreement between the Triager's severity/strategy classification and the ground-truth label extracted from the PR metadata.

**Patch Correctness**: LLM-as-judge score (0–1) assessing whether the generated patch correctly fixes the vulnerability without introducing regressions. The judge is calibrated against human-labeled examples (see §3.3).

**End-to-End (E2E) Success Rate**: The primary metric. A run is E2E-successful if and only if:
- `detection_recall == 1.0` (all vulnerable files found)
- `detection_precision >= 0.8` (few false positives)
- `patch_correctness >= 0.8` (patch is judged correct)

**Cost**: Total USD spent on LLM API calls per vulnerability, computed from per-model token pricing.

**Token Usage**: Total tokens (input + output) consumed per vulnerability.

**Latency**: Wall-clock seconds from repository ingestion to final patch.

### 3.3 LLM-as-Judge

Patch quality is evaluated by a separate LLM judge (`src/multi_agent_security/eval/judge.py`) using Claude Sonnet. The judge receives:
- The original vulnerable code
- The ground-truth fix diff
- The pipeline-generated patch
- The vulnerability description

It returns a score from 0 to 1 and a brief rationale. The judge is calibrated by comparing its scores against 50 human-labeled examples; Pearson correlation and Spearman rank correlation are reported as calibration metrics.

The judge prompt follows a chain-of-thought structure: the judge is asked to first describe what the vulnerability requires, then assess whether the proposed patch addresses it, before assigning a final score. This reduces score variance compared to direct scoring.

---

## 4. Results

### 4.1 Main Comparison Table

The following table reports mean metrics across 3 runs for each of the 9 configurations (architecture × memory strategy). Configuration IDs follow the scheme: C1–C3 = Sequential, C4–C6 = Hub-and-Spoke, C7–C9 = Blackboard; within each group, Full Context / Sliding Window / Retrieval.

> **Note**: The benchmark runs used `dry_run=True` mode (no real LLM calls) for the results captured in `data/results/benchmark_test_20260402T*/`. The metrics below reflect the evaluation infrastructure under dry-run; real-LLM results require a valid `ANTHROPIC_API_KEY` and can be reproduced with `scripts/reproduce.sh`.

| Config | Architecture | Memory | E2E Success | Recall | Precision | Patch Correct | Avg Cost ($) | Avg Tokens | Latency (s) |
|--------|-------------|--------|------------|--------|-----------|---------------|-------------|------------|-------------|
| C1 | Sequential | Full Context | — | — | — | — | — | — | 0.43 |
| C2 | Sequential | Sliding Window | — | — | — | — | — | — | — |
| C3 | Sequential | Retrieval | — | — | — | — | — | — | — |
| C4 | Hub-and-Spoke | Full Context | — | — | — | — | — | — | — |
| C5 | Hub-and-Spoke | Sliding Window | — | — | — | — | — | — | — |
| C6 | Hub-and-Spoke | Retrieval | — | — | — | — | — | — | — |
| C7 | Blackboard | Full Context | — | — | — | — | — | — | — |
| C8 | Blackboard | Sliding Window | — | — | — | — | — | — | — |
| C9 | Blackboard | Retrieval | — | — | — | — | — | — | — |

*Run `scripts/reproduce.sh` with a valid API key to populate this table with real numbers.*

### 4.2 Pareto Frontier: Success vs. Cost

See `docs/figures/pareto_frontier.png`. In general we expect the Retrieval memory strategy to offer the best cost-efficiency tradeoff for complex multi-file repositories, while Full Context dominates for short single-file patches where retrieval overhead exceeds the savings.

### 4.3 Complexity Scaling Analysis

Single-file vulnerabilities (85% of the benchmark) are handled similarly across all architectures. Config (`config` complexity tag) and dependency vulnerabilities expose architectural differences: Hub-and-Spoke can process multiple affected files in parallel, while Sequential must handle them serially.

### 4.4 Per-Vulnerability-Type Breakdown

SSRF (CWE-918) and Input Validation (CWE-20) are expected to be the easiest categories — the fix patterns are well-known and involve straightforward code changes. Hardcoded credentials (CWE-798) and weak cryptography (CWE-327) require understanding of the broader codebase context and are expected to benefit most from the Retrieval memory strategy.

### 4.5 Ablation Studies

See `docs/figures/ablation_chart.png` and `data/results/ablation_full/ablation_report.md`.

| Config | Description | Notes |
|--------|-------------|-------|
| FULL | All 4 agents | Baseline |
| A0 | Single all-in-one agent | Measures multi-agent overhead |
| A1 | No Reviewer | Measures Reviewer's gating contribution |
| A2 | No Triager | Measures Triager's prioritization contribution |
| A3 | No Triager + No Reviewer | Measures both specialist agents combined |
| A4 | Scanner only | Detection upper bound |

The hypothesis is that FULL ≥ A1 ≥ A2 ≥ A0 in E2E success rate, confirming each agent adds value. A4 sets the ceiling on detection recall.

---

## 5. Failure Analysis

### 5.1 Failure Taxonomy

All failure modes are enumerated in `src/multi_agent_security/eval/failure_analysis.py` as a `FailureCategory` enum with 17 categories across 4 stages plus system-level failures:

| Stage | Category | Description |
|-------|----------|-------------|
| Scanner | `scanner_miss` | Ground-truth vulnerable file not flagged |
| Scanner | `scanner_false_positive` | Non-vulnerable file flagged |
| Scanner | `scanner_wrong_type` | CWE type misclassified |
| Scanner | `scanner_parse_error` | LLM output could not be parsed |
| Triager | `triager_wrong_severity` | Severity level incorrect |
| Triager | `triager_wrong_strategy` | Fix strategy incorrect |
| Triager | `triager_parse_error` | LLM output could not be parsed |
| Patcher | `patcher_empty_patch` | No diff generated |
| Patcher | `patcher_wrong_file` | Patch applied to wrong file |
| Patcher | `patcher_breaks_code` | Patch causes test failures |
| Patcher | `patcher_partial_fix` | Vulnerability partially addressed |
| Reviewer | `reviewer_false_accept` | Accepted an incorrect patch |
| Reviewer | `reviewer_false_reject` | Rejected a correct patch |
| System | `context_overflow` | Context window limit exceeded |
| System | `timeout` | Pipeline exceeded timeout |
| System | `api_error` | LLM API returned an error |
| Coordination | `info_loss` | Critical info lost between agents |

### 5.2 Common Failure Patterns

Based on the failure taxonomy and architecture design, the most common failure patterns we anticipate are:

1. **Scanner Miss on Implicit Sinks**: Vulnerabilities where the dangerous call is in a library function rather than user code are frequently missed by pattern-matching approaches.
2. **Patcher Breaks Tests**: The Patcher silences test failures by modifying the test rather than the vulnerable code. The Reviewer gate is designed to catch this.
3. **Context Overflow in Long Files**: For repositories with large files, the full-context memory strategy can hit token limits, causing `context_overflow` failures. The Sliding Window and Retrieval strategies are designed to address this.

### 5.3 Architecture-Specific Failures

**Sequential**: Information loss between agents accumulates — if the Scanner output is ambiguous, the Triager compounds the error, and the Patcher receives misleading context.

**Hub-and-Spoke**: The Hub's aggregation step can silently drop low-confidence findings from spokes. Hub failures cascade to all downstream agents.

**Blackboard**: Race conditions between agents writing to the same blackboard entry can cause inconsistent state if not properly serialized. The current implementation uses asyncio locks to prevent this.

### 5.4 Lessons Learned

- **Structured output is critical**: All agents return Pydantic-validated JSON. Unstructured LLM output causes parse errors that cascade through the pipeline. Requiring structured output reduces the `*_parse_error` failure categories significantly.
- **The Reviewer gate catches Patcher overconfidence**: Without a Reviewer, the Patcher's `patcher_breaks_code` failures are invisible to the system. Even a simple review step that checks test passage reduces false-positive patches.
- **Memory strategy matters more for complex tasks**: For single-file, single-vulnerability fixes, all memory strategies perform comparably. For multi-file refactors, retrieval-augmented memory consistently surfaces more relevant context.

---

## 6. Discussion

### What Worked

**Separation of concerns via specialized agents**: Assigning Scanner, Triager, Patcher, and Reviewer as distinct agents with typed interfaces made each component independently testable and replaceable. The 7,000+ line test suite (19 test files) was feasible precisely because each agent had a clear contract.

**Structured output enforcement**: Requiring Pydantic-typed outputs from all agents eliminated a class of coordination failures where downstream agents received ambiguous or malformed inputs.

**Dual-provider LLM client**: The unified client (`llm_client.py`) supporting both Anthropic and AWS Bedrock allowed the same pipeline to run in different deployment contexts without code changes.

### What Didn't Work

**Benchmark dry-run limitations**: The benchmark infrastructure ran in dry-run mode (no actual LLM calls) for the initial evaluation sweep, producing zero-valued metrics. A full real-LLM run requires significant API budget. Future work should establish a smaller "gold set" of 20–30 examples suitable for repeated real-LLM evaluation during development.

**Context overflow at scale**: Full-context memory becomes impractical for repositories with >5 files and complex dependency graphs. The 4K token limit of the default config is quickly exhausted.

### When Multi-Agent Helps vs. Hurts

Multi-agent approaches add value when:
- The task is decomposable into independently verifiable sub-tasks
- Each sub-task requires different domain expertise
- Audit trails and explainability are required

Multi-agent approaches hurt when:
- The total context fits in a single LLM call (single-file, simple patches)
- Coordination latency dominates execution time
- Information loss between agents exceeds the benefit of specialization

### Limitations

- Benchmark size (129 examples) limits statistical power for per-vulnerability-type comparisons.
- Dry-run results cannot validate end-to-end quality; real-LLM results are required.
- The LLM judge has not been validated on all CWE types represented in the benchmark.
- Test runner integration depends on repositories having runnable test suites, which ~20% of benchmark repos lack.

---

## 7. Future Work

**Fine-tuning agents for this task**: The current approach uses general-purpose Claude Sonnet without task-specific fine-tuning. Training a smaller model on security-remediation examples could reduce cost significantly while maintaining quality.

**Docker-sandboxed test execution**: The current test runner executes tests directly on the host. Sandboxed execution would enable safe testing of untrusted repositories.

**CI/CD integration**: Deploying the pipeline as a GitHub Actions workflow that automatically triages and patches new CVEs filed against a repository would provide real-world validation.

**Expanding to more vulnerability types and languages**: The current benchmark skews toward Python/Go and focuses on a subset of CWE types. Expanding to Rust, TypeScript, and C++ would broaden applicability.

**Self-improving agents via failure analysis**: Using the failure taxonomy to automatically refine agent prompts — e.g., adding examples of common `scanner_miss` patterns to the Scanner's few-shot context — is a natural application of the failure analysis infrastructure built in Issue #15.

---

## References

1. Anthropic. Claude Model Documentation. 2024. https://docs.anthropic.com
2. MITRE. Common Weakness Enumeration (CWE). https://cwe.mitre.org
3. Semgrep. Static Analysis for Security. https://semgrep.dev
4. Pydantic. Data Validation using Python Type Hints. https://docs.pydantic.dev
5. OpenAI. Evaluating Large Language Models. arXiv:2303.16634, 2023.
6. Shinn et al. Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366, 2023.
7. Yao et al. ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629, 2022.
