# Demo Script (5 minutes)

A walkthrough of the Multi-Agent Security Remediation Pipeline, suitable for recording
as a 5-minute screen-capture demo.

**Setup before recording:**
- Terminal font size ≥ 18pt; dark background theme
- Keep terminal width at ~100 columns
- Have the architecture diagram (`docs/ARCHITECTURE.md`) open in a browser tab
- Pre-clone one target repo: `git clone https://github.com/owner/vulnerable-repo data/demo/repo`
- Set `ANTHROPIC_API_KEY` in your shell

---

## 0:00 – 0:30 — Introduction (30 seconds)

**Show:** `docs/ARCHITECTURE.md` rendered in browser — the Mermaid architecture diagrams.

**Say:**
> "This is a multi-agent system that automatically detects and patches security
> vulnerabilities in open-source codebases. It coordinates four specialist agents —
> Scanner, Triager, Patcher, and Reviewer — using three different communication
> architectures: a sequential pipeline, a hub-and-spoke design, and a shared blackboard.
> Let me show you what a live run looks like."

---

## 0:30 – 2:00 — Live Pipeline Run (90 seconds)

**Show:** Terminal. Run the pipeline on the pre-cloned demo repository.

```bash
MASR_ARCHITECTURE=sequential \
MASR_MEMORY_STRATEGY=full_context \
python scripts/run_pipeline.py \
    --repo data/demo/repo \
    --output data/demo/result.json \
    --verbose
```

**Point out as output scrolls:**
- `[Scanner]` — list of files analyzed, Semgrep findings, LLM vulnerability analysis
- Token count and cost per agent call (e.g., `tokens=1,247 cost=$0.0037`)
- `[Triager]` — severity classification (`high`), fix strategy (`refactor`)
- `[Patcher]` — diff being generated, test run results (`2 passed`)
- `[Reviewer]` — acceptance decision with rationale

**Say:**
> "The Scanner identified a SQL injection in `views.py` at line 42. The Triager classified
> it as high severity with a refactor fix strategy. The Patcher generated a parameterized
> query patch — tests passed on the first attempt, no revision needed. The Reviewer
> confirmed the fix is correct."

**Show:** The final diff output:
```bash
cat data/demo/result.json | python -m json.tool | grep -A 20 '"diff"'
```

---

## 2:00 – 3:30 — Benchmark Results (90 seconds)

**Show:** Terminal — the comparison table, then the Pareto frontier figure.

```bash
cat data/results/benchmark_test_*/comparison/comparison_table.md
```

**Say:**
> "We ran all nine configurations — three architectures by three memory strategies —
> across 129 real security-fix PRs from GitHub. Here's the comparison table."

**Show:** Open `docs/figures/pareto_frontier.png` in the image viewer.

**Say:**
> "This is the Pareto frontier: end-to-end success rate on the Y axis, average cost per
> vulnerability on the X axis. The ideal config sits in the top-left corner.
> In our runs, [describe the best config based on actual numbers].
> The key insight is that switching from full-context to retrieval memory cuts token usage
> significantly with minimal quality loss for single-file fixes."

**Show:** `docs/figures/architecture_comparison.png`

**Say:**
> "Looking at the architecture comparison: hub-and-spoke gains latency advantages for
> multi-vulnerability repositories, while the sequential pipeline remains competitive
> for the majority of our benchmark which is single-file fixes."

---

## 3:30 – 4:30 — Failure Analysis (60 seconds)

**Show:** `docs/figures/failure_heatmap.png`

**Say:**
> "Not everything works — let me show you where the pipeline fails."

```bash
cat data/results/*/failure_analysis/failure_report.md | head -50
```

**Say:**
> "The failure heatmap shows failure rates by stage and configuration. Patcher failures
> dominate — either an empty patch or a patch that breaks tests. The Reviewer gate
> catches many of these, but not all.
>
> The biggest bottleneck is Scanner recall on implicit sinks — cases where the vulnerable
> call is inside a library function. Hub-and-spoke specifically struggles when the Hub
> drops low-confidence Scanner findings during aggregation.
>
> This suggests that improving Scanner sensitivity and making the Hub's aggregation
> strategy more conservative are the highest-leverage improvements."

---

## 4:30 – 5:00 — Conclusion (30 seconds)

**Show:** Slide or terminal with project structure.

**Say:**
> "The most interesting finding: for single-file, simple fixes, the sequential pipeline
> with full-context memory is hard to beat — it's simpler, cheaper, and just as accurate.
> Multi-agent parallelism pays off only when there are multiple independent vulnerabilities
> or large codebases that overflow single-call context windows.
>
> If I were to continue this work, the highest-priority next step would be using the
> failure taxonomy to automatically refine agent prompts — feeding `scanner_miss` examples
> back into the Scanner's few-shot context to improve recall.
>
> The full codebase, benchmark dataset, and reproduction scripts are available on GitHub."

---

## Recording Notes

- Use QuickTime (macOS) or OBS Studio for screen recording
- Record at 1920×1080 minimum; export at 1080p
- Terminal: use a dark theme (e.g., Dracula or One Dark) for readability
- Zoom in on key output lines with `⌘+` before recording those sections
- Target audio: clear narration, no background noise; use a headset microphone
- Target file size: < 500MB (compress with HandBrake if needed)
- Upload to YouTube (unlisted) or Loom for sharing with Anthropic
