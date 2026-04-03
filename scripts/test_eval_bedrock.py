"""
Manual integration test for the Issue #12 eval framework using Bedrock.

Usage (from project root):
    python scripts/test_eval_bedrock.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_agent_security.config import load_config
from multi_agent_security.eval.runner import EvalRunner

BENCH_DIR = Path("/tmp/eval_bench")
OUTPUT_DIR = Path("/tmp/eval_out")
FIXTURE_REPO = Path(__file__).parent.parent / "tests/fixtures/vulnerable_repo"


def _write_benchmark_json():
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "id": "BENCH-local",
        "repo_url": str(FIXTURE_REPO.resolve()),
        "repo_name": "local/vulnerable_repo",
        "language": "python",
        "vulnerable_files": ["app.py"],
        "vuln_type": "CWE-89",
        "severity": "high",
        "ground_truth_diff": (
            "--- a/app.py\n+++ b/app.py\n@@ -6,2 +6,2 @@\n"
            "-    cursor.execute(f\"SELECT * FROM users WHERE username = '{username}'\")\n"
            "+    cursor.execute(\"SELECT * FROM users WHERE username = ?\", (username,))\n"
        ),
        "merge_status": "merged",
        "complexity_tag": "single_file",
        "negative": False,
        "classification_confidence": 1.0,
        "metadata": {},
    }
    out = BENCH_DIR / "BENCH-local.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"Benchmark JSON written: {out}")


async def main():
    _write_benchmark_json()

    config = load_config("config/arch_sequential_bedrock.yaml")
    runner = EvalRunner(config, output_dir=str(OUTPUT_DIR))

    print("\n--- Running eval (sequential + full_context) ---")
    report = await runner.run_eval(
        benchmark_dir=str(BENCH_DIR),
        architecture="sequential",
        memory_strategy="full_context",
        num_runs=1,
        parallel_workers=1,
    )

    agg = report.aggregate_metrics
    print(f"\nExamples:          {agg.n_examples}")
    print(f"Detection recall:  {agg.detection_recall.mean:.3f}")
    print(f"Triage accuracy:   {agg.triage_accuracy.mean:.3f}")
    print(f"Patch correctness: {agg.patch_correctness.mean:.3f}")
    print(f"E2E success rate:  {agg.e2e_success_rate:.3f}")
    print(f"Cost (USD):        ${agg.total_cost_usd.mean:.4f}")

    reports = list(OUTPUT_DIR.glob("*_report.json"))
    if reports:
        print(f"\nJSON report:       {reports[-1]}")
    mds = list(OUTPUT_DIR.glob("*_report.md"))
    if mds:
        print(f"Markdown report:   {mds[-1]}")
        print("\n--- Markdown report preview (first 40 lines) ---")
        lines = mds[-1].read_text().splitlines()[:40]
        print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
