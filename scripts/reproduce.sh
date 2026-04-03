#!/usr/bin/env bash
# Reproduce all results from the writeup.
#
# Prerequisites:
#   - Python 3.11+
#   - ANTHROPIC_API_KEY set in environment (or .env file loaded)
#   - Optional: AWS credentials for Bedrock provider
#
# Usage:
#   cp .env.example .env   # fill in your API key
#   bash scripts/reproduce.sh
set -euo pipefail

echo "============================================================"
echo " Multi-Agent Security Remediation Pipeline — Full Reproduce"
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
echo "=== [1/6] Installing dependencies ==="
pip install -e ".[viz,dev]"
echo

# ---------------------------------------------------------------------------
# 2. Validate dataset
# ---------------------------------------------------------------------------
echo "=== [2/6] Validating benchmark dataset ==="
python scripts/validate_dataset.py data/benchmark/
echo

# ---------------------------------------------------------------------------
# 3. Smoke test on dev split (fast, 3 configs, 1 run each)
# ---------------------------------------------------------------------------
echo "=== [3/6] Running dev-set smoke test (C1, C4, C7 × 1 run) ==="
python scripts/run_full_benchmark.py \
    --config config/arch_sequential.yaml \
    --split dev \
    --configs C1,C4,C7 \
    --runs 1 \
    --output-dir data/results/reproduce_smoke/
echo

# ---------------------------------------------------------------------------
# 4. Full benchmark on test split (all 9 configs × 3 runs, 4 parallel workers)
# ---------------------------------------------------------------------------
echo "=== [4/6] Running full benchmark (all 9 configs × 3 runs) ==="
echo "    This step requires a valid ANTHROPIC_API_KEY and takes ~30–60 min."
python scripts/run_full_benchmark.py \
    --config config/arch_sequential.yaml \
    --split test \
    --runs 3 \
    --parallel 4 \
    --output-dir data/results/reproduce_full/
echo

# ---------------------------------------------------------------------------
# 5. Ablation studies
# ---------------------------------------------------------------------------
echo "=== [5/6] Running ablation studies (FULL, A0–A4 × 3 runs) ==="
python scripts/run_ablations.py \
    --config config/arch_sequential.yaml \
    --split test \
    --runs 3 \
    --output-dir data/results/reproduce_ablations/
echo

# ---------------------------------------------------------------------------
# 6. Generate reports and figures
# ---------------------------------------------------------------------------
echo "=== [6/6] Generating comparison reports, failure analysis, and figures ==="

python scripts/generate_comparison.py \
    --results-dir data/results/reproduce_full/ \
    --output data/results/reproduce_full/comparison/

python scripts/generate_failure_report.py \
    --results-dir data/results/reproduce_full/ \
    --output data/results/reproduce_full/failure_analysis/

python scripts/generate_figures.py \
    --results-dir data/results/reproduce_full/ \
    --output docs/figures/
echo

echo "============================================================"
echo " Done!"
echo " Results:  data/results/reproduce_full/"
echo " Figures:  docs/figures/"
echo "============================================================"
