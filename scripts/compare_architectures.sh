#!/usr/bin/env bash
# Compare all three architectures on the same repo.
# Usage: bash scripts/compare_architectures.sh [/path/to/repo] [language]
#
# Defaults to the built-in vulnerable_repo fixture with language=python.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REPO="${1:-$REPO_ROOT/tests/fixtures/vulnerable_repo}"
LANGUAGE="${2:-python}"

CONFIGS=(
  "arch_sequential_bedrock"
  "arch_hub_spoke_bedrock"
  "arch_blackboard_bedrock"
)

summarise() {
  python3 - "$1" <<'PYEOF'
import json, sys
path = sys.argv[1]
with open(path) as f:
    d = json.load(f)
print(f"  status          : {d['status']}")
print(f"  vulnerabilities : {len(d['vulnerabilities'])}")
accepted = sum(1 for r in d['reviews'] if r['patch_accepted'])
print(f"  accepted patches: {accepted} / {len(d['reviews'])}")
total_tokens = sum(m['token_count_input'] + m['token_count_output'] for m in d['messages'])
total_cost   = sum(m['cost_usd'] for m in d['messages'])
print(f"  total tokens    : {total_tokens}")
print(f"  total cost USD  : ${total_cost:.4f}")
print(f"  agent sequence  : {[m['agent_name'] for m in d['messages']]}")
PYEOF
}

mkdir -p "$REPO_ROOT/data/results"

for ARCH in "${CONFIGS[@]}"; do
  CONFIG="$REPO_ROOT/config/${ARCH}.yaml"
  if [[ ! -f "$CONFIG" ]]; then
    echo "=== $ARCH: config not found, skipping ==="
    continue
  fi

  OUT="$REPO_ROOT/data/results/${ARCH}_result.json"
  echo ""
  echo "=== $ARCH ==="
  python3 "$SCRIPT_DIR/run_pipeline.py" \
    --config "$CONFIG" \
    --repo "$REPO" \
    --language "$LANGUAGE" \
    > "$OUT" 2>&1 || { echo "  FAILED — see $OUT for details"; continue; }

  summarise "$OUT"
done

echo ""
echo "Full results written to data/results/"
