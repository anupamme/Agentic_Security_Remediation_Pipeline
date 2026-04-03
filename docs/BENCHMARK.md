# Benchmark Dataset

The benchmark consists of real security-fix pull requests from public GitHub repositories,
curated and annotated for use as evaluation examples.

---

## Dataset Layout

```
data/benchmark/
├── test/          # Primary test split (129 examples: BENCH-0101 ... BENCH-0229)
├── dev/           # Development split (used for smoke tests)
├── hard/          # Hard difficulty examples (multi-file, complex fixes)
└── local/         # Local evaluation set (for quick iteration)
```

---

## Example Format

Each example is a JSON file named `BENCH-XXXX.json`:

```json
{
  "id": "BENCH-0101",
  "repo_url": "https://github.com/owner/repo",
  "repo_name": "owner/repo",
  "language": "python",
  "vulnerable_files": ["src/app/views.py"],
  "vuln_type": "CWE-89",
  "severity": "high",
  "ground_truth_diff": "--- a/src/app/views.py\n+++ b/src/app/views.py\n...",
  "merge_status": "merged",
  "complexity_tag": "single_file",
  "negative": false,
  "pr_url": "https://github.com/owner/repo/pull/42",
  "classification_confidence": 0.92,
  "metadata": {
    "pr_title": "Fix SQL injection in user search",
    "pr_merged_at": "2024-03-15T10:22:00Z"
  }
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique benchmark example ID |
| `repo_url` | str | GitHub repository URL |
| `repo_name` | str | `owner/repo` short form |
| `language` | str | Primary language (`python`, `go`, `javascript`, `java`) |
| `vulnerable_files` | list[str] | Files that contain the vulnerability |
| `vuln_type` | str | CWE identifier (e.g., `CWE-89`) or `OTHER` |
| `severity` | str | `critical` / `high` / `medium` / `low` / `info` |
| `ground_truth_diff` | str | Unified diff of the ground-truth fix |
| `merge_status` | str | `merged` (confirmed real fix) or `open` |
| `complexity_tag` | str | `single_file` / `config` / `dependency` |
| `negative` | bool | `true` = non-vulnerable repo (for false positive testing) |
| `pr_url` | str | Source pull request URL |
| `classification_confidence` | float | LLM classifier confidence (0–1) |
| `metadata` | dict | Additional PR metadata |

---

## Dataset Statistics (test split, n=129)

| Vulnerability Type | Count | % |
|-------------------|-------|---|
| OTHER | 17 | 13.2% |
| CWE-918 (SSRF) | 16 | 12.4% |
| CWE-502 (Deserialization) | 13 | 10.1% |
| CWE-327 (Weak Crypto) | 13 | 10.1% |
| CWE-798 (Hardcoded Credentials) | 13 | 10.1% |
| CWE-20 (Input Validation) | 11 | 8.5% |
| CWE-78 (OS Command Injection) | 10 | 7.8% |
| CWE-79 (XSS) | 9 | 7.0% |
| CWE-1104 (Third-party Component) | 9 | 7.0% |
| CWE-22 (Path Traversal) | 6 | 4.7% |
| CWE-89 (SQL Injection) | 6 | 4.7% |
| CWE-284 (Improper Access Control) | 5 | 3.9% |

**Complexity tags**: `single_file` (110, 85%), `config` (10, 8%), `dependency` (8, 6%)

**Languages**: Python (57, 44%), Go (34, 26%), JavaScript (28, 22%), Java (9, 7%)

---

## Adding New Examples

### Option 1: Manual Curation

1. Find a merged security-fix PR on GitHub.
2. Create a JSON file following the format above.
3. Place it in `data/benchmark/test/` with the next sequential ID.
4. Run validation:
   ```bash
   python scripts/validate_dataset.py data/benchmark/
   ```

### Option 2: Automated Curation

The `scripts/curate_dataset.py` script can discover and annotate PRs automatically:

```bash
python scripts/curate_dataset.py \
    --github-token $GITHUB_TOKEN \
    --query "type:pr label:security is:merged" \
    --output data/benchmark/new/ \
    --max-examples 100
```

The curation pipeline:
1. Searches GitHub for security-labelled merged PRs
2. Filters to PRs that touch ≤ 5 files with a clear diff
3. Runs a CWE classifier (Claude Sonnet) to assign `vuln_type`
4. Keeps only examples with `classification_confidence >= 0.7`

### Option 3: From Local Repositories

```bash
python scripts/curate_dataset.py \
    --local-repo /path/to/repo \
    --commit-range v1.0..v1.1 \
    --output data/benchmark/local/
```

---

## Validation

Run `python scripts/validate_dataset.py data/benchmark/` to check:

- All required fields are present
- `ground_truth_diff` is a valid unified diff
- `vuln_type` is a known CWE or `OTHER`
- `complexity_tag` is one of `single_file` / `config` / `dependency`
- No duplicate `id` values
- `classification_confidence` is in [0, 1]

---

## Splits

| Split | Purpose | Size |
|-------|---------|------|
| `dev` | Quick iteration and smoke tests | Small |
| `test` | Primary evaluation (report results on this) | 129 |
| `hard` | Stress-testing with multi-file fixes | Small |
| `local` | Ad-hoc local experiments | Variable |

To run evaluation on a specific split:
```bash
python scripts/run_full_benchmark.py --split test --configs C1,C4,C7 --runs 3
```
