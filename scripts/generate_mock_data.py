"""Generate synthetic PR records for testing the curation pipeline.

Produces structurally realistic BenchmarkExample-compatible JSON records that
can be fed directly into curate_dataset.py via --input.

Usage:
    python scripts/generate_mock_data.py --count 500 --output data/raw/mock_prs.json --seed 42
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_agent_security.types import BenchmarkExample, VulnSeverity, VulnType

# ---------------------------------------------------------------------------
# Mock repos (20 fictional GitHub repos with varied languages)
# ---------------------------------------------------------------------------

def _build_mock_repos() -> list[dict[str, str]]:
    """Build a diverse set of 100 fictional GitHub repos."""
    orgs = [
        "acme-corp", "example-org", "opentools", "devhub", "startup-co",
        "secure-io", "cloudnative", "fintech-oss", "web-platform", "infra-tools",
    ]
    services = [
        ("webapp", "python"), ("api-gateway", "go"), ("frontend", "javascript"),
        ("payment-svc", "java"), ("auth-service", "python"), ("data-pipeline", "python"),
        ("mobile-api", "go"), ("admin-panel", "javascript"), ("cli-util", "go"),
        ("lib-crypto", "python"),
    ]
    repos = []
    for org in orgs:
        for svc, lang in services:
            repos.append(
                {"repo_url": f"https://github.com/{org}/{svc}", "language": lang}
            )
    return repos


MOCK_REPOS = _build_mock_repos()

# ---------------------------------------------------------------------------
# Minimal realistic unified diffs per vulnerability type
# ---------------------------------------------------------------------------

MOCK_DIFFS: dict[str, str] = {
    VulnType.SQL_INJECTION: (
        "--- a/src/db.py\n+++ b/src/db.py\n"
        "@@ -10,7 +10,7 @@\n"
        " def get_user(user_id):\n"
        "-    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n"
        "+    query = \"SELECT * FROM users WHERE id = %s\"\n"
        "     cursor.execute(query, (user_id,))\n"
        "     return cursor.fetchone()\n"
    ),
    VulnType.XSS: (
        "--- a/src/views.py\n+++ b/src/views.py\n"
        "@@ -5,5 +5,5 @@\n"
        " def render_comment(comment):\n"
        "-    return f\"<div>{comment}</div>\"\n"
        "+    return f\"<div>{html.escape(comment)}</div>\"\n"
    ),
    VulnType.PATH_TRAVERSAL: (
        "--- a/src/files.py\n+++ b/src/files.py\n"
        "@@ -8,7 +8,10 @@\n"
        " def read_file(filename):\n"
        "-    path = os.path.join(BASE_DIR, filename)\n"
        "+    # Prevent directory traversal\n"
        "+    safe_name = os.path.basename(filename)\n"
        "+    path = os.path.join(BASE_DIR, safe_name)\n"
        "+    path = os.path.realpath(path)\n"
        "     with open(path) as f:\n"
        "         return f.read()\n"
    ),
    VulnType.COMMAND_INJECTION: (
        "--- a/src/runner.py\n+++ b/src/runner.py\n"
        "@@ -3,5 +3,5 @@\n"
        " def run_tool(name):\n"
        "-    os.system(f\"tool {name}\")\n"
        "+    subprocess.run([\"tool\", name], check=True)\n"
    ),
    VulnType.INSECURE_DESERIALIZATION: (
        "--- a/src/loader.py\n+++ b/src/loader.py\n"
        "@@ -2,5 +2,5 @@\n"
        " def load_data(blob):\n"
        "-    return pickle.loads(blob)\n"
        "+    return json.loads(blob)\n"
    ),
    VulnType.HARDCODED_CREDENTIALS: (
        "--- a/src/config.py\n+++ b/src/config.py\n"
        "@@ -1,5 +1,5 @@\n"
        "-API_KEY = \"hardcoded-secret-key-123\"\n"
        "+API_KEY = os.environ[\"API_KEY\"]\n"
    ),
    VulnType.INSECURE_DEPENDENCY: (
        "--- a/requirements.txt\n+++ b/requirements.txt\n"
        "@@ -3,3 +3,3 @@\n"
        "-requests==2.18.0\n"
        "+requests==2.31.0\n"
    ),
    VulnType.IMPROPER_INPUT_VALIDATION: (
        "--- a/src/api.py\n+++ b/src/api.py\n"
        "@@ -7,5 +7,7 @@\n"
        " def process_age(value):\n"
        "-    age = int(value)\n"
        "+    age = int(value)\n"
        "+    if age < 0 or age > 150:\n"
        "+        raise ValueError(\"Invalid age\")\n"
        "     return age\n"
    ),
    VulnType.BROKEN_ACCESS_CONTROL: (
        "--- a/src/admin.py\n+++ b/src/admin.py\n"
        "@@ -5,5 +5,7 @@\n"
        " def delete_user(user_id):\n"
        "+    if not current_user.is_admin:\n"
        "+        raise PermissionError(\"Admin access required\")\n"
        "     db.delete(User, user_id)\n"
    ),
    VulnType.CRYPTO_WEAKNESS: (
        "--- a/src/auth.py\n+++ b/src/auth.py\n"
        "@@ -4,5 +4,5 @@\n"
        " def hash_password(pw):\n"
        "-    return hashlib.md5(pw.encode()).hexdigest()\n"
        "+    return hashlib.sha256(pw.encode()).hexdigest()\n"
    ),
    VulnType.SSRF: (
        "--- a/src/proxy.py\n+++ b/src/proxy.py\n"
        "@@ -6,5 +6,10 @@\n"
        " def fetch_url(url):\n"
        "+    parsed = urlparse(url)\n"
        "+    if parsed.hostname in BLOCKED_HOSTS:\n"
        "+        raise ValueError(\"Blocked host\")\n"
        "     return requests.get(url).text\n"
    ),
    VulnType.OTHER: (
        "--- a/src/misc.py\n+++ b/src/misc.py\n"
        "@@ -1,3 +1,4 @@\n"
        " def process(data):\n"
        "+    # Security hardening\n"
        "     return data\n"
    ),
}

# Complexity weight distribution
COMPLEXITY_WEIGHTS = [
    ("single_file", 50),
    ("multi_file", 25),
    ("dependency", 15),
    ("config", 10),
]
_COMPLEXITY_CHOICES = [c for c, w in COMPLEXITY_WEIGHTS for _ in range(w)]

# Files per complexity tag
_FILES_BY_COMPLEXITY: dict[str, dict[str, list[str]]] = {
    "python": {
        "single_file": ["src/auth.py"],
        "multi_file": ["src/auth.py", "src/db.py"],
        "dependency": ["requirements.txt"],
        "config": ["config.yaml"],
    },
    "javascript": {
        "single_file": ["src/app.js"],
        "multi_file": ["src/app.js", "src/utils.js"],
        "dependency": ["package.json"],
        "config": [".env"],
    },
    "go": {
        "single_file": ["pkg/handler.go"],
        "multi_file": ["pkg/handler.go", "pkg/middleware.go"],
        "dependency": ["go.mod"],
        "config": ["config.yaml"],
    },
    "java": {
        "single_file": ["src/main/java/App.java"],
        "multi_file": ["src/main/java/App.java", "src/main/java/Service.java"],
        "dependency": ["pom.xml"],
        "config": ["src/main/resources/application.properties"],
    },
}

CWE_SEVERITY_MAP = {
    "CWE-89": "critical",
    "CWE-78": "critical",
    "CWE-502": "critical",
    "CWE-918": "high",
    "CWE-79": "high",
    "CWE-22": "high",
    "CWE-798": "high",
    "CWE-284": "high",
    "CWE-1104": "medium",
    "CWE-20": "medium",
    "CWE-327": "medium",
    "OTHER": "low",
}


def generate_one(index: int, rng: random.Random) -> BenchmarkExample:
    """Generate a single realistic synthetic BenchmarkExample."""
    repo = rng.choice(MOCK_REPOS)
    repo_url: str = repo["repo_url"]
    language: str = repo["language"]
    repo_name = "/".join(repo_url.rstrip("/").split("/")[-2:])

    vuln_type = rng.choice(list(VulnType))
    cwe = vuln_type.value
    severity_str = CWE_SEVERITY_MAP.get(cwe, "low")
    severity = VulnSeverity(severity_str)

    complexity_tag = rng.choice(_COMPLEXITY_CHOICES)
    lang_files = _FILES_BY_COMPLEXITY.get(language, _FILES_BY_COMPLEXITY["python"])
    vulnerable_files = lang_files[complexity_tag]

    diff = MOCK_DIFFS.get(vuln_type, MOCK_DIFFS[VulnType.OTHER])

    merge_status = "merged" if rng.random() < 0.8 else "rejected"
    confidence = round(0.7 + rng.random() * 0.3, 3)

    # Diff stats
    lines_added = rng.randint(1, 10)
    lines_removed = rng.randint(0, 5)

    return BenchmarkExample(
        id=f"MOCK-{index:04d}",
        repo_url=repo_url,
        repo_name=repo_name,
        language=language,
        vulnerable_files=vulnerable_files,
        vuln_type=vuln_type,
        severity=severity,
        ground_truth_diff=diff,
        merge_status=merge_status,
        complexity_tag=complexity_tag,
        negative=False,
        pr_url=f"{repo_url}/pull/{index}",
        classification_confidence=confidence,
        metadata={
            "files_in_repo": rng.randint(5, 200),
            "diff_lines_added": lines_added,
            "diff_lines_removed": lines_removed,
            "source_format": "mock",
        },
    )


def _example_to_ingest_dict(ex: BenchmarkExample) -> dict:
    """Convert a BenchmarkExample to a dict in the format FileIngester expects."""
    return {
        "repo_url": ex.repo_url,
        "diff": ex.ground_truth_diff,
        "pr_title": f"Fix {ex.vuln_type} in {ex.repo_name}",
        "pr_body": f"Security fix for {ex.vuln_type}",
        "merge_status": ex.merge_status,
        "pr_url": ex.pr_url,
        "vuln_type": ex.vuln_type,
        "language": ex.language,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic PR data for benchmarking."
    )
    parser.add_argument("--count", type=int, default=500, help="Number of examples")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/mock_prs.json"),
        help="Output file path (.json or .csv)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    examples = [generate_one(i + 1, rng) for i in range(args.count)]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = [_example_to_ingest_dict(ex) for ex in examples]

    if args.format == "json":
        args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2))
        print(f"Wrote {len(rows)} records to {args.output.with_suffix('.json')}")
    else:
        out_path = args.output.with_suffix(".csv")
        if rows:
            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f"Wrote {len(rows)} records to {out_path}")


if __name__ == "__main__":
    main()
