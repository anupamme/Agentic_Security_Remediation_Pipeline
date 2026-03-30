"""Unit tests for the dataset curation pipeline.

Covers: keyword classification, complexity tagging, dataset splitting,
and negative example generation.
"""

import random
import sys
from pathlib import Path

import pytest

# Import from scripts without requiring package installation
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from curate_dataset import (
    CWE_SEVERITY_MAP,
    KEYWORD_CWE_MAP,
    assign_severity,
    classify_by_keywords,
    generate_negatives,
    make_id,
    parse_diff_stats,
    split_dataset,
    tag_complexity,
)
from generate_mock_data import generate_one
from multi_agent_security.types import BenchmarkExample, VulnSeverity, VulnType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_example(
    index: int,
    repo_name: str = "org/repo",
    vuln_type: VulnType = VulnType.SQL_INJECTION,
    severity: VulnSeverity = VulnSeverity.CRITICAL,
    complexity_tag: str = "single_file",
    language: str = "python",
) -> BenchmarkExample:
    return BenchmarkExample(
        id=make_id(index),
        repo_url=f"https://github.com/{repo_name}",
        repo_name=repo_name,
        language=language,
        vulnerable_files=["src/auth.py"],
        vuln_type=vuln_type,
        severity=severity,
        ground_truth_diff="--- a/src/auth.py\n+++ b/src/auth.py\n@@ -1 +1 @@\n-bad\n+good\n",
        merge_status="merged",
        complexity_tag=complexity_tag,
    )


# ---------------------------------------------------------------------------
# 1. Keyword classification
# ---------------------------------------------------------------------------


def test_classify_sql_injection():
    cwe, confidence = classify_by_keywords("fixed sql injection in login form")
    assert cwe == "CWE-89"
    assert confidence == 0.7


def test_classify_xss():
    cwe, _ = classify_by_keywords("prevent xss via innerHTML escaping")
    assert cwe == "CWE-79"


def test_classify_ssrf():
    cwe, _ = classify_by_keywords("block ssrf from untrusted input URLs")
    assert cwe == "CWE-918"


def test_classify_fallback_to_other():
    cwe, confidence = classify_by_keywords("refactor login module for readability")
    assert cwe == "OTHER"
    assert confidence == 0.5


def test_classify_case_insensitive():
    cwe, _ = classify_by_keywords("SQL INJECTION vulnerability patched")
    assert cwe == "CWE-89"


def test_classify_confidence_values():
    _, hit_conf = classify_by_keywords("xss attack fixed")
    _, miss_conf = classify_by_keywords("upgrade to python 3.12")
    assert hit_conf == 0.7
    assert miss_conf == 0.5


# ---------------------------------------------------------------------------
# 2. Complexity tagging
# ---------------------------------------------------------------------------


def test_complexity_single_file():
    assert tag_complexity(["src/auth.py"]) == "single_file"


def test_complexity_multi_file():
    assert tag_complexity(["src/auth.py", "src/db.py"]) == "multi_file"


def test_complexity_dependency_requirements():
    # dependency takes priority even when mixed with code files
    assert tag_complexity(["requirements.txt", "src/auth.py"]) == "dependency"


def test_complexity_dependency_package_json():
    assert tag_complexity(["package.json"]) == "dependency"


def test_complexity_config_only():
    assert tag_complexity([".env", "config.yaml"]) == "config"


def test_complexity_config_mixed_with_code():
    # Config + code file → multi_file, NOT config
    assert tag_complexity([".env", "src/auth.py"]) == "multi_file"


# ---------------------------------------------------------------------------
# 3. Dataset splitting
# ---------------------------------------------------------------------------


def _make_pool(n: int, num_repos: int = 40) -> list[BenchmarkExample]:
    """Create n examples distributed across num_repos repos."""
    vuln_types = list(VulnType)
    examples = []
    for i in range(n):
        repo = f"org/repo-{i % num_repos}"
        # Ensure some hard-set candidates exist
        if i % 5 == 0:
            complexity = "multi_file"
            severity = VulnSeverity.CRITICAL
        else:
            complexity = "single_file"
            severity = VulnSeverity.MEDIUM
        vt = vuln_types[i % len(vuln_types)]
        examples.append(
            _make_example(
                i + 1,
                repo_name=repo,
                vuln_type=vt,
                severity=severity,
                complexity_tag=complexity,
            )
        )
    return examples


def test_split_no_dev_test_repo_overlap():
    pool = _make_pool(300, num_repos=40)
    splits = split_dataset(pool, seed=42)
    dev_repos = {e.repo_name for e in splits["dev"]}
    test_repos = {e.repo_name for e in splits["test"]}
    assert dev_repos.isdisjoint(test_repos), (
        f"Repo overlap: {dev_repos & test_repos}"
    )


def test_split_seed_reproducibility():
    pool = _make_pool(300)
    splits_a = split_dataset(pool, seed=42)
    splits_b = split_dataset(pool, seed=42)
    assert [e.id for e in splits_a["dev"]] == [e.id for e in splits_b["dev"]]
    assert [e.id for e in splits_a["test"]] == [e.id for e in splits_b["test"]]


def test_split_hard_set_criteria():
    pool = _make_pool(500)
    splits = split_dataset(pool, seed=42)
    for ex in splits["hard"]:
        assert ex.complexity_tag in ("multi_file", "dependency"), (
            f"{ex.id} has complexity_tag={ex.complexity_tag}"
        )
        assert ex.severity in (VulnSeverity.CRITICAL, VulnSeverity.HIGH), (
            f"{ex.id} has severity={ex.severity}"
        )


def test_split_respects_max_sizes():
    pool = _make_pool(500)
    splits = split_dataset(pool, seed=42)
    assert len(splits["dev"]) <= 100
    assert len(splits["test"]) <= 300
    assert len(splits["hard"]) <= 50


def test_split_ids_are_sequential():
    pool = _make_pool(300)
    splits = split_dataset(pool, seed=42)
    # dev IDs start at BENCH-0001
    first_dev = splits["dev"][0].id if splits["dev"] else None
    if first_dev:
        assert first_dev == "BENCH-0001"
    # test IDs start at BENCH-0101
    first_test = splits["test"][0].id if splits["test"] else None
    if first_test:
        assert first_test == "BENCH-0101"


# ---------------------------------------------------------------------------
# 4. Negative example generation
# ---------------------------------------------------------------------------


def test_negative_ratio():
    positives = [_make_example(i) for i in range(1, 101)]
    rng = random.Random(42)
    negatives = generate_negatives(positives, rng)
    assert len(negatives) == 10


def test_negatives_are_flagged():
    positives = [_make_example(i) for i in range(1, 51)]
    rng = random.Random(42)
    negatives = generate_negatives(positives, rng)
    assert all(e.negative for e in negatives)


def test_negative_vuln_type():
    positives = [_make_example(i) for i in range(1, 51)]
    rng = random.Random(42)
    negatives = generate_negatives(positives, rng)
    assert all(e.vuln_type == VulnType.OTHER for e in negatives)


def test_negative_ids_unique():
    positives = [_make_example(i) for i in range(1, 101)]
    rng = random.Random(42)
    negatives = generate_negatives(positives, rng)
    pos_ids = {e.id for e in positives}
    neg_ids = {e.id for e in negatives}
    assert pos_ids.isdisjoint(neg_ids), f"ID collision: {pos_ids & neg_ids}"


# ---------------------------------------------------------------------------
# 5. Misc helpers
# ---------------------------------------------------------------------------


def test_parse_diff_stats():
    diff = (
        "--- a/src/auth.py\n+++ b/src/auth.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-old line\n"
        "+new line\n"
        " context\n"
    )
    added, removed, files = parse_diff_stats(diff)
    assert added == 1
    assert removed == 1
    assert "src/auth.py" in files


def test_parse_diff_stats_multi_file():
    diff = (
        "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n+x\n"
        "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-y\n"
    )
    _, _, files = parse_diff_stats(diff)
    assert set(files) == {"a.py", "b.py"}


def test_assign_severity_known():
    assert assign_severity("CWE-89") == "critical"
    assert assign_severity("CWE-79") == "high"
    assert assign_severity("CWE-327") == "medium"


def test_assign_severity_unknown():
    assert assign_severity("CWE-9999") == "low"
    assert assign_severity("OTHER") == "low"


def test_make_id():
    assert make_id(1) == "BENCH-0001"
    assert make_id(999) == "BENCH-0999"
    assert make_id(1000) == "BENCH-1000"


# ---------------------------------------------------------------------------
# 6. Mock data generator
# ---------------------------------------------------------------------------


def test_generate_one_mock_is_valid():
    rng = random.Random(42)
    ex = generate_one(1, rng)
    # Round-trip through model validation
    validated = BenchmarkExample.model_validate(ex.model_dump())
    assert validated.id == "MOCK-0001"
    assert validated.classification_confidence >= 0.7
    assert validated.classification_confidence <= 1.0
    assert validated.metadata.get("source_format") == "mock"


def test_generate_mock_diversity():
    """Check that 100 mocks have at least 3 distinct vuln_types and 2 languages."""
    rng = random.Random(42)
    examples = [generate_one(i, rng) for i in range(1, 101)]
    vuln_types = {e.vuln_type for e in examples}
    languages = {e.language for e in examples}
    assert len(vuln_types) >= 3
    assert len(languages) >= 2
