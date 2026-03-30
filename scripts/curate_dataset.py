"""Benchmark dataset curation pipeline.

Ingests raw security-fix PR data from three sources (CSV/JSON file, GitHub API,
local git repos), classifies vulnerabilities, tags complexity, generates negative
examples, splits into dev/test/hard sets, and writes structured JSON output.

Usage:
    python scripts/curate_dataset.py --input data/raw/prs.json --output data/ --seed 42
    python scripts/curate_dataset.py --github-query "security fix" --github-token $TOKEN
    python scripts/curate_dataset.py --repos-dir /path/to/clones
"""

import argparse
import asyncio
import csv
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

# Allow importing from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from multi_agent_security.tools.repo_cloner import RepoCloner
from multi_agent_security.types import BenchmarkExample, VulnSeverity, VulnType

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("masr.curate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CWE_SEVERITY_MAP: dict[str, str] = {
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

# Ordered so more specific patterns come first
KEYWORD_CWE_MAP: dict[str, str] = {
    "sql injection": "CWE-89",
    "sqli": "CWE-89",
    "execute(": "CWE-89",
    "cursor.execute": "CWE-89",
    "cross-site scripting": "CWE-79",
    "xss": "CWE-79",
    "innerhtml": "CWE-79",
    "document.write": "CWE-79",
    "path traversal": "CWE-22",
    "directory traversal": "CWE-22",
    "../": "CWE-22",
    "command injection": "CWE-78",
    "os.system": "CWE-78",
    "subprocess.call": "CWE-78",
    "shell=true": "CWE-78",
    "pickle.loads": "CWE-502",
    "yaml.load(": "CWE-502",
    "deserializ": "CWE-502",
    "insecure deserialization": "CWE-502",
    "server-side request forgery": "CWE-918",
    "ssrf": "CWE-918",
    "hardcoded credential": "CWE-798",
    "hardcoded": "CWE-798",
    "password =": "CWE-798",
    "api_key =": "CWE-798",
    "secret =": "CWE-798",
    "broken access control": "CWE-284",
    "access control": "CWE-284",
    "authorization": "CWE-284",
    "privilege escalat": "CWE-284",
    "weak cipher": "CWE-327",
    "md5(": "CWE-327",
    "sha1(": "CWE-327",
    "des ": "CWE-327",
    "crypto weakness": "CWE-327",
    "input validation": "CWE-20",
    "sanitiz": "CWE-20",
    "improper validation": "CWE-20",
    "vulnerable dependency": "CWE-1104",
    "cve-": "CWE-1104",
}

DEPENDENCY_FILES = frozenset(
    {
        "requirements.txt",
        "requirements-dev.txt",
        "package.json",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "Pipfile",
        "poetry.lock",
        "setup.cfg",
        "Gemfile",
        "Cargo.toml",
    }
)

CONFIG_EXTENSIONS = frozenset(
    {".yaml", ".yml", ".json", ".ini", ".toml", ".cfg", ".conf"}
)

# Dotfiles with no extension that are config files
CONFIG_FILENAMES = frozenset(
    {".env", ".env.example", ".env.local", ".gitconfig", ".npmrc", ".editorconfig"}
)

EXTENSION_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "javascript",
    ".jsx": "javascript",
    ".tsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".rs": "rust",
}

SPLIT_SIZES = {"dev": 100, "test": 300, "hard": 50}
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Intermediate raw PR dataclass (transient — not persisted)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RawPR:
    repo_url: str
    diff: str
    pr_title: str
    pr_body: str
    merge_status: str  # "merged" | "rejected"
    pr_url: Optional[str] = None
    provided_cwe: Optional[str] = None
    language: Optional[str] = None


# ---------------------------------------------------------------------------
# Pure helper functions (no I/O — unit-testable)
# ---------------------------------------------------------------------------


def classify_by_keywords(text: str) -> tuple[str, float]:
    """Match lowercased text against KEYWORD_CWE_MAP.

    Returns (cwe_string, confidence).  Keyword match → 0.7, no match → 0.5.
    """
    lower = text.lower()
    for keyword, cwe in KEYWORD_CWE_MAP.items():
        if keyword in lower:
            return cwe, 0.7
    return "OTHER", 0.5


def _is_config_file(path: str) -> bool:
    """Return True if the file is a configuration file."""
    name = Path(path).name
    suffix = Path(path).suffix.lower()
    return name in CONFIG_FILENAMES or suffix in CONFIG_EXTENSIONS


def tag_complexity(changed_files: list[str]) -> str:
    """Return a complexity tag based on which files were changed.

    Priority: dependency > config (all files) > single_file > multi_file
    """
    names = [Path(f).name for f in changed_files]
    if any(n in DEPENDENCY_FILES for n in names):
        return "dependency"
    if changed_files and all(_is_config_file(f) for f in changed_files):
        return "config"
    if len(changed_files) == 1:
        return "single_file"
    return "multi_file"


def parse_diff_stats(diff_text: str) -> tuple[int, int, list[str]]:
    """Parse a unified diff and return (lines_added, lines_removed, changed_files).

    Changed files are extracted from '+++ b/<path>' headers.
    """
    added = 0
    removed = 0
    files: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:])
        elif line.startswith("--- a/") or line.startswith("+++ b/"):
            continue
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed, files


def assign_severity(cwe: str) -> str:
    """Map a CWE string to a severity label."""
    return CWE_SEVERITY_MAP.get(cwe, "low")


def make_id(index: int) -> str:
    return f"BENCH-{index:04d}"


def extract_repo_name(repo_url: str) -> str:
    """Return 'owner/repo' from a GitHub URL."""
    parts = repo_url.rstrip("/").split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1] if parts else repo_url


def detect_language(file_paths: list[str]) -> str:
    """Infer the primary language from a list of file paths using extension plurality."""
    counts: dict[str, int] = defaultdict(int)
    for path in file_paths:
        ext = Path(path).suffix.lower()
        lang = EXTENSION_LANGUAGE.get(ext)
        if lang:
            counts[lang] += 1
    if not counts:
        return "unknown"
    return max(counts, key=lambda k: counts[k])


# ---------------------------------------------------------------------------
# Format A: CSV / JSON file ingestion
# ---------------------------------------------------------------------------


class FileIngester:
    """Ingest PR data from a CSV or JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def ingest(self) -> list[RawPR]:
        suffix = self.path.suffix.lower()
        if suffix == ".json":
            return self._ingest_json()
        if suffix == ".csv":
            return self._ingest_csv()
        raise ValueError(f"Unsupported file format: {suffix}. Use .json or .csv")

    def _row_to_raw_pr(self, row: dict[str, Any]) -> Optional[RawPR]:
        """Convert a normalised dict row to a RawPR, returning None to skip."""
        # Normalise keys
        row = {k.lower().strip(): v for k, v in row.items()}

        repo_url = row.get("repo_url", "").strip()
        diff = (row.get("diff") or row.get("diff_patch") or "").strip()

        if not repo_url or not diff:
            logger.warning("Skipping row — missing repo_url or diff")
            return None

        merge_status = (
            row.get("merge_status") or row.get("status") or "unknown"
        ).strip()
        # Normalise: "closed" without merge → "rejected"
        if merge_status not in ("merged", "rejected"):
            merge_status = "merged" if merge_status == "merged" else "rejected"

        return RawPR(
            repo_url=repo_url,
            diff=diff,
            pr_title=(row.get("title") or row.get("pr_title") or "").strip(),
            pr_body=(row.get("body") or row.get("pr_body") or "").strip(),
            merge_status=merge_status,
            pr_url=(row.get("pr_url") or "").strip() or None,
            provided_cwe=(
                row.get("vuln_type") or row.get("cwe") or ""
            ).strip() or None,
            language=(row.get("language") or "").strip() or None,
        )

    def _ingest_json(self) -> list[RawPR]:
        data = json.loads(self.path.read_text())
        if isinstance(data, dict):
            data = [data]
        results = []
        for row in data:
            pr = self._row_to_raw_pr(row)
            if pr is not None:
                results.append(pr)
        logger.info("FileIngester: loaded %d PRs from %s", len(results), self.path)
        return results

    def _ingest_csv(self) -> list[RawPR]:
        results = []
        with self.path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pr = self._row_to_raw_pr(dict(row))
                if pr is not None:
                    results.append(pr)
        logger.info("FileIngester: loaded %d PRs from %s", len(results), self.path)
        return results


# ---------------------------------------------------------------------------
# Format B: GitHub API ingestion
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    """Raised when the GitHub API rate limit is exhausted."""


class GitHubIngester:
    """Fetch PR data from the GitHub REST API."""

    API_BASE = "https://api.github.com"

    def __init__(
        self,
        token: Optional[str],
        cache_dir: Path,
        max_results: int = 500,
    ) -> None:
        self.token = token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_results = max_results
        self._repo_language_cache: dict[str, str] = {}

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            h["Authorization"] = f"token {self.token}"
        return h

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[Any]:
        p = self._cache_path(key)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def _save_cache(self, key: str, data: Any) -> None:
        self._cache_path(key).write_text(json.dumps(data))

    async def _fetch_with_rate_limit(
        self, client: httpx.AsyncClient, url: str, **kwargs: Any
    ) -> httpx.Response:
        response = await client.get(url, headers=self._headers(), **kwargs)
        if response.status_code in (403, 429):
            reset_ts = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(1, reset_ts - int(time.time()))
            logger.warning("Rate limited. Waiting %ds before retry.", wait)
            raise RateLimitError(f"Rate limited, reset in {wait}s")
        response.raise_for_status()
        return response

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(RateLimitError),
    )
    async def _get(self, client: httpx.AsyncClient, url: str, **kwargs: Any) -> Any:
        resp = await self._fetch_with_rate_limit(client, url, **kwargs)
        return resp.json()

    async def _get_repo_language(
        self, client: httpx.AsyncClient, owner: str, repo: str
    ) -> str:
        key = f"{owner}_{repo}"
        if key in self._repo_language_cache:
            return self._repo_language_cache[key]
        cached = self._load_cache(f"repo_{key}")
        if cached:
            lang = (cached.get("language") or "unknown").lower()
            self._repo_language_cache[key] = lang
            return lang
        try:
            data = await self._get(client, f"{self.API_BASE}/repos/{owner}/{repo}")
            self._save_cache(f"repo_{key}", data)
            lang = (data.get("language") or "unknown").lower()
        except Exception:
            lang = "unknown"
        self._repo_language_cache[key] = lang
        return lang

    async def ingest(self, query: str) -> list[RawPR]:
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        results: list[RawPR] = []

        async with httpx.AsyncClient(timeout=30) as client:
            page = 1
            while len(results) < self.max_results:
                cache_key = f"search_{query_hash}_page{page}"
                data = self._load_cache(cache_key)
                if data is None:
                    try:
                        data = await self._get(
                            client,
                            f"{self.API_BASE}/search/issues",
                            params={
                                "q": f"{query} is:pr is:merged",
                                "per_page": 100,
                                "page": page,
                            },
                        )
                        self._save_cache(cache_key, data)
                    except Exception as e:
                        logger.error("GitHub search failed: %s", e)
                        break

                items = data.get("items", [])
                if not items:
                    break

                for item in items:
                    if len(results) >= self.max_results:
                        break
                    pr = await self._fetch_pr(client, item)
                    if pr:
                        results.append(pr)

                page += 1

        logger.info("GitHubIngester: fetched %d PRs", len(results))
        return results

    async def _fetch_pr(
        self, client: httpx.AsyncClient, item: dict[str, Any]
    ) -> Optional[RawPR]:
        html_url: str = item.get("html_url", "")
        # html_url: https://github.com/owner/repo/pull/123
        parts = html_url.rstrip("/").split("/")
        if len(parts) < 7 or parts[-2] != "pull":
            return None
        owner, repo, pr_number = parts[-4], parts[-3], parts[-1]

        cache_key = f"pr_{owner}_{repo}_{pr_number}"
        cached_files = self._load_cache(f"{cache_key}_files")

        if cached_files is None:
            try:
                cached_files = await self._get(
                    client,
                    f"{self.API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/files",
                )
                self._save_cache(f"{cache_key}_files", cached_files)
            except Exception as e:
                logger.warning("Could not fetch files for %s: %s", html_url, e)
                cached_files = []

        # Build a synthetic unified diff from patch fragments
        diff_parts = []
        changed_files = []
        for f in cached_files:
            filename = f.get("filename", "")
            patch = f.get("patch", "")
            changed_files.append(filename)
            if patch:
                diff_parts.append(
                    f"--- a/{filename}\n+++ b/{filename}\n{patch}"
                )
        diff = "\n".join(diff_parts)

        language = await self._get_repo_language(client, owner, repo)
        if language == "unknown" and changed_files:
            language = detect_language(changed_files)

        return RawPR(
            repo_url=f"https://github.com/{owner}/{repo}",
            diff=diff,
            pr_title=item.get("title", ""),
            pr_body=item.get("body", "") or "",
            merge_status="merged",
            pr_url=html_url,
            language=language,
        )


# ---------------------------------------------------------------------------
# Format C: Local git repo ingestion
# ---------------------------------------------------------------------------


class LocalGitIngester:
    """Extract diffs from locally cloned git repositories."""

    def __init__(self, cloner: RepoCloner) -> None:
        self.cloner = cloner

    def _get_remote_url(self, repo_path: Path) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.decode().strip()
        return str(repo_path)

    def ingest(self, repo_paths: list[Path]) -> list[RawPR]:
        results = []
        for path in repo_paths:
            path = Path(path)
            if not path.exists():
                logger.warning("Repo path does not exist: %s", path)
                continue
            diff = self.cloner.get_diff(path, "HEAD~1", "HEAD")
            if not diff.strip():
                logger.debug("Empty diff for %s, skipping", path)
                continue
            repo_url = self._get_remote_url(path)
            _, _, changed_files = parse_diff_stats(diff)
            language = detect_language(changed_files)
            results.append(
                RawPR(
                    repo_url=repo_url,
                    diff=diff,
                    pr_title="",
                    pr_body="",
                    merge_status="merged",
                    language=language,
                )
            )
        logger.info("LocalGitIngester: loaded %d diffs from %d repos", len(results), len(repo_paths))
        return results


# ---------------------------------------------------------------------------
# LLM classification (optional)
# ---------------------------------------------------------------------------


async def classify_with_llm(
    diff: str,
    title: str,
    body: str,
    llm_client: Any,
) -> tuple[str, float]:
    """Classify vulnerability type using an LLM, falling back to keywords."""
    prompt = (
        "You are a security vulnerability classifier. "
        "Given the following code diff from a security fix PR, classify the vulnerability "
        "into exactly one CWE category.\n\n"
        "Respond with JSON only:\n"
        '{"vuln_type": "CWE-XXX", "confidence": 0.0-1.0, "reasoning": "..."}\n\n'
        f"PR Title: {title}\n\nDiff:\n{diff[:4000]}"
    )
    try:
        response = await llm_client.complete(
            system="You are a security expert. Respond with valid JSON only.",
            user=prompt,
        )
        data = json.loads(response.content)
        cwe = data.get("vuln_type", "OTHER")
        confidence = float(data.get("confidence", 0.8))
        # Validate the CWE is one we know
        if cwe not in CWE_SEVERITY_MAP:
            cwe = "OTHER"
        return cwe, confidence
    except Exception as e:
        logger.debug("LLM classification failed (%s), falling back to keywords", e)
        return classify_by_keywords(f"{diff} {title} {body}")


# ---------------------------------------------------------------------------
# Negative example generation
# ---------------------------------------------------------------------------


def generate_negatives(
    positives: list[BenchmarkExample],
    rng: random.Random,
) -> list[BenchmarkExample]:
    """Generate ~10% negative examples by sampling from positives."""
    n = max(1, len(positives) // 10)
    sampled = rng.sample(positives, min(n, len(positives)))
    negatives = []
    for ex in sampled:
        data = ex.model_dump()
        data["negative"] = True
        data["id"] = ex.id + "-NEG"
        data["vuln_type"] = VulnType.OTHER
        data["severity"] = VulnSeverity.INFO
        data["classification_confidence"] = 1.0
        data.setdefault("metadata", {})["is_synthetic_negative"] = True
        negatives.append(BenchmarkExample.model_validate(data))
    return negatives


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------


def _stratified_sample(
    pool: list[BenchmarkExample],
    n: int,
    rng: random.Random,
    excluded_repos: set[str],
) -> tuple[list[BenchmarkExample], set[str]]:
    """Draw n examples from pool with stratification by (vuln_type, language).

    Excludes examples whose repo_name is in excluded_repos.
    Returns (selected, repo_names_used).
    """
    candidates = [e for e in pool if e.repo_name not in excluded_repos]
    if not candidates:
        logger.warning("No candidates available after repo exclusion")
        return [], set()

    # Group by stratum
    strata: dict[tuple[str, str], list[BenchmarkExample]] = defaultdict(list)
    for ex in candidates:
        strata[(ex.vuln_type, ex.language)].append(ex)

    total = len(candidates)
    selected: list[BenchmarkExample] = []
    repos_used: set[str] = set()

    # Allocate proportionally to each stratum
    remaining_n = n
    strata_list = list(strata.items())
    rng.shuffle(strata_list)

    for i, (_, group) in enumerate(strata_list):
        if remaining_n <= 0:
            break
        proportion = len(group) / total
        is_last = i == len(strata_list) - 1
        alloc = remaining_n if is_last else max(1, round(proportion * n))
        alloc = min(alloc, len(group), remaining_n)
        rng.shuffle(group)
        chosen = group[:alloc]
        selected.extend(chosen)
        repos_used.update(e.repo_name for e in chosen)
        remaining_n -= len(chosen)

    if len(selected) < n:
        logger.warning(
            "Could only allocate %d/%d examples (pool too small)", len(selected), n
        )

    return selected, repos_used


def split_dataset(
    examples: list[BenchmarkExample],
    seed: int = RANDOM_SEED,
) -> dict[str, list[BenchmarkExample]]:
    """Split examples into dev / test / hard sets.

    Hard set is selected first (multi_file/dependency + critical/high severity).
    Dev and test are stratified with no repo overlap between them.
    Hard shares repos with dev/test (only item IDs are deduplicated across splits).
    All examples are re-indexed with sequential BENCH-XXXX IDs.
    """
    rng = random.Random(seed)
    pool = list(examples)

    # --- Hard set (selected by criteria; repos may overlap with dev/test) ---
    hard_candidates = [
        e
        for e in pool
        if e.complexity_tag in ("multi_file", "dependency")
        and e.severity in (VulnSeverity.CRITICAL, VulnSeverity.HIGH)
    ]
    rng.shuffle(hard_candidates)
    hard = hard_candidates[: SPLIT_SIZES["hard"]]
    hard_ids = {e.id for e in hard}

    # Remaining pool for dev/test: exclude hard item IDs only (not repos)
    dev_test_pool = [e for e in pool if e.id not in hard_ids]

    # --- Dev set (no repo exclusion at this stage) ---
    dev, dev_repos = _stratified_sample(dev_test_pool, SPLIT_SIZES["dev"], rng, set())
    dev_ids = {e.id for e in dev}
    dev_test_pool = [e for e in dev_test_pool if e.id not in dev_ids]

    # --- Test set (no overlap with dev repos) ---
    test, _ = _stratified_sample(dev_test_pool, SPLIT_SIZES["test"], rng, dev_repos)

    # Re-index all three splits with global sequential IDs
    # dev: 0001-0100, test: 0101-0400, hard: 0401-0450
    def reindex(lst: list[BenchmarkExample], start: int) -> list[BenchmarkExample]:
        out = []
        for i, ex in enumerate(lst):
            data = ex.model_dump()
            data["id"] = make_id(start + i)
            out.append(BenchmarkExample.model_validate(data))
        return out

    return {
        "dev": reindex(dev, 1),
        "test": reindex(test, 101),
        "hard": reindex(hard, 401),
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _build_manifest(split_name: str, examples: list[BenchmarkExample], seed: int) -> dict[str, Any]:
    positives = [e for e in examples if not e.negative]
    negatives = [e for e in examples if e.negative]

    by_vuln_type: dict[str, int] = defaultdict(int)
    by_complexity: dict[str, int] = defaultdict(int)
    by_language: dict[str, int] = defaultdict(int)
    for e in examples:
        by_vuln_type[e.vuln_type] += 1
        by_complexity[e.complexity_tag] += 1
        by_language[e.language] += 1

    return {
        "split": split_name,
        "total_examples": len(examples),
        "positive_examples": len(positives),
        "negative_examples": len(negatives),
        "by_vuln_type": dict(by_vuln_type),
        "by_complexity": dict(by_complexity),
        "by_language": dict(by_language),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": seed,
    }


def write_output(
    splits: dict[str, list[BenchmarkExample]],
    base_dir: Path,
    seed: int = RANDOM_SEED,
) -> None:
    """Write each split as individual JSON files + a manifest."""
    for split_name, examples in splits.items():
        split_dir = base_dir / "benchmark" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for ex in examples:
            out_path = split_dir / f"{ex.id}.json"
            out_path.write_text(ex.model_dump_json(indent=2))

        manifest = _build_manifest(split_name, examples, seed)
        (split_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        logger.info(
            "Wrote %d examples to %s/", len(examples), split_dir
        )


# ---------------------------------------------------------------------------
# Raw PR → BenchmarkExample conversion
# ---------------------------------------------------------------------------


def _cwe_to_vuln_type(cwe: str) -> VulnType:
    """Map a CWE string like 'CWE-89' to a VulnType enum value."""
    for member in VulnType:
        if member.value == cwe:
            return member
    return VulnType.OTHER


def raw_pr_to_example(
    raw: RawPR,
    cwe: str,
    confidence: float,
    index: int,
    repo_file_count: int = 0,
) -> BenchmarkExample:
    """Convert a RawPR into a BenchmarkExample."""
    lines_added, lines_removed, changed_files = parse_diff_stats(raw.diff)
    complexity = tag_complexity(changed_files)
    severity_str = assign_severity(cwe)
    language = raw.language or detect_language(changed_files)
    vuln_type = _cwe_to_vuln_type(cwe)
    severity = VulnSeverity(severity_str)
    repo_name = extract_repo_name(raw.repo_url)

    return BenchmarkExample(
        id=make_id(index),
        repo_url=raw.repo_url,
        repo_name=repo_name,
        language=language,
        vulnerable_files=changed_files or ["unknown"],
        vuln_type=vuln_type,
        severity=severity,
        ground_truth_diff=raw.diff,
        merge_status=raw.merge_status,
        complexity_tag=complexity,
        negative=False,
        pr_url=raw.pr_url,
        classification_confidence=confidence,
        metadata={
            "files_in_repo": repo_file_count,
            "diff_lines_added": lines_added,
            "diff_lines_removed": lines_removed,
            "source_format": "file" if raw.pr_url is None else "github_api",
        },
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_summary(splits: dict[str, list[BenchmarkExample]]) -> None:
    print("\n=== Dataset Summary ===")
    for name, examples in splits.items():
        from collections import Counter

        vtypes = Counter(e.vuln_type for e in examples)
        print(
            f"  {name:8s}: {len(examples):4d} examples  "
            f"| vuln_types: {dict(vtypes.most_common(3))}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate benchmark dataset from security-fix PR data."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Path to CSV or JSON file (Format A)")
    source.add_argument("--github-query", help="GitHub search query string (Format B)")
    source.add_argument("--repos-dir", type=Path, help="Directory of local git repos (Format C)")

    parser.add_argument("--output", type=Path, default=Path("data"), help="Output root directory")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--use-llm-classification", action="store_true")
    parser.add_argument("--llm-config", type=Path, help="Path to YAML config for LLM")
    parser.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/cache"))
    parser.add_argument("--max-results", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing files")

    args = parser.parse_args()

    # Ingest
    raw_prs: list[RawPR] = []

    if args.input:
        raw_prs = FileIngester(args.input).ingest()
    elif args.github_query:
        ingester = GitHubIngester(
            token=args.github_token,
            cache_dir=args.cache_dir,
            max_results=args.max_results,
        )
        raw_prs = asyncio.run(ingester.ingest(args.github_query))
    elif args.repos_dir:
        cloner = RepoCloner(cache_dir=args.cache_dir / "clones")
        local_ingester = LocalGitIngester(cloner=cloner)
        repo_paths = [p for p in Path(args.repos_dir).iterdir() if p.is_dir()]
        raw_prs = local_ingester.ingest(repo_paths)

    if not raw_prs:
        logger.error("No PR data ingested. Exiting.")
        sys.exit(1)

    logger.info("Ingested %d raw PRs", len(raw_prs))

    # LLM client setup (optional)
    llm_client = None
    if args.use_llm_classification and args.llm_config:
        try:
            from multi_agent_security.config import load_config
            from multi_agent_security.llm_client import LLMClient

            config = load_config(args.llm_config)
            llm_client = LLMClient(config)
        except Exception as e:
            logger.warning("Could not initialise LLM client: %s. Falling back to keywords.", e)

    # Classify, tag, build BenchmarkExamples
    examples: list[BenchmarkExample] = []

    async def _build_examples() -> list[BenchmarkExample]:
        out = []
        for i, raw in enumerate(raw_prs):
            search_text = f"{raw.diff} {raw.pr_title} {raw.pr_body}"
            if raw.provided_cwe:
                cwe, confidence = raw.provided_cwe, 1.0
            elif args.use_llm_classification and llm_client:
                cwe, confidence = await classify_with_llm(
                    raw.diff, raw.pr_title, raw.pr_body, llm_client
                )
            else:
                cwe, confidence = classify_by_keywords(search_text)
            out.append(raw_pr_to_example(raw, cwe, confidence, i + 1))
        return out

    examples = asyncio.run(_build_examples())
    logger.info("Built %d BenchmarkExamples", len(examples))

    # Negative examples
    rng = random.Random(args.seed)
    negatives = generate_negatives(examples, rng)
    all_examples = examples + negatives
    logger.info("Added %d negative examples (total: %d)", len(negatives), len(all_examples))

    # Split
    splits = split_dataset(all_examples, seed=args.seed)

    _print_summary(splits)

    if args.dry_run:
        logger.info("--dry-run set; skipping file writes.")
        return

    write_output(splits, args.output, seed=args.seed)
    logger.info("Done. Output written to %s/benchmark/", args.output)


if __name__ == "__main__":
    main()
