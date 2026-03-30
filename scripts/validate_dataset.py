"""Validate a curated benchmark dataset.

Loads every example from the dev/test/hard splits, validates each against the
BenchmarkExample Pydantic model, checks repo overlap, stratification, diff
quality, and negative-example labelling.

Usage:
    python scripts/validate_dataset.py data/
    python scripts/validate_dataset.py data/ --strict   # exit 1 on any error
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pydantic import ValidationError

from multi_agent_security.types import BenchmarkExample, VulnType


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_split(split_dir: Path) -> tuple[list[BenchmarkExample], list[str]]:
    """Load all BenchmarkExample JSON files in split_dir (skip manifest.json).

    Returns (examples, validation_errors).  Errors are human-readable strings.
    """
    examples: list[BenchmarkExample] = []
    errors: list[str] = []

    for json_file in sorted(split_dir.glob("*.json")):
        if json_file.name == "manifest.json":
            continue
        text = json_file.read_text()
        try:
            ex = BenchmarkExample.model_validate_json(text)
            examples.append(ex)
        except ValidationError as e:
            errors.append(f"{json_file.name}: {e}")
        except Exception as e:
            errors.append(f"{json_file.name}: unexpected error — {e}")

    return examples, errors


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_no_repo_overlap(
    a: list[BenchmarkExample],
    b: list[BenchmarkExample],
    name_a: str,
    name_b: str,
) -> list[str]:
    """Return errors for any repo_name appearing in both splits."""
    repos_a = {e.repo_name for e in a}
    repos_b = {e.repo_name for e in b}
    overlap = repos_a & repos_b
    if not overlap:
        return []
    return [
        f"Repo overlap between {name_a} and {name_b}: {', '.join(sorted(overlap))}"
    ]


def check_stratification(
    examples: list[BenchmarkExample],
    reference: list[BenchmarkExample],
    split_name: str,
    tolerance: float = 0.05,
) -> list[str]:
    """Check that vuln_type proportions in 'examples' match 'reference' within ±tolerance."""
    if not reference:
        return []
    errors: list[str] = []
    ref_total = len(reference)
    split_total = len(examples)

    ref_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    for e in reference:
        ref_counts[e.vuln_type] += 1
    for e in examples:
        split_counts[e.vuln_type] += 1

    for vtype, ref_count in ref_counts.items():
        expected_proportion = ref_count / ref_total
        actual_proportion = split_counts.get(vtype, 0) / split_total if split_total else 0
        diff = abs(actual_proportion - expected_proportion)
        if diff > tolerance:
            errors.append(
                f"{split_name}: {vtype} proportion {actual_proportion:.2%} "
                f"vs reference {expected_proportion:.2%} "
                f"(delta {diff:.2%} > {tolerance:.0%} tolerance)"
            )
    return errors


def check_diffs(examples: list[BenchmarkExample]) -> list[str]:
    """Check that every diff is non-empty and structurally parseable."""
    errors: list[str] = []
    for ex in examples:
        diff = ex.ground_truth_diff.strip()
        if not diff:
            errors.append(f"{ex.id}: ground_truth_diff is empty")
            continue
        if "+++ b/" not in diff and "+++" not in diff:
            errors.append(
                f"{ex.id}: ground_truth_diff appears unparseable (no '+++ b/' header found)"
            )
    return errors


def check_negatives(examples: list[BenchmarkExample]) -> list[str]:
    """Check that negative examples are labelled with VulnType.OTHER."""
    errors: list[str] = []
    for ex in examples:
        if ex.negative and ex.vuln_type != VulnType.OTHER:
            errors.append(
                f"{ex.id}: negative=True but vuln_type={ex.vuln_type} (expected OTHER)"
            )
    return errors


def check_ids_unique(
    splits: dict[str, list[BenchmarkExample]],
) -> list[str]:
    """Check that IDs are unique within each split."""
    errors: list[str] = []
    for name, examples in splits.items():
        seen: dict[str, int] = defaultdict(int)
        for e in examples:
            seen[e.id] += 1
        dups = [id_ for id_, count in seen.items() if count > 1]
        if dups:
            errors.append(f"{name}: duplicate IDs: {', '.join(dups)}")
    return errors


# ---------------------------------------------------------------------------
# Summary table (stdlib only)
# ---------------------------------------------------------------------------


def print_summary_table(
    splits: dict[str, list[BenchmarkExample]],
    all_errors: list[str],
) -> None:
    print("\n" + "=" * 70)
    print(f"{'Split':<8}  {'Total':>6}  {'Positive':>8}  {'Negative':>8}  {'Errors':>6}")
    print("-" * 70)
    for name, examples in splits.items():
        pos = sum(1 for e in examples if not e.negative)
        neg = sum(1 for e in examples if e.negative)
        err_count = sum(1 for e in all_errors if e.startswith(name) or any(
            x.id in e for x in examples
        ))
        print(f"{name:<8}  {len(examples):>6}  {pos:>8}  {neg:>8}  {err_count:>6}")

    print("=" * 70)

    # vuln_type breakdown per split
    for name, examples in splits.items():
        counts: dict[str, int] = defaultdict(int)
        for e in examples:
            counts[e.vuln_type] += 1
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        row = "  ".join(f"{k}:{v}" for k, v in sorted_counts[:5])
        print(f"  {name}: {row}")

    print()
    if all_errors:
        print(f"ERRORS ({len(all_errors)}):")
        for err in all_errors:
            print(f"  ✗ {err}")
    else:
        print("All checks passed.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a curated benchmark dataset."
    )
    parser.add_argument("data_dir", type=Path, help="Root data directory (contains benchmark/)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any validation error is found",
    )
    args = parser.parse_args()

    benchmark_dir = args.data_dir / "benchmark"
    if not benchmark_dir.exists():
        print(f"ERROR: {benchmark_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    splits: dict[str, list[BenchmarkExample]] = {}
    all_errors: list[str] = []

    for split_name in ("dev", "test", "hard"):
        split_dir = benchmark_dir / split_name
        if not split_dir.exists():
            all_errors.append(f"Split directory not found: {split_dir}")
            splits[split_name] = []
            continue
        examples, parse_errors = load_split(split_dir)
        splits[split_name] = examples
        all_errors.extend(parse_errors)

    # Check overlap
    if "dev" in splits and "test" in splits:
        all_errors.extend(
            check_no_repo_overlap(splits["dev"], splits["test"], "dev", "test")
        )

    # Check stratification (dev and test against combined pool)
    combined = splits.get("dev", []) + splits.get("test", [])
    if combined:
        all_errors.extend(check_stratification(splits.get("dev", []), combined, "dev"))
        all_errors.extend(check_stratification(splits.get("test", []), combined, "test"))

    # Check diffs and negatives for all splits
    for name, examples in splits.items():
        all_errors.extend(check_diffs(examples))
        all_errors.extend(check_negatives(examples))

    # Check ID uniqueness
    all_errors.extend(check_ids_unique(splits))

    print_summary_table(splits, all_errors)

    if args.strict and all_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
