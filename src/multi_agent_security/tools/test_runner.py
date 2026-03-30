import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

from pydantic import BaseModel

from multi_agent_security.types import Patch

logger = logging.getLogger(__name__)

_TEST_COMMANDS: dict[str, list[str]] = {
    "python": [sys.executable, "-m", "pytest", "--tb=short", "-q"],
    "javascript": ["npm", "test", "--", "--watchAll=false"],
    "go": ["go", "test", "./..."],
    "java": ["mvn", "test", "-q"],
}

_MAX_OUTPUT_CHARS = 2000


class TestResult(BaseModel):
    passed: bool
    output: str  # stdout + stderr, truncated to _MAX_OUTPUT_CHARS
    tests_run: int
    tests_failed: int
    error: Optional[str] = None


async def run_tests_on_patched_code(
    repo_path: str,
    patch: Patch,
    language: str,
    timeout_seconds: int = 60,
) -> TestResult:
    """Run the repo's test suite against a patched file.

    1. Copies the repo to a temp directory.
    2. Applies the patch (writes patched_code to the target file).
    3. Runs the appropriate test command for the language.
    4. Returns a TestResult with pass/fail status and output.
    """
    return await asyncio.to_thread(
        _run_tests_sync, repo_path, patch, language, timeout_seconds
    )


def _run_tests_sync(
    repo_path: str,
    patch: Patch,
    language: str,
    timeout_seconds: int,
) -> TestResult:
    cmd = _TEST_COMMANDS.get(language.lower())
    if cmd is None:
        return TestResult(
            passed=False,
            output="",
            tests_run=0,
            tests_failed=0,
            error=f"Unsupported language for test runner: {language!r}",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = os.path.join(tmp_dir, "repo")

        # Copy repo, skipping .git to save space
        shutil.copytree(
            repo_path,
            dest,
            ignore=shutil.ignore_patterns(".git"),
        )

        # Apply patch: write patched_code to the target file
        target_file = os.path.join(dest, patch.file_path)
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(patch.patched_code)

        # Run test suite
        try:
            result = subprocess.run(
                cmd,
                cwd=dest,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                passed=False,
                output="",
                tests_run=0,
                tests_failed=0,
                error=f"Test suite timed out after {timeout_seconds}s",
            )
        except Exception as exc:
            return TestResult(
                passed=False,
                output="",
                tests_run=0,
                tests_failed=0,
                error=str(exc),
            )

        combined = result.stdout + result.stderr
        tests_run, tests_failed = _parse_test_counts(combined, language.lower())
        output = combined[:_MAX_OUTPUT_CHARS]

        return TestResult(
            passed=result.returncode == 0,
            output=output,
            tests_run=tests_run,
            tests_failed=tests_failed,
        )


def _parse_test_counts(output: str, language: str) -> tuple[int, int]:
    """Extract (tests_run, tests_failed) from test runner output."""
    if language == "python":
        # pytest: "5 passed, 2 failed" or "7 passed"
        passed_m = re.search(r"(\d+) passed", output)
        failed_m = re.search(r"(\d+) failed", output)
        passed = int(passed_m.group(1)) if passed_m else 0
        failed = int(failed_m.group(1)) if failed_m else 0
        return passed + failed, failed

    elif language == "javascript":
        # Jest: "Tests: 2 failed, 5 passed, 7 total"
        total_m = re.search(r"(\d+) total", output)
        failed_m = re.search(r"(\d+) failed", output)
        total = int(total_m.group(1)) if total_m else 0
        failed = int(failed_m.group(1)) if failed_m else 0
        return total, failed

    elif language == "go":
        # "--- FAIL: TestFoo" per failure; "ok  github.com/..." per passing package
        failed = len(re.findall(r"^--- FAIL:", output, re.MULTILINE))
        passed_pkgs = len(re.findall(r"^ok\s", output, re.MULTILINE))
        return passed_pkgs + failed, failed

    elif language == "java":
        # Maven: "Tests run: 7, Failures: 2, Errors: 0"
        run_m = re.search(r"Tests run:\s*(\d+)", output)
        fail_m = re.search(r"Failures:\s*(\d+)", output)
        run = int(run_m.group(1)) if run_m else 0
        failed = int(fail_m.group(1)) if fail_m else 0
        return run, failed

    return 0, 0
