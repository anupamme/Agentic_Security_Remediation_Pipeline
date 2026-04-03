"""Utility for cloning git repositories to a local cache directory."""

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

_DEFAULT_CLONE_TIMEOUT = 300  # seconds; shallow clone of large repos can be slow

logger = logging.getLogger("masr")


class RepoCloner:
    """Shallow-clones git repositories and caches them locally."""

    def __init__(self, cache_dir: Path = Path("data/raw/clones")) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(self, repo_url: str) -> str:
        """Derive a filesystem-safe directory name from a repo URL."""
        parts = repo_url.rstrip("/").split("/")
        name = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)

    def clone(
        self,
        repo_url: str,
        ref: Optional[str] = None,
        timeout: int = _DEFAULT_CLONE_TIMEOUT,
    ) -> Optional[Path]:
        """Shallow-clone repo_url to the cache directory.

        If ref is provided, fetch and checkout that specific ref (e.g. a parent
        commit SHA) so callers can inspect the pre-fix state of the tree.

        Returns the local path on success, None if the clone fails (private,
        deleted, or timed out).
        """
        dest = self.cache_dir / self._safe_name(repo_url)

        if dest.exists():
            return dest

        try:
            result = subprocess.run(
                ["git", "clone", "--depth=1", repo_url, str(dest)],
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "Clone of %s timed out after %ds. "
                "Try cloning manually: git clone --depth=1 %s %s",
                repo_url, timeout, repo_url, dest,
            )
            # Remove any partial clone so a retry starts clean.
            if dest.exists():
                import shutil
                shutil.rmtree(dest, ignore_errors=True)
            return None

        if result.returncode != 0:
            logger.warning(
                "Failed to clone %s (may be private or deleted): %s",
                repo_url,
                result.stderr.decode(errors="replace").strip(),
            )
            # Remove any partial directory left by git so the next call doesn't
            # mistake it for a successful cached clone.
            if dest.exists():
                import shutil
                shutil.rmtree(dest, ignore_errors=True)
            return None

        if ref:
            fetch = subprocess.run(
                ["git", "-C", str(dest), "fetch", "--depth=1", "origin", ref],
                capture_output=True,
                timeout=60,
            )
            if fetch.returncode == 0:
                subprocess.run(
                    ["git", "-C", str(dest), "checkout", "FETCH_HEAD"],
                    capture_output=True,
                    timeout=30,
                )
            else:
                logger.warning("Could not fetch ref %s for %s", ref, repo_url)

        return dest

    def get_diff(self, repo_path: Path, base_ref: str, head_ref: str) -> str:
        """Return the unified diff between base_ref and head_ref."""
        result = subprocess.run(
            ["git", "-C", str(repo_path), "diff", base_ref, head_ref],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning(
                "git diff failed in %s: %s",
                repo_path,
                result.stderr.decode(errors="replace").strip(),
            )
            return ""
        return result.stdout.decode(errors="replace")

    def count_repo_files(self, repo_path: Path) -> int:
        """Return the number of tracked files in the repository."""
        result = subprocess.run(
            ["git", "-C", str(repo_path), "ls-files"],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return 0
        return len(result.stdout.decode(errors="replace").splitlines())
