import glob as _glob
import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

_LANG_EXTENSIONS: dict[str, list[str]] = {
    "python": ["*.py"],
    "javascript": ["*.js", "*.ts", "*.jsx", "*.tsx"],
    "go": ["*.go"],
    "java": ["*.java"],
}

_DEPENDENCY_MANIFESTS = [
    "requirements.txt",
    "Pipfile",
    "pyproject.toml",
    "package.json",
    "go.mod",
    "pom.xml",
    "build.gradle",
]

_TEST_DIRS = {"tests", "test", "spec", "__tests__"}

_PYTHON_FRAMEWORK_PATTERNS = [
    (re.compile(r"(?:^|\n)\s*(?:import|from)\s+(django)\b"), "django"),
    (re.compile(r"(?:^|\n)\s*(?:import|from)\s+(flask)\b"), "flask"),
    (re.compile(r"(?:^|\n)\s*(?:import|from)\s+(fastapi)\b"), "fastapi"),
    (re.compile(r"(?:^|\n)\s*(?:import|from)\s+(tornado)\b"), "tornado"),
]

_JS_FRAMEWORKS = ["express", "next", "react", "angular", "vue"]

_GO_FRAMEWORK_IMPORTS = {
    "github.com/gin-gonic/gin": "gin",
    "github.com/labstack/echo": "echo",
    "github.com/gofiber/fiber": "fiber",
}

_JAVA_FRAMEWORK_PATTERNS = [
    (re.compile(r"spring-boot", re.IGNORECASE), "spring-boot"),
    (re.compile(r"\bstruts\b", re.IGNORECASE), "struts"),
    (re.compile(r"\bplay\b", re.IGNORECASE), "play"),
]


def detect_framework(repo_path: str, language: str) -> Optional[str]:
    """Detect the web framework used in the repo.

    Checks:
    - Python: imports of django, flask, fastapi, tornado in .py files
    - JavaScript: dependencies in package.json (express, next, react, angular, vue)
    - Go: imports of gin, echo, fiber in .go files
    - Java: spring-boot, struts, play in pom.xml / build.gradle

    Returns the framework name or None.
    """
    try:
        if language == "python":
            return _detect_python_framework(repo_path)
        elif language in ("javascript", "typescript"):
            return _detect_js_framework(repo_path)
        elif language == "go":
            return _detect_go_framework(repo_path)
        elif language == "java":
            return _detect_java_framework(repo_path)
    except Exception as exc:
        logger.debug("Framework detection error for %s/%s: %s", repo_path, language, exc)
    return None


def _detect_python_framework(repo_path: str) -> Optional[str]:
    # Check .py files up to depth 2
    for pattern in ["*.py", "*/*.py"]:
        for path in _glob.glob(os.path.join(repo_path, pattern)):
            try:
                content = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for regex, name in _PYTHON_FRAMEWORK_PATTERNS:
                if regex.search(content):
                    return name
    return None


def _detect_js_framework(repo_path: str) -> Optional[str]:
    pkg_path = os.path.join(repo_path, "package.json")
    if not os.path.isfile(pkg_path):
        return None
    try:
        with open(pkg_path, encoding="utf-8") as f:
            pkg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    all_deps: dict = {}
    all_deps.update(pkg.get("dependencies", {}))
    all_deps.update(pkg.get("devDependencies", {}))
    for fw in _JS_FRAMEWORKS:
        if fw in all_deps:
            return fw
    return None


def _detect_go_framework(repo_path: str) -> Optional[str]:
    for path in _glob.glob(os.path.join(repo_path, "**", "*.go"), recursive=True):
        try:
            content = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for import_path, name in _GO_FRAMEWORK_IMPORTS.items():
            if import_path in content:
                return name
    return None


def _detect_java_framework(repo_path: str) -> Optional[str]:
    for manifest in ("pom.xml", "build.gradle"):
        path = os.path.join(repo_path, manifest)
        if not os.path.isfile(path):
            continue
        try:
            content = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for regex, name in _JAVA_FRAMEWORK_PATTERNS:
            if regex.search(content):
                return name
    return None


def extract_repo_metadata(repo_path: str, language: str):
    """Extract repository metadata for use by the TriagerAgent.

    Returns a RepoMetadata instance with:
    - repo_name from .git/config or directory name
    - framework via detect_framework()
    - has_tests: True if tests/, test/, spec/ exists or test files found
    - file_count: count of source files matching language extensions
    - dependency_files: list of found package manifest paths (relative)
    """
    # Import here to avoid circular imports at module load time
    from multi_agent_security.agents.triager import RepoMetadata

    repo_name = _extract_repo_name(repo_path)
    framework = detect_framework(repo_path, language)
    has_tests = _check_has_tests(repo_path)
    file_count = _count_source_files(repo_path, language)
    dependency_files = _find_dependency_files(repo_path)

    return RepoMetadata(
        repo_name=repo_name,
        language=language,
        framework=framework,
        has_tests=has_tests,
        file_count=file_count,
        dependency_files=dependency_files,
    )


def _extract_repo_name(repo_path: str) -> str:
    git_config = os.path.join(repo_path, ".git", "config")
    if os.path.isfile(git_config):
        try:
            with open(git_config, encoding="utf-8") as f:
                for line in f:
                    m = re.search(r"url\s*=\s*(.+)", line)
                    if m:
                        url = m.group(1).strip()
                        # Strip trailing .git and take basename
                        basename = os.path.basename(url)
                        if basename.endswith(".git"):
                            basename = basename[:-4]
                        if basename:
                            return basename
        except OSError:
            pass
    return os.path.basename(os.path.abspath(repo_path))


def _check_has_tests(repo_path: str) -> bool:
    # Check for test directories
    for name in _TEST_DIRS:
        if os.path.isdir(os.path.join(repo_path, name)):
            return True
    # Check for test files
    test_file_patterns = [
        os.path.join(repo_path, "**", "test_*.py"),
        os.path.join(repo_path, "**", "*_test.py"),
        os.path.join(repo_path, "**", "*_test.go"),
        os.path.join(repo_path, "**", "*.spec.js"),
        os.path.join(repo_path, "**", "*.spec.ts"),
        os.path.join(repo_path, "**", "*.test.js"),
        os.path.join(repo_path, "**", "*.test.ts"),
    ]
    for pattern in test_file_patterns:
        if _glob.glob(pattern, recursive=True):
            return True
    return False


def _count_source_files(repo_path: str, language: str) -> int:
    extensions = _LANG_EXTENSIONS.get(language, ["*.*"])
    paths: set[str] = set()
    for ext in extensions:
        paths.update(_glob.glob(os.path.join(repo_path, "**", ext), recursive=True))
    return len(paths)


def _find_dependency_files(repo_path: str) -> list[str]:
    found: list[str] = []
    for manifest in _DEPENDENCY_MANIFESTS:
        full_path = os.path.join(repo_path, manifest)
        if os.path.isfile(full_path):
            found.append(manifest)
    return found


class CodeParser:
    def parse(self, source: str, language: str):
        raise NotImplementedError
