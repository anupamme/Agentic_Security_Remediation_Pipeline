from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import Optional
from datetime import datetime, timezone


class VulnSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnType(str, Enum):
    """Top CWE categories relevant to code scanning."""
    SQL_INJECTION = "CWE-89"
    XSS = "CWE-79"
    PATH_TRAVERSAL = "CWE-22"
    COMMAND_INJECTION = "CWE-78"
    INSECURE_DESERIALIZATION = "CWE-502"
    HARDCODED_CREDENTIALS = "CWE-798"
    INSECURE_DEPENDENCY = "CWE-1104"
    IMPROPER_INPUT_VALIDATION = "CWE-20"
    BROKEN_ACCESS_CONTROL = "CWE-284"
    CRYPTO_WEAKNESS = "CWE-327"
    SSRF = "CWE-918"
    OTHER = "OTHER"


class FixStrategy(str, Enum):
    ONE_LINER = "one_liner"
    REFACTOR = "refactor"
    DEPENDENCY_BUMP = "dependency_bump"
    CONFIG_CHANGE = "config_change"
    MULTI_FILE = "multi_file"


class FileContext(BaseModel):
    """A single file's content and metadata."""
    path: str
    content: str
    language: str
    line_count: int


class Vulnerability(BaseModel):
    """Scanner output: a detected vulnerability."""
    id: str = Field(description="Unique ID, e.g., VULN-001")
    file_path: str
    line_start: int
    line_end: int
    vuln_type: VulnType
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    code_snippet: str = Field(description="The vulnerable code fragment")
    scanner_reasoning: str = Field(description="Why the scanner flagged this")


class TriageResult(BaseModel):
    """Triager output: prioritized vulnerability with fix strategy."""
    vuln_id: str
    severity: VulnSeverity
    exploitability_score: float = Field(ge=0.0, le=1.0)
    fix_strategy: FixStrategy
    estimated_complexity: str = Field(description="low / medium / high")
    triage_reasoning: str


class Patch(BaseModel):
    """Patcher output: a proposed code fix."""
    vuln_id: str
    file_path: str
    original_code: str
    patched_code: str
    unified_diff: str
    patch_reasoning: str = Field(description="Why this fix is correct")


class ReviewResult(BaseModel):
    """Reviewer output: accept/reject with reasoning."""
    vuln_id: str
    patch_accepted: bool
    correctness_score: float = Field(ge=0.0, le=1.0)
    security_score: float = Field(ge=0.0, le=1.0)
    style_score: float = Field(ge=0.0, le=1.0)
    review_reasoning: str
    revision_request: Optional[str] = Field(default=None, description="If rejected, what to fix")


class AgentMessage(BaseModel):
    """A message in the agent communication log."""
    agent_name: str
    timestamp: datetime
    content: str  # The actual message/output (JSON string of agent output)
    token_count_input: int
    token_count_output: int
    latency_ms: float
    cost_usd: float

    @field_validator("timestamp")
    @classmethod
    def require_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("timestamp must be timezone-aware (e.g. datetime.now(timezone.utc))")
        return v


class TaskState(BaseModel):
    """Global state for a single vulnerability remediation task."""
    task_id: str
    repo_url: str
    vulnerabilities: list[Vulnerability] = []
    triage_results: list[TriageResult] = []
    patches: list[Patch] = []
    reviews: list[ReviewResult] = []
    messages: list[AgentMessage] = []
    status: str = "pending"  # pending, scanning, triaging, patching, reviewing, complete, failed
    revision_count: int = 0
    max_revisions: int = 3


class BenchmarkExample(BaseModel):
    """A single example in the benchmark dataset."""
    id: str
    repo_url: str
    repo_name: str
    language: str
    vulnerable_files: list[str]
    vuln_type: VulnType
    severity: VulnSeverity
    ground_truth_diff: str
    merge_status: str  # "merged" or "rejected"
    complexity_tag: str  # "single_file", "multi_file", "dependency", "config"
    negative: bool = False  # True if this is a negative example (no real vuln)


class EvalResult(BaseModel):
    """Evaluation result for a single benchmark example."""
    example_id: str
    architecture: str
    memory_strategy: str
    detection_recall: float
    detection_precision: float
    triage_accuracy: float
    patch_correctness: float
    end_to_end_success: bool
    total_tokens: int
    total_cost_usd: float
    latency_seconds: float
    revision_loops: int
    failure_stage: Optional[str] = None  # "scanner", "triager", "patcher", "reviewer", None
    failure_reason: Optional[str] = None
