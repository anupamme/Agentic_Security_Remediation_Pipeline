from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import Any, Optional
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
    language: str = "python"
    target_files: list[str] = []  # Empty = scan all files in repo
    vulnerabilities: list[Vulnerability] = []
    triage_results: list[TriageResult] = []
    patches: list[Patch] = []
    reviews: list[ReviewResult] = []
    messages: list[AgentMessage] = []
    status: str = "pending"  # pending, scanning, triaging, patching, reviewing, complete, failed
    revision_count: int = 0
    max_revisions: int = 3
    failed_vulns: list[str] = []


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
    pr_url: Optional[str] = None
    classification_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    vuln_type: Optional[str] = None       # for grouping in aggregate_metrics
    complexity_tag: Optional[str] = None  # for grouping in aggregate_metrics


class MetricStats(BaseModel):
    """Descriptive statistics for a single metric across multiple eval runs."""
    mean: float
    std: float
    median: float
    min: float
    max: float


class AggregateMetrics(BaseModel):
    """Aggregated metrics across all eval examples."""
    n_examples: int
    detection_recall: MetricStats
    detection_precision: MetricStats
    triage_accuracy: MetricStats
    patch_correctness: MetricStats
    e2e_success_rate: float
    total_cost_usd: MetricStats
    total_tokens: MetricStats
    latency_seconds: MetricStats
    revision_loops: MetricStats
    by_vuln_type: dict[str, "AggregateMetrics"] = {}
    by_complexity: dict[str, "AggregateMetrics"] = {}


class JudgeScore(BaseModel):
    """LLM-as-judge score for a patch."""
    correctness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)
    reasoning: str


class CalibrationResult(BaseModel):
    """Judge calibration against human scores."""
    pearson_r: float
    spearman_r: float
    mean_absolute_error: float
    n_examples: int
    per_example: list[dict]  # {"example_id", "human_score", "judge_score", "delta"}


class EvalReport(BaseModel):
    """Complete evaluation report for a benchmark run."""
    run_id: str
    config: dict[str, Any]
    benchmark_split: str
    num_examples: int
    num_runs: int
    aggregate_metrics: AggregateMetrics
    per_example_results: list[EvalResult]
    cost_summary: dict[str, Any]  # serialized RunCostSummary
    judge_calibration: Optional[CalibrationResult] = None
    timestamp: datetime
