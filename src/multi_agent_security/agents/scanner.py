import glob as _glob
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from multi_agent_security.agents.base import BaseAgent
from multi_agent_security.tools.static_analysis import (
    StaticAnalysisFinding,
    run_dependency_audit,
    run_semgrep,
)
from multi_agent_security.types import AgentMessage, FileContext, Vulnerability, VulnType

logger = logging.getLogger(__name__)

_LANG_EXTENSIONS: dict[str, list[str]] = {
    "python": ["*.py"],
    "javascript": ["*.js", "*.ts", "*.jsx", "*.tsx"],
    "go": ["*.go"],
    "java": ["*.java"],
}

_SYSTEM_PROMPT = """\
You are a security vulnerability scanner. Your job is to analyze source code and identify security vulnerabilities.

You will be given:
1. The source code of a file
2. (Optionally) Findings from static analysis tools like Semgrep

Your task:
- Identify ALL security vulnerabilities in the code
- For each vulnerability, provide: exact location (line numbers), CWE category, description, confidence score, and the vulnerable code snippet
- Consider: injection flaws, authentication issues, access control problems, cryptographic weaknesses, hardcoded secrets, insecure dependencies, input validation gaps, SSRF, path traversal, and insecure deserialization
- If static analysis findings are provided, validate them (confirm or dismiss) and look for additional issues the tools missed
- If you find NO vulnerabilities, return an empty list — do not fabricate findings

CRITICAL: Minimize false positives. Only flag issues you are genuinely confident about. A confidence score below 0.5 means you should not include it.
"""

_SMALL_FILE_MAX_LINES = 100       # batch multiple files together
_MEDIUM_FILE_MAX_LINES = 500      # full content, single batch
_LARGE_FILE_MAX_LINES = 2000      # single batch with note
_SMALL_BATCH_MAX_CHARS = 12000    # ~3000 tokens


# ---------------------------------------------------------------------------
# Input / output models
# ---------------------------------------------------------------------------

class ScannerInput(BaseModel):
    repo_path: str
    target_files: list[str] = Field(default_factory=list)  # empty = scan all
    language: str
    static_analysis_results: Optional[list[StaticAnalysisFinding]] = None


class ScannerOutput(BaseModel):
    vulnerabilities: list[Vulnerability]
    files_scanned: int
    scan_summary: str


# ---------------------------------------------------------------------------
# Intermediate LLM response models (no `id` — assigned post-scan)
# ---------------------------------------------------------------------------

class _VulnCandidate(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    vuln_type: VulnType
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    code_snippet: str
    scanner_reasoning: str


class _LLMScanResult(BaseModel):
    vulnerabilities: list[_VulnCandidate]


# ---------------------------------------------------------------------------
# Batching utilities
# ---------------------------------------------------------------------------

class ScanBatch(BaseModel):
    files: list[FileContext]
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None


class FileChunker:
    def __init__(self, max_lines_per_chunk: int = 1000, overlap: int = 100):
        self.max_lines = max_lines_per_chunk
        self.overlap = overlap

    def chunk_files(self, files: list[FileContext]) -> list[ScanBatch]:
        batches: list[ScanBatch] = []
        small_bucket: list[FileContext] = []
        small_bucket_chars = 0

        for fc in files:
            lines = fc.line_count

            if lines <= _SMALL_FILE_MAX_LINES:
                # Batch with other small files up to the char budget
                if (small_bucket_chars + len(fc.content) > _SMALL_BATCH_MAX_CHARS
                        and small_bucket):
                    batches.append(ScanBatch(files=list(small_bucket)))
                    small_bucket = []
                    small_bucket_chars = 0
                small_bucket.append(fc)
                small_bucket_chars += len(fc.content)

            elif lines <= _MEDIUM_FILE_MAX_LINES:
                # Flush small bucket first
                if small_bucket:
                    batches.append(ScanBatch(files=list(small_bucket)))
                    small_bucket = []
                    small_bucket_chars = 0
                batches.append(ScanBatch(files=[fc]))

            elif lines <= _LARGE_FILE_MAX_LINES:
                if small_bucket:
                    batches.append(ScanBatch(files=list(small_bucket)))
                    small_bucket = []
                    small_bucket_chars = 0
                batches.append(ScanBatch(files=[fc]))

            else:
                # Split into overlapping chunks
                if small_bucket:
                    batches.append(ScanBatch(files=list(small_bucket)))
                    small_bucket = []
                    small_bucket_chars = 0
                content_lines = fc.content.splitlines()
                chunks = self._split_lines(content_lines, self.max_lines, self.overlap)
                total = len(chunks)
                for idx, chunk_lines in enumerate(chunks):
                    chunk_content = "\n".join(chunk_lines)
                    chunk_fc = FileContext(
                        path=fc.path,
                        content=chunk_content,
                        language=fc.language,
                        line_count=len(chunk_lines),
                    )
                    batches.append(ScanBatch(
                        files=[chunk_fc],
                        chunk_index=idx,
                        total_chunks=total,
                    ))

        if small_bucket:
            batches.append(ScanBatch(files=list(small_bucket)))

        return batches

    @staticmethod
    def _split_lines(
        lines: list[str], chunk_size: int, overlap: int
    ) -> list[list[str]]:
        chunks = []
        start = 0
        while start < len(lines):
            end = min(start + chunk_size, len(lines))
            chunks.append(lines[start:end])
            if end == len(lines):
                break
            start = end - overlap
        return chunks


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _line_range(c: _VulnCandidate) -> int:
    return max(1, c.line_end - c.line_start + 1)


def _overlap_pct(a: _VulnCandidate, b: _VulnCandidate) -> float:
    overlap = max(0, min(a.line_end, b.line_end) - max(a.line_start, b.line_start) + 1)
    min_range = min(_line_range(a), _line_range(b))
    return overlap / min_range if min_range > 0 else 0.0


def _deduplicate(candidates: list[_VulnCandidate]) -> list[_VulnCandidate]:
    kept: list[_VulnCandidate] = []
    removed = 0
    for candidate in candidates:
        is_dup = False
        for i, existing in enumerate(kept):
            if (existing.file_path == candidate.file_path
                    and _overlap_pct(existing, candidate) > 0.5):
                is_dup = True
                if candidate.confidence > existing.confidence:
                    kept[i] = candidate
                removed += 1
                break
        if not is_dup:
            kept.append(candidate)
    if removed:
        logger.debug("Deduplication removed %d overlapping findings", removed)
    return kept


# ---------------------------------------------------------------------------
# Scanner Agent
# ---------------------------------------------------------------------------

class ScannerAgent(BaseAgent):
    name = "scanner"

    async def run(
        self, input_data: BaseModel, context: list[AgentMessage]
    ) -> tuple[ScannerOutput, AgentMessage]:
        assert isinstance(input_data, ScannerInput)
        inp: ScannerInput = input_data

        # 1. Resolve file list
        file_contexts = self._load_files(inp)
        files_scanned = len(file_contexts)

        # 2. Static analysis
        sa_results = inp.static_analysis_results
        if sa_results is None and self.config.agents.scanner.use_static_analysis:
            sa_results = []
            try:
                sa_results += await run_semgrep(inp.repo_path, inp.language)
                sa_results += await run_dependency_audit(inp.repo_path, inp.language)
            except Exception as exc:
                logger.warning("Static analysis error: %s", exc)

        # 3. Batch files
        chunker = FileChunker()
        batches = chunker.chunk_files(file_contexts)

        # 4. Scan each batch, accumulate metrics
        all_candidates: list[_VulnCandidate] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0
        total_cost_usd = 0.0

        for i, batch in enumerate(batches, 1):
            logger.info(
                "Scanning batch %d/%d (files: %s)",
                i, len(batches),
                [fc.path for fc in batch.files],
            )
            candidates, response = await self._scan_batch(batch, inp.language, sa_results or [])
            all_candidates.extend(candidates)
            total_input_tokens += response.get("input_tokens", 0)
            total_output_tokens += response.get("output_tokens", 0)
            total_latency_ms += response.get("latency_ms", 0.0)
            total_cost_usd += response.get("cost_usd", 0.0)

        # 5. Deduplicate
        all_candidates = _deduplicate(all_candidates)

        # 6. Filter, assign IDs, sort
        all_candidates = [c for c in all_candidates if c.confidence >= 0.5]
        all_candidates.sort(key=lambda c: c.confidence, reverse=True)

        vulnerabilities = [
            Vulnerability(
                id=f"VULN-{i + 1:03d}",
                file_path=c.file_path,
                line_start=c.line_start,
                line_end=c.line_end,
                vuln_type=c.vuln_type,
                description=c.description,
                confidence=c.confidence,
                code_snippet=c.code_snippet,
                scanner_reasoning=c.scanner_reasoning,
            )
            for i, c in enumerate(all_candidates)
        ]

        summary = (
            f"Scanned {files_scanned} file(s); found {len(vulnerabilities)} vulnerability(s). "
            + (
                f"Types: {', '.join(sorted({v.vuln_type.value for v in vulnerabilities}))}."
                if vulnerabilities
                else "No vulnerabilities detected."
            )
        )

        output = ScannerOutput(
            vulnerabilities=vulnerabilities,
            files_scanned=files_scanned,
            scan_summary=summary,
        )

        message = AgentMessage(
            agent_name=type(self).name,
            timestamp=datetime.now(timezone.utc),
            content=output.model_dump_json(),
            token_count_input=total_input_tokens,
            token_count_output=total_output_tokens,
            latency_ms=total_latency_ms,
            cost_usd=total_cost_usd,
        )

        return output, message

    def _load_files(self, inp: ScannerInput) -> list[FileContext]:
        if inp.target_files:
            paths = [os.path.join(inp.repo_path, f) for f in inp.target_files]
        else:
            patterns = _LANG_EXTENSIONS.get(inp.language, ["*.*"])
            paths = []
            for pattern in patterns:
                paths.extend(
                    _glob.glob(os.path.join(inp.repo_path, "**", pattern), recursive=True)
                )

        contexts: list[FileContext] = []
        for path in paths:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                contexts.append(FileContext(
                    path=os.path.relpath(path, inp.repo_path),
                    content=content,
                    language=inp.language,
                    line_count=content.count("\n") + 1,
                ))
            except OSError as exc:
                logger.warning("Could not read %s: %s", path, exc)

        return contexts

    async def _scan_batch(
        self,
        batch: ScanBatch,
        language: str,
        sa_results: list[StaticAnalysisFinding],
    ) -> tuple[list[_VulnCandidate], dict]:
        user_prompt = self._build_user_prompt(batch, language, sa_results)
        response = await self.llm.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format=_LLMScanResult,
            agent_name=type(self).name,
        )

        metrics = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
        }

        try:
            scan_result = _LLMScanResult.model_validate_json(response.content)
            return scan_result.vulnerabilities, metrics
        except Exception as exc:
            logger.warning("Failed to parse LLM scan result: %s", exc)
            return [], metrics

    def _build_user_prompt(
        self,
        batch: ScanBatch,
        language: str,
        sa_results: list[StaticAnalysisFinding],
    ) -> str:
        parts: list[str] = []

        for fc in batch.files:
            chunk_note = ""
            if batch.chunk_index is not None:
                chunk_note = (
                    f" [Chunk {batch.chunk_index + 1}/{batch.total_chunks} — "
                    "focus on security-relevant sections]"
                )
            elif fc.line_count > _MEDIUM_FILE_MAX_LINES:
                chunk_note = " [Large file — focus on security-relevant sections]"

            parts.append(
                f"## File: {fc.path}{chunk_note}\n"
                f"## Language: {language}\n\n"
                f"### Source Code:\n"
                f"```{language}\n{fc.content}\n```"
            )

        # Filter SA findings relevant to this batch's files
        batch_paths = {fc.path for fc in batch.files}
        relevant_sa = [f for f in sa_results if f.file_path in batch_paths]
        if relevant_sa:
            formatted = "\n".join(
                f"- [{f.tool}] {f.rule_id} @ {f.file_path}:{f.line_start}-{f.line_end} "
                f"({f.severity}): {f.message}"
                + (f" [{f.cwe}]" if f.cwe else "")
                for f in relevant_sa
            )
            parts.append(
                "\n### Static Analysis Findings:\n"
                "The following issues were flagged by automated tools. Validate each one "
                "(confirm or dismiss) and identify any additional vulnerabilities the tools missed.\n\n"
                + formatted
            )

        return "\n\n".join(parts)
