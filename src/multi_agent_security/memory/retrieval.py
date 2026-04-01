import hashlib
import json
import math
import logging
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from multi_agent_security.memory.base import BaseMemory
from multi_agent_security.types import AgentMessage
from multi_agent_security.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_QUERIES: dict[str, str] = {
    "scanner": "security vulnerabilities to detect in source code",
    "triager": "vulnerability findings to prioritize and assess severity",
    "patcher": "vulnerability details and fix strategy for generating a code patch",
    "reviewer": "code patch to review for security correctness and quality",
}

_EMBEDDING_DIM = 256


class ScratchpadEntry(BaseModel):
    source_agent: str
    entry_type: str  # "vulnerability" | "triage" | "patch" | "review" | "summary"
    content: str
    timestamp: datetime
    vuln_id: Optional[str] = None


class RetrievalMemory(BaseMemory):
    """Agents write structured notes to a scratchpad (key-value store).
    When an agent is invoked, relevant entries are retrieved via
    embedding similarity to the current task."""

    def __init__(
        self,
        top_k: int = 5,
        embedding_model: str = "text-embedding-3-small",
        embedding_provider: str = "local",
    ):
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self._entries: list[ScratchpadEntry] = []
        self._embeddings: list[list[float]] = []
        self._embedding_cache: dict[str, list[float]] = {}

    def store(self, message: AgentMessage) -> None:
        """Parse the agent message into structured scratchpad entries and embed each."""
        entries = self._parse_into_entries(message)
        for entry in entries:
            embedding = self._get_embedding(entry.content)
            self._entries.append(entry)
            self._embeddings.append(embedding)

    def retrieve(self, agent_name: str, query: Optional[str] = None) -> list[AgentMessage]:
        """Find top_k most similar entries by cosine similarity and return as AgentMessages."""
        if not self._entries:
            return []

        if query is None:
            query = DEFAULT_QUERIES.get(agent_name, agent_name)

        query_embedding = self._get_embedding(query)
        similarities = [cosine_similarity(query_embedding, e) for e in self._embeddings]
        top_indices = sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )[: self.top_k]

        results = []
        for idx in top_indices:
            entry = self._entries[idx]
            results.append(
                AgentMessage(
                    agent_name=entry.source_agent,
                    timestamp=datetime.now(timezone.utc),
                    content=entry.content,
                    token_count_input=0,
                    token_count_output=0,
                    latency_ms=0.0,
                    cost_usd=0.0,
                )
            )
        return results

    def clear(self) -> None:
        self._entries = []
        self._embeddings = []
        self._embedding_cache = {}

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_into_entries(self, message: AgentMessage) -> list[ScratchpadEntry]:
        """Parse an agent's output into individual scratchpad entries."""
        try:
            data = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            return [
                ScratchpadEntry(
                    source_agent=message.agent_name,
                    entry_type="summary",
                    content=message.content,
                    timestamp=message.timestamp,
                )
            ]

        agent = message.agent_name
        entries: list[ScratchpadEntry] = []

        if agent == "scanner" and isinstance(data, dict):
            vulns = data.get("vulnerabilities", [])
            if isinstance(vulns, list):
                for v in vulns:
                    if isinstance(v, dict):
                        content = (
                            f"Vulnerability: {v.get('vuln_type', 'unknown')} "
                            f"in {v.get('file_path', 'unknown')} "
                            f"(severity: {v.get('severity', 'unknown')}). "
                            f"{v.get('description', '')}"
                        )
                        entries.append(
                            ScratchpadEntry(
                                source_agent=agent,
                                entry_type="vulnerability",
                                content=content,
                                timestamp=message.timestamp,
                                vuln_id=v.get("id"),
                            )
                        )
        elif agent == "triager" and isinstance(data, dict):
            results = data.get("triage_results", [])
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict):
                        content = (
                            f"Triage: {r.get('vuln_id', 'unknown')} "
                            f"priority={r.get('priority', 'unknown')} "
                            f"strategy={r.get('fix_strategy', 'unknown')}. "
                            f"{r.get('reasoning', '')}"
                        )
                        entries.append(
                            ScratchpadEntry(
                                source_agent=agent,
                                entry_type="triage",
                                content=content,
                                timestamp=message.timestamp,
                                vuln_id=r.get("vuln_id"),
                            )
                        )
        elif agent == "patcher" and isinstance(data, dict):
            patches = data.get("patches", [])
            if isinstance(patches, list):
                for p in patches:
                    if isinstance(p, dict):
                        content = (
                            f"Patch for {p.get('vuln_id', 'unknown')}: "
                            f"{p.get('patch_reasoning', '')} "
                            f"file={p.get('file_path', 'unknown')}"
                        )
                        entries.append(
                            ScratchpadEntry(
                                source_agent=agent,
                                entry_type="patch",
                                content=content,
                                timestamp=message.timestamp,
                                vuln_id=p.get("vuln_id"),
                            )
                        )
        elif agent == "reviewer" and isinstance(data, dict):
            reviews = data.get("reviews", [])
            if isinstance(reviews, list):
                for r in reviews:
                    if isinstance(r, dict):
                        content = (
                            f"Review for {r.get('vuln_id', 'unknown')}: "
                            f"accepted={r.get('patch_accepted', False)} "
                            f"score={r.get('correctness_score', 0)}. "
                            f"{r.get('feedback', '')}"
                        )
                        entries.append(
                            ScratchpadEntry(
                                source_agent=agent,
                                entry_type="review",
                                content=content,
                                timestamp=message.timestamp,
                                vuln_id=r.get("vuln_id"),
                            )
                        )

        if not entries:
            # Generic fallback: one entry for the entire message
            entries.append(
                ScratchpadEntry(
                    source_agent=agent,
                    entry_type="summary",
                    content=message.content,
                    timestamp=message.timestamp,
                )
            )

        return entries

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding vector with caching."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        if self.embedding_provider == "api":
            embedding = self._get_embedding_api(text)
        else:
            embedding = self._get_embedding_local(text)

        self._embedding_cache[text] = embedding
        return embedding

    def _get_embedding_local(self, text: str) -> list[float]:
        """Bag-of-words hash embedding into a fixed-dim vector. No external deps."""
        import re

        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        vec = [0.0] * _EMBEDDING_DIM
        for token in tokens:
            # Use SHA-256 to map token to a bucket index
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % _EMBEDDING_DIM
            vec[idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _get_embedding_api(self, text: str) -> list[float]:
        """API-based embedding. Requires OPENAI_API_KEY or equivalent."""
        try:
            import openai  # type: ignore

            client = openai.OpenAI()
            response = client.embeddings.create(input=text, model=self.embedding_model)
            return response.data[0].embedding
        except ImportError:
            logger.warning(
                "openai package not installed; falling back to local embedding. "
                "Install openai or set embedding_provider=local."
            )
            return self._get_embedding_local(text)
        except Exception as exc:
            logger.warning("API embedding failed (%s); falling back to local embedding.", exc)
            return self._get_embedding_local(text)
