import asyncio
import hashlib
import json
import logging
import math
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

_EMBEDDING_DIM = 256  # Default dimension for the local hash-based provider

# Known output dimensions for remote embedding models.
# Used to ensure the local fallback vector matches the expected dimension so
# cosine_similarity is never called with vectors of different lengths.
_MODEL_DIMS: dict[str, int] = {
    # Amazon Titan
    "amazon.titan-embed-text-v1": 1536,
    "amazon.titan-embed-text-v2:0": 1024,
    # Cohere (Bedrock and direct)
    "cohere.embed-english-v3": 1024,
    "cohere.embed-multilingual-v3": 1024,
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


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
        embedding_model: str = "amazon.titan-embed-text-v2:0",
        embedding_provider: str = "local",
        aws_region: Optional[str] = None,
        aws_profile: Optional[str] = None,
    ):
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.aws_region = aws_region
        self.aws_profile = aws_profile
        self._entries: list[ScratchpadEntry] = []
        self._embeddings: list[list[float]] = []
        self._embedding_cache: dict[str, list[float]] = {}
        self._openai_client = None  # Lazily initialised once on first API call
        self._bedrock_client = None  # Lazily initialised once on first Bedrock call

    async def store(self, message: AgentMessage) -> None:
        """Parse the agent message into structured scratchpad entries and embed each."""
        entries = self._parse_into_entries(message)
        for entry in entries:
            embedding = await asyncio.to_thread(self._get_embedding, entry.content)
            self._entries.append(entry)
            self._embeddings.append(embedding)

    async def retrieve(self, agent_name: str, query: Optional[str] = None) -> list[AgentMessage]:
        """Find top_k most similar entries by cosine similarity and return as AgentMessages."""
        if not self._entries:
            return []

        if query is None:
            query = DEFAULT_QUERIES.get(agent_name, agent_name)

        query_embedding = await asyncio.to_thread(self._get_embedding, query)
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

        if self.embedding_provider == "bedrock":
            embedding = self._get_embedding_bedrock(text)
        elif self.embedding_provider == "api":
            embedding = self._get_embedding_api(text)
        else:
            embedding = self._get_embedding_local(text)

        self._embedding_cache[text] = embedding
        return embedding

    def _fallback_dim(self) -> int:
        """Return the vector dimension expected for the configured model.

        When the primary provider fails and we fall back to local hashing, we
        must produce a vector of this length so that cosine_similarity is never
        called with mismatched dimensions (zip truncates silently, which corrupts
        the dot product while the norms are computed on the full vectors).
        """
        if self.embedding_provider == "local":
            return _EMBEDDING_DIM
        return _MODEL_DIMS.get(self.embedding_model, _EMBEDDING_DIM)

    def _get_embedding_local(self, text: str, dim: int = _EMBEDDING_DIM) -> list[float]:
        """Bag-of-words hash embedding into a fixed-dim vector. No external deps."""
        import re

        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        vec = [0.0] * dim
        for token in tokens:
            # Use SHA-256 to map token to a bucket index
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            vec[idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _get_embedding_bedrock(self, text: str) -> list[float]:
        """Bedrock-native embedding via boto3. Supports Titan and Cohere Embed models."""
        try:
            import boto3  # type: ignore

            if self._bedrock_client is None:
                session = boto3.Session(
                    region_name=self.aws_region,
                    profile_name=self.aws_profile,
                )
                self._bedrock_client = session.client("bedrock-runtime")

            if self.embedding_model.startswith("amazon.titan"):
                body = json.dumps({"inputText": text})
                response = self._bedrock_client.invoke_model(
                    modelId=self.embedding_model,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                result = json.loads(response["body"].read())
                return result["embedding"]

            elif self.embedding_model.startswith("cohere.embed"):
                body = json.dumps({"texts": [text], "input_type": "search_document"})
                response = self._bedrock_client.invoke_model(
                    modelId=self.embedding_model,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                result = json.loads(response["body"].read())
                return result["embeddings"][0]

            else:
                raise ValueError(f"Unsupported Bedrock embedding model: {self.embedding_model}")

        except ImportError:
            logger.warning("boto3 not installed; falling back to local embedding.")
            return self._get_embedding_local(text, dim=self._fallback_dim())
        except Exception as exc:
            logger.warning("Bedrock embedding failed (%s); falling back to local embedding.", exc)
            return self._get_embedding_local(text, dim=self._fallback_dim())

    def _get_embedding_api(self, text: str) -> list[float]:
        """API-based embedding. Requires OPENAI_API_KEY or equivalent."""
        try:
            import openai  # type: ignore

            if self._openai_client is None:
                self._openai_client = openai.OpenAI()
            response = self._openai_client.embeddings.create(input=text, model=self.embedding_model)
            return response.data[0].embedding
        except ImportError:
            logger.warning(
                "openai package not installed; falling back to local embedding. "
                "Install openai or set embedding_provider=local."
            )
            return self._get_embedding_local(text, dim=self._fallback_dim())
        except Exception as exc:
            logger.warning("API embedding failed (%s); falling back to local embedding.", exc)
            return self._get_embedding_local(text, dim=self._fallback_dim())
