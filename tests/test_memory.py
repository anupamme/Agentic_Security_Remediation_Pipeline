"""Tests for sliding window memory, retrieval memory, cosine similarity, and factory."""
import asyncio
import json
from datetime import datetime, timezone

import pytest

from multi_agent_security.config import AppConfig, MemoryConfig
from multi_agent_security.memory import create_memory
from multi_agent_security.memory.full_context import FullContextMemory
from multi_agent_security.memory.retrieval import RetrievalMemory
from multi_agent_security.memory.sliding_window import SlidingWindowMemory
from multi_agent_security.types import AgentMessage
from multi_agent_security.utils.similarity import cosine_similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_message(agent_name: str = "scanner", content: str = "test") -> AgentMessage:
    return AgentMessage(
        agent_name=agent_name,
        timestamp=datetime.now(timezone.utc),
        content=content,
        token_count_input=10,
        token_count_output=10,
        latency_ms=50.0,
        cost_usd=0.001,
    )


def _make_config(**memory_overrides) -> AppConfig:
    """Build a minimal AppConfig with memory overrides."""
    memory_data = {
        "strategy": "full_context",
        "sliding_window_size": 5,
        "retrieval_top_k": 3,
        "embedding_provider": "local",
    }
    memory_data.update(memory_overrides)
    return AppConfig.model_validate({"memory": memory_data})


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_known_values(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        dot = 1 * 4 + 2 * 5 + 3 * 6  # 32
        import math
        na = math.sqrt(1 + 4 + 9)
        nb = math.sqrt(16 + 25 + 36)
        expected = dot / (na * nb)
        assert cosine_similarity(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# SlidingWindowMemory
# ---------------------------------------------------------------------------

class TestSlidingWindowMemory:
    def test_below_window_no_summary(self):
        mem = SlidingWindowMemory(window_size=5, llm_client=None)
        for i in range(3):
            mem.store(_make_message(content=f"msg {i}"))
        result = mem.retrieve("scanner")
        assert len(result) == 3
        assert all(m.agent_name != "memory_summary" for m in result)

    def test_window_enforcement_extractive(self):
        """Store 10 messages (window=3) → summary + last 3."""
        mem = SlidingWindowMemory(window_size=3, llm_client=None)
        for i in range(10):
            mem.store(_make_message(content=f"message number {i}"))
        result = mem.retrieve("scanner")
        # At most window_size + 1 summary entry
        window_msgs = [m for m in result if m.agent_name != "memory_summary"]
        summary_msgs = [m for m in result if m.agent_name == "memory_summary"]
        assert len(window_msgs) == 3
        assert len(summary_msgs) == 1
        # Last 3 messages should be 7, 8, 9
        assert "message number 7" in window_msgs[0].content
        assert "message number 9" in window_msgs[2].content

    def test_extractive_summary_content(self):
        """Summary should reference earlier messages."""
        mem = SlidingWindowMemory(window_size=2, llm_client=None)
        mem.store(_make_message(agent_name="scanner", content="SQL injection found"))
        mem.store(_make_message(agent_name="triager", content="high priority vuln"))
        mem.store(_make_message(agent_name="patcher", content="patch generated"))
        result = mem.retrieve("reviewer")
        summary_msgs = [m for m in result if m.agent_name == "memory_summary"]
        assert summary_msgs, "Expected a summary message"
        summary_text = summary_msgs[0].content
        assert "SQL injection found" in summary_text or "scanner" in summary_text

    def test_clear_resets_state(self):
        mem = SlidingWindowMemory(window_size=3, llm_client=None)
        for i in range(5):
            mem.store(_make_message(content=f"msg {i}"))
        mem.clear()
        result = mem.retrieve("scanner")
        assert result == []
        assert mem._summary is None
        assert mem._summary_covers_up_to == 0

    def test_async_summary_with_llm_client(self):
        """With a dry-run LLM client, summarization should not raise."""
        from multi_agent_security.llm_client import LLMClient
        from multi_agent_security.config import LLMConfig

        llm = LLMClient(LLMConfig(), dry_run=True)
        mem = SlidingWindowMemory(window_size=2, llm_client=llm)

        async def _run():
            for i in range(5):
                mem.store(_make_message(content=f"finding {i}"))
            await mem._update_summary()

        asyncio.run(_run())
        result = mem.retrieve("reviewer")
        summary_msgs = [m for m in result if m.agent_name == "memory_summary"]
        assert summary_msgs, "Expected a summary after LLM summarization"

    def test_no_duplicate_summarization(self):
        """_update_summary should not re-summarize already-covered range."""
        mem = SlidingWindowMemory(window_size=2, llm_client=None)
        for i in range(4):
            mem.store(_make_message(content=f"msg {i}"))
        original_summary = mem._summary
        original_covers = mem._summary_covers_up_to
        # Manually calling again should be a no-op (already covered)
        asyncio.run(mem._update_summary())
        assert mem._summary_covers_up_to == original_covers


# ---------------------------------------------------------------------------
# RetrievalMemory
# ---------------------------------------------------------------------------

class TestRetrievalMemory:
    def _store_varied_messages(self, mem: RetrievalMemory) -> None:
        messages = [
            ("scanner", '{"vulnerabilities": [{"id": "V1", "vuln_type": "sql_injection", "file_path": "app.py", "description": "SQL injection via user input", "severity": "high"}]}'),
            ("scanner", '{"vulnerabilities": [{"id": "V2", "vuln_type": "xss", "file_path": "view.py", "description": "Cross-site scripting in template", "severity": "medium"}]}'),
            ("scanner", '{"vulnerabilities": [{"id": "V3", "vuln_type": "path_traversal", "file_path": "upload.py", "description": "Path traversal in file upload", "severity": "high"}]}'),
            ("triager", '{"triage_results": [{"vuln_id": "V1", "priority": "critical", "fix_strategy": "parameterized_query", "reasoning": "Direct SQL injection risk"}]}'),
            ("triager", '{"triage_results": [{"vuln_id": "V2", "priority": "medium", "fix_strategy": "escape_output", "reasoning": "XSS via template rendering"}]}'),
            ("patcher", '{"patches": [{"vuln_id": "V1", "file_path": "app.py", "patch_reasoning": "Use parameterized queries to fix SQL injection"}]}'),
            ("patcher", '{"patches": [{"vuln_id": "V2", "file_path": "view.py", "patch_reasoning": "Escape HTML output to prevent XSS"}]}'),
            ("reviewer", '{"reviews": [{"vuln_id": "V1", "patch_accepted": true, "correctness_score": 0.95, "feedback": "Correct fix for SQL injection"}]}'),
            ("scanner", "overflow buffer in C code stack smashing detected"),
            ("triager", "authentication bypass vulnerability in login form"),
        ]
        for agent, content in messages:
            mem.store(_make_message(agent_name=agent, content=content))

    def test_relevance_ordering_sql_injection(self):
        mem = RetrievalMemory(top_k=3, embedding_provider="local")
        self._store_varied_messages(mem)
        results = mem.retrieve("patcher", query="SQL injection vulnerability fix")
        assert results, "Expected at least one result"
        combined = " ".join(r.content.lower() for r in results)
        assert "sql" in combined or "injection" in combined

    def test_top_k_capped(self):
        mem = RetrievalMemory(top_k=3, embedding_provider="local")
        self._store_varied_messages(mem)
        results = mem.retrieve("scanner")
        assert len(results) <= 3

    def test_empty_store_returns_empty(self):
        mem = RetrievalMemory(top_k=5, embedding_provider="local")
        assert mem.retrieve("scanner") == []

    def test_clear_resets_state(self):
        mem = RetrievalMemory(top_k=5, embedding_provider="local")
        self._store_varied_messages(mem)
        mem.clear()
        assert mem.retrieve("scanner") == []
        assert mem._entries == []
        assert mem._embeddings == []
        assert mem._embedding_cache == {}

    def test_local_embedding_deterministic(self):
        mem = RetrievalMemory(embedding_provider="local")
        e1 = mem._get_embedding_local("hello world")
        e2 = mem._get_embedding_local("hello world")
        assert e1 == e2

    def test_local_embedding_different_texts(self):
        mem = RetrievalMemory(embedding_provider="local")
        e1 = mem._get_embedding_local("sql injection attack")
        e2 = mem._get_embedding_local("cross site scripting xss")
        assert e1 != e2

    def test_local_embedding_normalized(self):
        import math
        mem = RetrievalMemory(embedding_provider="local")
        e = mem._get_embedding_local("some text here")
        norm = math.sqrt(sum(x * x for x in e))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_embedding_cache(self):
        mem = RetrievalMemory(embedding_provider="local")
        text = "cached embedding test"
        e1 = mem._get_embedding(text)
        assert text in mem._embedding_cache
        e2 = mem._get_embedding(text)
        assert e1 is e2  # Same object from cache

    def test_parse_fallback_for_non_json(self):
        mem = RetrievalMemory(embedding_provider="local")
        msg = _make_message(content="plain text not json")
        entries = mem._parse_into_entries(msg)
        assert len(entries) == 1
        assert entries[0].entry_type == "summary"
        assert entries[0].content == "plain text not json"

    def test_default_query_used_when_none(self):
        """retrieve() with query=None should use DEFAULT_QUERIES."""
        from multi_agent_security.memory.retrieval import DEFAULT_QUERIES
        mem = RetrievalMemory(top_k=2, embedding_provider="local")
        mem.store(_make_message(agent_name="scanner", content="vuln found"))
        # Should not raise; uses default query for "patcher"
        results = mem.retrieve("patcher", query=None)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# create_memory factory
# ---------------------------------------------------------------------------

class TestCreateMemoryFactory:
    def test_full_context(self):
        config = _make_config(strategy="full_context")
        mem = create_memory(config)
        assert isinstance(mem, FullContextMemory)

    def test_sliding_window(self):
        config = _make_config(strategy="sliding_window", sliding_window_size=7)
        mem = create_memory(config)
        assert isinstance(mem, SlidingWindowMemory)
        assert mem.window_size == 7

    def test_retrieval(self):
        config = _make_config(strategy="retrieval", retrieval_top_k=4)
        mem = create_memory(config)
        assert isinstance(mem, RetrievalMemory)
        assert mem.top_k == 4

    def test_unknown_strategy_raises(self):
        config = _make_config(strategy="nonexistent")
        with pytest.raises(ValueError, match="Unknown memory strategy"):
            create_memory(config)

    def test_sliding_window_passes_llm_client(self):
        from multi_agent_security.llm_client import LLMClient
        from multi_agent_security.config import LLMConfig

        config = _make_config(strategy="sliding_window")
        llm = LLMClient(LLMConfig(), dry_run=True)
        mem = create_memory(config, llm_client=llm)
        assert isinstance(mem, SlidingWindowMemory)
        assert mem.llm_client is llm

    def test_retrieval_embedding_provider(self):
        config = _make_config(strategy="retrieval", embedding_provider="local")
        mem = create_memory(config)
        assert isinstance(mem, RetrievalMemory)
        assert mem.embedding_provider == "local"
