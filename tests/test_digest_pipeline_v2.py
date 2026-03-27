"""Tests for the Digest Pipeline V2 — chunk_events audit, length-tiered dedup,
structured reports, and connect mode enhancements.

Worker: brainlayer-worker-A-R3
Run: pytest tests/test_digest_pipeline_v2.py -v
"""

import time
from unittest.mock import MagicMock

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_store(tmp_path):
    """Create a real VectorStore backed by a temp DB for isolated testing."""
    from brainlayer.vector_store import VectorStore

    db_path = tmp_path / "test.db"
    store = VectorStore(db_path)
    yield store
    store.close()


@pytest.fixture
def mock_store():
    """Create a mock VectorStore for unit tests that don't need real DB."""
    store = MagicMock()
    store.hybrid_search.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    store.record_event.return_value = 1
    return store


@pytest.fixture
def mock_embed():
    """Create a mock embedding function returning deterministic 1024-dim vectors."""
    return MagicMock(return_value=[0.1] * 1024)


@pytest.fixture
def mock_store_with_results():
    """Mock store that returns configurable search results."""

    def _make(chunks):
        store = MagicMock()
        ids = [c.get("id", "") for c in chunks]
        docs = [c.get("content", "") for c in chunks]
        metas = [{k: v for k, v in c.items() if k not in ("id", "content", "score")} for c in chunks]
        dists = [1 - c.get("score", 0) for c in chunks]
        store.hybrid_search.return_value = {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }
        store.record_event.return_value = 1
        return store

    return _make


RESEARCH_500_WORDS = """
BrainLayer Architecture Decision Record: Vector Search Backend Selection

After extensive benchmarking across three candidate backends (sqlite-vec, Qdrant, and Weaviate),
we have decided to continue with sqlite-vec as the primary vector search engine for BrainLayer.
This decision was driven by several key factors documented below.

Performance Benchmarks:
Our testing with 226,000 chunks showed that sqlite-vec brute-force search completes in approximately
150-200ms for a single query, which is well within our 2-second SLA for the digest pipeline.
When combined with FTS5 keyword search via RRF fusion, total hybrid search latency stays under 400ms.
Binary quantization could reduce this to approximately 5ms per query, though we have not yet
implemented this optimization.

Storage Efficiency:
The current database sits at approximately 8GB for 297,000 chunks including metadata, embeddings,
and FTS indexes. Qdrant would require a separate service consuming additional memory (estimated
2-4GB RSS for this dataset size). Weaviate's JVM-based architecture would need even more.
sqlite-vec keeps everything in a single file, simplifying backup, migration, and deployment.

Operational Simplicity:
BrainLayer runs as a local-first tool on developer machines. Adding Qdrant or Weaviate would
require Docker or a separate daemon, increasing complexity for end users. sqlite-vec loads as
a SQLite extension and requires zero additional infrastructure. The APSW binding provides
robust connection management with WAL mode for concurrent reads during enrichment.

Trade-offs Accepted:
We acknowledge that sqlite-vec's brute-force approach will not scale beyond approximately
1 million chunks without binary quantization or IVF indexing. For our current use case of
300K chunks growing at roughly 50K per month, this gives us approximately 14 months before
we need to implement quantization. Additionally, sqlite-vec does not support filtered vector
search natively, so we implement filtering in application code after retrieval.

Entity Extraction Impact:
The digest pipeline's entity extraction step uses seed-based matching supplemented by local
LLM extraction when available. With the current pipeline, we extract approximately 3.2 entities
per chunk on average, with a dedup rate of roughly 15 percent across the knowledge graph.
The Phase 2 extraction pipeline handles person, company, project, and agent entity types.

Enrichment Integration:
The new enrichment controller from PR #112 provides three backends (realtime via Gemini Flash-Lite,
batch via Gemini Batch API, and local via MLX/Ollama). Content-hash dedup ensures that already-enriched
chunks are not re-processed. The digest pipeline benefits from this by automatically enriching
newly digested content in realtime mode when BRAINLAYER_DIGEST_V2 is enabled.

Conclusion:
sqlite-vec remains the right choice for BrainLayer's current scale and operational model.
We will revisit this decision when chunk count exceeds 750K or when sub-10ms latency becomes
a hard requirement for interactive search experiences.
"""


# =============================================================================
# 1. Chunk Events Audit Table Tests
# =============================================================================


class TestChunkEventsTable:
    """Tests for the chunk_events audit table in vector_store.py."""

    def test_table_exists(self, tmp_store):
        """chunk_events table is created during _init_db."""
        cursor = tmp_store.conn.cursor()
        tables = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "chunk_events" in tables

    def test_table_schema(self, tmp_store):
        """chunk_events table has the required columns."""
        cursor = tmp_store.conn.cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunk_events)")}
        assert cols >= {"id", "chunk_id", "action", "timestamp", "by_whom", "reason"}

    def test_record_event_returns_id(self, tmp_store):
        """record_event returns a positive integer row ID."""
        row_id = tmp_store.record_event(
            chunk_id="test-chunk-001",
            action="created",
            by_whom="test_suite",
            reason="Testing audit trail",
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_record_event_minimal(self, tmp_store):
        """record_event works with only required fields."""
        row_id = tmp_store.record_event(chunk_id="c-001", action="archived")
        assert row_id > 0

    def test_get_chunk_events_returns_recorded(self, tmp_store):
        """get_chunk_events returns events that were recorded."""
        tmp_store.record_event(
            chunk_id="c-002",
            action="digest_created",
            by_whom="digest_pipeline_v2",
            reason="Digested 500 chars",
        )
        events = tmp_store.get_chunk_events("c-002")
        assert len(events) == 1
        assert events[0]["chunk_id"] == "c-002"
        assert events[0]["action"] == "digest_created"
        assert events[0]["by_whom"] == "digest_pipeline_v2"
        assert events[0]["reason"] == "Digested 500 chars"
        assert events[0]["timestamp"] is not None

    def test_get_chunk_events_newest_first(self, tmp_store):
        """Events are returned newest first."""
        tmp_store.record_event(chunk_id="c-003", action="created")
        tmp_store.record_event(chunk_id="c-003", action="enriched")
        tmp_store.record_event(chunk_id="c-003", action="superseded")

        events = tmp_store.get_chunk_events("c-003")
        assert len(events) == 3
        assert events[0]["action"] == "superseded"
        assert events[2]["action"] == "created"

    def test_get_chunk_events_empty_for_unknown(self, tmp_store):
        """get_chunk_events returns empty list for unknown chunk_id."""
        events = tmp_store.get_chunk_events("nonexistent-chunk")
        assert events == []

    def test_get_chunk_events_respects_limit(self, tmp_store):
        """get_chunk_events respects the limit parameter."""
        for i in range(10):
            tmp_store.record_event(chunk_id="c-004", action=f"action_{i}")

        events = tmp_store.get_chunk_events("c-004", limit=3)
        assert len(events) == 3

    def test_events_isolated_by_chunk_id(self, tmp_store):
        """Events for different chunk_ids don't mix."""
        tmp_store.record_event(chunk_id="c-a", action="created")
        tmp_store.record_event(chunk_id="c-b", action="created")
        tmp_store.record_event(chunk_id="c-a", action="enriched")

        events_a = tmp_store.get_chunk_events("c-a")
        events_b = tmp_store.get_chunk_events("c-b")
        assert len(events_a) == 2
        assert len(events_b) == 1

    def test_indexes_exist(self, tmp_store):
        """Required indexes are created."""
        cursor = tmp_store.conn.cursor()
        indexes = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        assert "idx_chunk_events_chunk" in indexes
        assert "idx_chunk_events_action" in indexes
        assert "idx_chunk_events_timestamp" in indexes


# =============================================================================
# 2. Length-Tiered Cosine Dedup Tests
# =============================================================================


class TestLengthTieredDedup:
    """Tests for the length-tiered cosine similarity dedup logic."""

    def test_token_estimation(self):
        """Token estimation produces reasonable counts."""
        from brainlayer.pipeline.digest import _estimate_tokens

        assert _estimate_tokens("hello world") == int(2 * 1.3)
        assert _estimate_tokens("a b c d e") == int(5 * 1.3)
        assert _estimate_tokens("") == 1  # min 1

    def test_threshold_short(self):
        """Short content (<50 tokens) uses strict 0.95 threshold."""
        from brainlayer.pipeline.digest import _get_dedup_threshold

        assert _get_dedup_threshold(10) == 0.95
        assert _get_dedup_threshold(49) == 0.95

    def test_threshold_medium(self):
        """Medium content (50-200 tokens) uses 0.90 threshold."""
        from brainlayer.pipeline.digest import _get_dedup_threshold

        assert _get_dedup_threshold(50) == 0.90
        assert _get_dedup_threshold(100) == 0.90
        assert _get_dedup_threshold(200) == 0.90

    def test_threshold_long(self):
        """Long content (200+ tokens) uses 0.88 threshold."""
        from brainlayer.pipeline.digest import _get_dedup_threshold

        assert _get_dedup_threshold(201) == 0.88
        assert _get_dedup_threshold(1000) == 0.88

    def test_find_duplicates_returns_matches(self, mock_store_with_results, mock_embed):
        """find_duplicates returns chunks above the threshold."""
        from brainlayer.pipeline.digest import find_duplicates

        chunks = [
            {"id": "dup-001", "content": "matching content", "score": 0.96},
            {"id": "low-001", "content": "different stuff", "score": 0.70},
        ]
        store = mock_store_with_results(chunks)

        result = find_duplicates(
            content="short text",  # <50 tokens -> 0.95 threshold
            embedding=[0.1] * 1024,
            store=store,
        )

        assert len(result) == 1
        assert result[0]["chunk_id"] == "dup-001"
        assert result[0]["score"] >= 0.95

    def test_find_duplicates_long_content_lower_threshold(self, mock_store_with_results):
        """Long content uses lower threshold, catching more near-duplicates."""
        from brainlayer.pipeline.digest import find_duplicates

        chunks = [
            {"id": "near-dup", "content": "x " * 100, "score": 0.89},
        ]
        store = mock_store_with_results(chunks)

        long_content = "word " * 200  # ~260 tokens -> 0.88 threshold
        result = find_duplicates(
            content=long_content,
            embedding=[0.1] * 1024,
            store=store,
        )

        assert len(result) == 1
        assert result[0]["chunk_id"] == "near-dup"

    def test_find_duplicates_no_matches(self, mock_store, mock_embed):
        """find_duplicates returns empty list when nothing is close enough."""
        from brainlayer.pipeline.digest import find_duplicates

        result = find_duplicates(
            content="unique content",
            embedding=[0.1] * 1024,
            store=mock_store,
        )
        assert result == []

    def test_find_duplicates_handles_search_failure(self, mock_embed):
        """find_duplicates returns empty list on search error."""
        from brainlayer.pipeline.digest import find_duplicates

        store = MagicMock()
        store.hybrid_search.side_effect = Exception("DB locked")

        result = find_duplicates(
            content="test content",
            embedding=[0.1] * 1024,
            store=store,
        )
        assert result == []

    def test_duplicate_result_structure(self, mock_store_with_results):
        """Duplicate results contain expected fields."""
        from brainlayer.pipeline.digest import find_duplicates

        chunks = [{"id": "dup-x", "content": "exact match", "score": 0.99}]
        store = mock_store_with_results(chunks)

        result = find_duplicates(
            content="tiny",
            embedding=[0.1] * 1024,
            store=store,
        )

        assert len(result) == 1
        dup = result[0]
        assert "chunk_id" in dup
        assert "score" in dup
        assert "threshold" in dup
        assert "token_count" in dup
        assert "content_preview" in dup


# =============================================================================
# 3. Structured Report Tests
# =============================================================================


class TestStructuredReports:
    """Tests for the enhanced structured report format."""

    def test_digest_mode_report_has_required_stats(self, mock_store, mock_embed):
        """digest mode report contains all required stat fields."""
        from brainlayer.pipeline.digest import digest_content

        result = digest_content(
            content="We decided to use Groq for enrichment. Action: test it.",
            store=mock_store,
            embed_fn=mock_embed,
            faceted_enrich_fn=lambda **kw: {"status": "skipped"},
        )

        stats = result["stats"]
        assert "entities_found" in stats
        assert "chunks_created" in stats
        assert stats["chunks_created"] == 1
        assert "connections_made" in stats
        assert "supersedes_proposed" in stats

    def test_connect_mode_report_has_required_stats(self, mock_store, mock_embed):
        """connect mode report contains all required stat fields."""
        from brainlayer.pipeline.digest import digest_connect

        result = digest_connect(
            content="We decided to use JWT tokens for authentication.",
            store=mock_store,
            embed_fn=mock_embed,
        )

        stats = result["stats"]
        assert "entities_found" in stats
        assert "connections_made" in stats
        assert "chunks_created" in stats
        assert stats["chunks_created"] == 0  # connect mode doesn't create
        assert "supersedes_proposed" in stats
        assert "duplicates_found" in stats

    def test_connect_mode_returns_related_chunks(self, mock_store_with_results, mock_embed):
        """connect mode returns related_chunks key."""
        from brainlayer.pipeline.digest import digest_connect

        chunks = [
            {"id": "rel-001", "content": "related auth content", "score": 0.8, "tags": ["auth"]},
        ]
        store = mock_store_with_results(chunks)

        result = digest_connect(
            content="Authentication uses JWT tokens with 24h expiry.",
            store=store,
            embed_fn=mock_embed,
        )

        assert "related_chunks" in result
        assert "connections" in result  # backward compat alias
        assert result["related_chunks"] == result["connections"]

    def test_connect_mode_returns_duplicates(self, mock_store_with_results, mock_embed):
        """connect mode returns duplicates list."""
        from brainlayer.pipeline.digest import digest_connect

        chunks = [
            {"id": "dup-001", "content": "JWT auth decision", "score": 0.96, "tags": []},
        ]
        store = mock_store_with_results(chunks)

        result = digest_connect(
            content="JWT auth",  # short -> 0.95 threshold
            store=store,
            embed_fn=mock_embed,
        )

        assert "duplicates" in result
        assert isinstance(result["duplicates"], list)

    def test_connect_mode_returns_contradictions(self, mock_store_with_results, mock_embed):
        """connect mode returns contradictions list."""
        from brainlayer.pipeline.digest import digest_connect

        result = digest_connect(
            content="We decided to use JWT tokens.",
            store=mock_store_with_results([]),
            embed_fn=mock_embed,
        )

        assert "contradictions" in result
        assert isinstance(result["contradictions"], list)


# =============================================================================
# 4. Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests that existing behavior is preserved."""

    def test_digest_mode_default_is_unchanged(self, mock_store, mock_embed):
        """Default digest mode still creates chunk and returns digest_id."""
        from brainlayer.pipeline.digest import digest_content

        result = digest_content(
            content="Test content for backward compatibility.",
            store=mock_store,
            embed_fn=mock_embed,
            faceted_enrich_fn=lambda **kw: {"status": "skipped"},
        )

        assert "digest_id" in result
        assert result["digest_id"].startswith("digest-")
        assert "summary" in result
        assert "entities" in result
        assert "relations" in result
        assert "sentiment" in result
        mock_store.upsert_chunks.assert_called_once()

    def test_connect_mode_still_proposal(self, mock_store, mock_embed):
        """Connect mode still returns proposal status."""
        from brainlayer.pipeline.digest import digest_connect

        result = digest_connect(
            content="Test proposal content.",
            store=mock_store,
            embed_fn=mock_embed,
        )

        assert result["status"] == "proposal"
        mock_store.upsert_chunks.assert_not_called()

    def test_connect_mode_old_fields_still_present(self, mock_store, mock_embed):
        """Connect mode still has all original fields."""
        from brainlayer.pipeline.digest import digest_connect

        result = digest_connect(
            content="Decided to switch from cookies to JWT.",
            store=mock_store,
            embed_fn=mock_embed,
        )

        assert "extracted" in result
        assert "connections" in result
        assert "contradictions" in result
        assert "supersede_proposals" in result
        assert "suggested_stores" in result
        assert "stats" in result

    def test_empty_content_still_raises(self, mock_store, mock_embed):
        """Empty content still raises ValueError in both modes."""
        from brainlayer.pipeline.digest import digest_connect, digest_content

        with pytest.raises(ValueError, match="non-empty"):
            digest_content(content="", store=mock_store, embed_fn=mock_embed)

        with pytest.raises(ValueError, match="non-empty"):
            digest_connect(content="  ", store=mock_store, embed_fn=mock_embed)


# =============================================================================
# 5. Feature Flag Tests
# =============================================================================


class TestFeatureFlag:
    """Tests for the BRAINLAYER_DIGEST_V2 feature flag."""

    def test_v2_enabled_by_default(self):
        """DIGEST_V2 is enabled by default."""
        from brainlayer.pipeline.digest import DIGEST_V2_ENABLED

        assert DIGEST_V2_ENABLED is True

    def test_v2_disabled_skips_dedup(self, mock_store, mock_embed):
        """When V2 is disabled, digest_content skips dedup."""
        import brainlayer.pipeline.digest as digest_mod

        original = digest_mod.DIGEST_V2_ENABLED
        try:
            digest_mod.DIGEST_V2_ENABLED = False

            result = digest_mod.digest_content(
                content="Test content without dedup.",
                store=mock_store,
                embed_fn=mock_embed,
                faceted_enrich_fn=lambda **kw: {"status": "skipped"},
            )

            assert "duplicates" not in result
            mock_store.record_event.assert_not_called()
        finally:
            digest_mod.DIGEST_V2_ENABLED = original

    def test_v2_enabled_runs_dedup(self, mock_store, mock_embed):
        """When V2 is enabled, digest_content runs dedup and records events."""
        import brainlayer.pipeline.digest as digest_mod

        original = digest_mod.DIGEST_V2_ENABLED
        try:
            digest_mod.DIGEST_V2_ENABLED = True

            result = digest_mod.digest_content(
                content="Test content with dedup enabled.",
                store=mock_store,
                embed_fn=mock_embed,
                faceted_enrich_fn=lambda **kw: {"status": "skipped"},
            )

            mock_store.record_event.assert_called_once()
            call_args = mock_store.record_event.call_args
            assert call_args.kwargs["action"] == "digest_created"
            assert call_args.kwargs["by_whom"] == "digest_pipeline_v2"
        finally:
            digest_mod.DIGEST_V2_ENABLED = original


# =============================================================================
# 6. MCP Handler Routing Tests
# =============================================================================


class TestMCPHandlerRouting:
    """Tests for the MCP handler routing to v2 pipeline."""

    @pytest.mark.asyncio
    async def test_connect_mode_valid(self):
        """mode='connect' passes validation."""
        from brainlayer.mcp.store_handler import _brain_digest

        result = await _brain_digest(content=None, mode="connect")
        error_text = result.content[0].text
        assert "content is required" in error_text
        assert "Unknown" not in error_text

    @pytest.mark.asyncio
    async def test_digest_mode_valid(self):
        """mode='digest' passes validation."""
        from brainlayer.mcp.store_handler import _brain_digest

        result = await _brain_digest(content=None, mode="digest")
        error_text = result.content[0].text
        assert "content is required" in error_text

    @pytest.mark.asyncio
    async def test_invalid_mode_rejected(self):
        """Invalid mode is rejected."""
        from brainlayer.mcp.store_handler import _brain_digest

        result = await _brain_digest(content="test", mode="bogus")
        error_text = result.content[0].text
        assert "Unknown" in error_text


# =============================================================================
# 7. Integration Test with 500-word Research Output
# =============================================================================


class TestIntegration:
    """Integration tests using a real VectorStore (tmp_path isolated)."""

    def test_digest_500_word_research(self, tmp_store):
        """Digest a 500-word research output end-to-end with real DB."""
        from brainlayer.pipeline.digest import digest_content

        mock_embed = MagicMock(return_value=[0.05] * 1024)
        def mock_faceted(**kw):
            return {
                "status": "enriched",
                "topics": ["vector-search", "sqlite-vec"],
                "activity": "act:deciding",
                "domains": ["dom:database", "dom:python"],
                "confidence": 0.85,
            }

        result = digest_content(
            content=RESEARCH_500_WORDS,
            store=tmp_store,
            embed_fn=mock_embed,
            title="Vector Search Backend Selection ADR",
            project="brainlayer",
            faceted_enrich_fn=mock_faceted,
        )

        assert result["digest_id"].startswith("digest-")
        assert result["summary"] == "Vector Search Backend Selection ADR"
        assert "vector-search" in result["tags"]
        assert result["stats"]["entities_found"] >= 0
        assert result["stats"]["chunks_created"] == 1

        chunk = tmp_store.get_chunk(result["digest_id"])
        assert chunk is not None
        assert chunk["project"] == "brainlayer"

        events = tmp_store.get_chunk_events(result["digest_id"])
        assert len(events) >= 1
        assert events[0]["action"] == "digest_created"

    def test_connect_500_word_research(self, tmp_store):
        """Connect mode with 500-word research returns valid proposal."""
        from brainlayer.pipeline.digest import digest_connect

        mock_embed = MagicMock(return_value=[0.05] * 1024)

        result = digest_connect(
            content=RESEARCH_500_WORDS,
            store=tmp_store,
            embed_fn=mock_embed,
            title="Vector Search Backend Selection",
            project="brainlayer",
        )

        assert result["status"] == "proposal"
        assert "extracted" in result
        assert "related_chunks" in result
        assert "duplicates" in result
        assert result["stats"]["chunks_created"] == 0

    def test_digest_records_audit_trail(self, tmp_store):
        """Digest creates an audit trail in chunk_events."""
        from brainlayer.pipeline.digest import digest_content

        mock_embed = MagicMock(return_value=[0.05] * 1024)

        result = digest_content(
            content="Short test content for audit trail verification.",
            store=tmp_store,
            embed_fn=mock_embed,
            faceted_enrich_fn=lambda **kw: {"status": "skipped"},
        )

        events = tmp_store.get_chunk_events(result["digest_id"])
        assert len(events) == 1
        assert events[0]["action"] == "digest_created"
        assert "digest_pipeline_v2" in events[0]["by_whom"]

    def test_digest_then_digest_finds_duplicate(self, tmp_store):
        """Second digest of similar content detects the first as a duplicate."""
        from brainlayer.pipeline.digest import digest_content

        content = "We decided to use sqlite-vec for vector search in BrainLayer."
        mock_embed = MagicMock(return_value=[0.1] * 1024)

        result1 = digest_content(
            content=content,
            store=tmp_store,
            embed_fn=mock_embed,
            faceted_enrich_fn=lambda **kw: {"status": "skipped"},
        )

        result2 = digest_content(
            content=content + " This is slightly modified.",
            store=tmp_store,
            embed_fn=mock_embed,
            faceted_enrich_fn=lambda **kw: {"status": "skipped"},
        )

        assert result1["digest_id"] != result2["digest_id"]
        assert "duplicates" in result2, "V2 should include duplicates key"
        if result2["duplicates"]:
            dup_ids = [d["chunk_id"] for d in result2["duplicates"]]
            assert result1["digest_id"] in dup_ids, (
                f"Expected first digest {result1['digest_id']} in duplicates {dup_ids}"
            )


# =============================================================================
# 8. Performance Target Tests
# =============================================================================


class TestPerformance:
    """Tests that the digest pipeline meets performance targets."""

    def test_digest_under_2_seconds(self, tmp_store):
        """Digest of 1000-word content completes in <2 seconds."""
        from brainlayer.pipeline.digest import digest_content

        content = " ".join(["word"] * 1000)
        mock_embed = MagicMock(return_value=[0.05] * 1024)

        start = time.monotonic()
        digest_content(
            content=content,
            store=tmp_store,
            embed_fn=mock_embed,
            faceted_enrich_fn=lambda **kw: {"status": "skipped"},
        )
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"Digest took {elapsed:.2f}s, expected <2s"

    def test_connect_under_2_seconds(self, tmp_store):
        """Connect mode for 1000-word content completes in <2 seconds."""
        from brainlayer.pipeline.digest import digest_connect

        content = " ".join(["word"] * 1000)
        mock_embed = MagicMock(return_value=[0.05] * 1024)

        start = time.monotonic()
        digest_connect(
            content=content,
            store=tmp_store,
            embed_fn=mock_embed,
        )
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"Connect took {elapsed:.2f}s, expected <2s"


# =============================================================================
# 9. Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_importance_heuristic_capped_at_10(self):
        """Auto-importance never exceeds 10."""
        from brainlayer.pipeline.digest import _auto_propose_importance

        entities = [{"confidence": 0.95}] * 10
        decisions = ["d1", "d2", "d3"]
        actions = ["a1", "a2"]
        score = _auto_propose_importance(entities, decisions, actions)
        assert score <= 10

    def test_dedup_threshold_boundary_50(self):
        """Token count exactly at 50 uses medium threshold."""
        from brainlayer.pipeline.digest import _get_dedup_threshold

        assert _get_dedup_threshold(50) == 0.90

    def test_dedup_threshold_boundary_200(self):
        """Token count exactly at 200 uses medium threshold."""
        from brainlayer.pipeline.digest import _get_dedup_threshold

        assert _get_dedup_threshold(200) == 0.90

    def test_dedup_threshold_boundary_201(self):
        """Token count at 201 uses long threshold."""
        from brainlayer.pipeline.digest import _get_dedup_threshold

        assert _get_dedup_threshold(201) == 0.88

    def test_find_duplicates_excludes_self(self, mock_store_with_results):
        """find_duplicates excludes chunk IDs in exclude_ids."""
        from brainlayer.pipeline.digest import find_duplicates

        chunks = [
            {"id": "self-chunk", "content": "exact same", "score": 0.99},
            {"id": "other-chunk", "content": "also similar", "score": 0.97},
        ]
        store = mock_store_with_results(chunks)

        result = find_duplicates(
            content="tiny",
            embedding=[0.1] * 1024,
            store=store,
            exclude_ids={"self-chunk"},
        )

        dup_ids = [d["chunk_id"] for d in result]
        assert "self-chunk" not in dup_ids
        assert "other-chunk" in dup_ids
