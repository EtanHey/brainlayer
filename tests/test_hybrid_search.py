"""Tests for binary-quantized RRF hybrid search."""

import json
import uuid

import apsw
import pytest

from brainlayer._helpers import serialize_f32
from brainlayer.search_repo import _contains_precompact_or_quarantined_meta, _has_recency_intent, _hybrid_cache
from brainlayer.vector_store import VectorStore


@pytest.fixture(autouse=True)
def clear_hybrid_cache():
    """Keep module-level hybrid cache isolated across tests."""
    _hybrid_cache.clear()
    yield
    _hybrid_cache.clear()


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore on a temporary DB."""
    s = VectorStore(tmp_path / "hybrid.db")
    yield s
    s.close()


def _embed(text: str) -> list[float]:
    """Deterministic 1024-dim embedding for test data."""
    seed = (sum(ord(c) for c in text[:40]) % 97) / 1000.0
    return [seed + (i / 10000.0) for i in range(1024)]


def _insert_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    content: str,
    embedding: list[float],
    metadata_obj: dict | None = None,
    summary: str | None = None,
    tags: list[str] | None = None,
    resolved_query: str | None = None,
    importance: float | None = None,
    created_at: str | None = "2026-04-05T00:00:00Z",
    content_type: str = "assistant_text",
    project: str = "hybrid-test",
    sender: str | None = None,
    language: str | None = None,
):
    """Insert a chunk and its float vector directly."""
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, sender, language, summary, tags, resolved_query, importance, created_at
        ) VALUES (?, ?, ?, 'test.jsonl', ?, ?, ?, 'claude_code', ?, ?, ?, ?, ?, ?, ?)""",
        (
            chunk_id,
            content,
            json.dumps(metadata_obj) if metadata_obj is not None else "{}",
            project,
            content_type,
            len(content),
            sender,
            language,
            summary,
            json.dumps(tags) if tags else None,
            resolved_query,
            importance,
            created_at,
        ),
    )
    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_f32(embedding)),
    )


def _chunk_vector_count(store: VectorStore, table: str) -> int:
    cursor = store.conn.cursor()
    return cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class TestBinaryIndexLifecycle:
    def test_binary_table_created(self, store):
        cursor = store.conn.cursor()
        tables = {
            row[0]
            for row in cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table' OR type = 'virtual table'")
        }
        assert "chunk_vectors_binary" in tables

    def test_build_binary_index(self, store):
        _insert_chunk(store, chunk_id="chunk-a", content="alpha", embedding=_embed("alpha"))
        _insert_chunk(store, chunk_id="chunk-b", content="beta", embedding=_embed("beta"))

        store.build_binary_index()

        assert _chunk_vector_count(store, "chunk_vectors") == 2
        assert _chunk_vector_count(store, "chunk_vectors_binary") == 2

    def test_new_chunks_inserted_into_both_tables(self, store):
        chunk_id = f"chunk-{uuid.uuid4().hex[:8]}"
        store.upsert_chunks(
            [
                {
                    "id": chunk_id,
                    "content": "binary dual insert test",
                    "metadata": {},
                    "source_file": "test.jsonl",
                    "project": "hybrid-test",
                    "content_type": "assistant_text",
                    "char_count": 23,
                    "source": "claude_code",
                    "created_at": "2026-04-05T00:00:00Z",
                }
            ],
            [_embed("dual insert")],
        )

        cursor = store.conn.cursor()
        float_rows = cursor.execute("SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,)).fetchone()[0]
        binary_rows = cursor.execute(
            "SELECT COUNT(*) FROM chunk_vectors_binary WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()[0]
        assert float_rows == 1
        assert binary_rows == 1


class TestHybridSearch:
    def test_hybrid_search_returns_results(self, store):
        query_embedding = _embed("hybrid auth query")
        _insert_chunk(
            store,
            chunk_id="both-match",
            content="hybrid auth query with reciprocal rank fusion",
            summary="RRF search design",
            tags=["search", "rrf"],
            embedding=query_embedding,
            importance=5.0,
        )
        _insert_chunk(
            store,
            chunk_id="noise",
            content="unrelated gardening notes",
            embedding=_embed("gardening"),
        )
        store.build_binary_index()
        store.conn.cursor().execute("DELETE FROM chunk_vectors")

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="hybrid auth query",
            n_results=5,
        )

        assert results["ids"][0]
        assert results["ids"][0][0] == "both-match"

    def test_hybrid_search_rrf_scoring(self, store):
        query_embedding = _embed("search fusion")
        _insert_chunk(
            store,
            chunk_id="both",
            content="search fusion architecture and ranking details",
            embedding=query_embedding,
            importance=5.0,
        )
        _insert_chunk(
            store,
            chunk_id="fts-only",
            content="search fusion exact keywords but distant embedding",
            embedding=_embed("very distant vector"),
            importance=5.0,
        )
        _insert_chunk(
            store,
            chunk_id="vec-only",
            content="semantic neighbor without exact keywords present",
            embedding=[v + 0.00005 for v in query_embedding],
            importance=5.0,
        )
        store.build_binary_index()
        store.conn.cursor().execute("DELETE FROM chunk_vectors")

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="search fusion",
            n_results=3,
        )

        ids = results["ids"][0]
        assert ids[0] == "both", ids
        assert set(ids) == {"both", "fts-only", "vec-only"}

    def test_hybrid_search_fts_only_fallback(self, store):
        _insert_chunk(
            store,
            chunk_id="fts-hit",
            content="exact keyword fallback for full text only",
            embedding=_embed("distant vector"),
        )
        store.build_binary_index()
        cursor = store.conn.cursor()
        cursor.execute("DELETE FROM chunk_vectors")
        cursor.execute("DELETE FROM chunk_vectors_binary")

        results = store.hybrid_search(
            query_embedding=_embed("nothing close"),
            query_text="keyword fallback",
            n_results=5,
        )

        assert "fts-hit" in results["ids"][0]

    def test_hybrid_search_fts_only_respects_sender_and_language_filters(self, store):
        _insert_chunk(
            store,
            chunk_id="sender-lang-hit",
            content="keyword fallback sender language exact hit",
            embedding=_embed("distant vector"),
            sender="etan",
            language="he",
        )
        _insert_chunk(
            store,
            chunk_id="sender-lang-miss",
            content="keyword fallback sender language exact hit",
            embedding=_embed("another distant vector"),
            sender="other",
            language="en",
        )
        store.build_binary_index()
        cursor = store.conn.cursor()
        cursor.execute("DELETE FROM chunk_vectors")
        cursor.execute("DELETE FROM chunk_vectors_binary")

        results = store.hybrid_search(
            query_embedding=_embed("nothing close"),
            query_text="keyword fallback sender language",
            n_results=5,
            sender_filter="etan",
            language_filter="he",
        )

        assert results["ids"][0] == ["sender-lang-hit"]

    def test_hybrid_search_trigram_only_respects_project_filter(self, store):
        _insert_chunk(
            store,
            chunk_id="trigram-hit",
            content="stalker-golem queue note",
            embedding=_embed("distant vector"),
            project="other-project",
        )
        store.build_binary_index()
        cursor = store.conn.cursor()
        cursor.execute("DELETE FROM chunk_vectors")
        cursor.execute("DELETE FROM chunk_vectors_binary")

        results = store.hybrid_search(
            query_embedding=_embed("nothing close"),
            query_text="alker-go",
            n_results=5,
            project_filter="brainlayer",
        )

        assert results["ids"][0] == []

    def test_hybrid_search_vec_only(self, store):
        query_embedding = _embed("vector only query")
        _insert_chunk(
            store,
            chunk_id="vec-hit",
            content="content without overlapping keywords",
            embedding=query_embedding,
        )
        store.build_binary_index()
        store.conn.cursor().execute("DELETE FROM chunk_vectors")

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="keywords not present anywhere",
            n_results=5,
        )

        assert "vec-hit" in results["ids"][0]

    def test_hybrid_search_filters_meta_noise_by_default(self, store):
        query_embedding = _embed("brain search prompt")
        _insert_chunk(
            store,
            chunk_id="meta-noise",
            content="brain_search(query='auth') returned a tool transcript block",
            embedding=query_embedding,
            importance=5.0,
        )
        _insert_chunk(
            store,
            chunk_id="real-hit",
            content="authentication decision: use sqlite session tokens with refresh rotation",
            embedding=[v + 0.00005 for v in query_embedding],
            importance=5.0,
        )
        store.build_binary_index()
        store.conn.cursor().execute("DELETE FROM chunk_vectors")

        filtered = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="brain search prompt",
            n_results=5,
        )
        unfiltered = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="brain search prompt",
            n_results=5,
            filter_meta_noise=False,
        )

        assert "meta-noise" not in filtered["ids"][0]
        assert "real-hit" in filtered["ids"][0]
        assert "meta-noise" in unfiltered["ids"][0]

    def test_hybrid_search_filters_meta_noise_case_insensitively(self, store):
        query_embedding = _embed("entity lookup prompt")
        _insert_chunk(
            store,
            chunk_id="meta-noise-upper",
            content="BRAIN_ENTITY(query='Avi') returned a tool transcript block",
            embedding=query_embedding,
            importance=5.0,
        )
        _insert_chunk(
            store,
            chunk_id="real-hit-lower",
            content="avi profile: whatsapp name is aviel and taba is inactive",
            embedding=[v + 0.00005 for v in query_embedding],
            importance=5.0,
        )
        store.build_binary_index()
        store.conn.cursor().execute("DELETE FROM chunk_vectors")

        filtered = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="entity lookup prompt",
            n_results=5,
        )

        assert "meta-noise-upper" not in filtered["ids"][0]
        assert "real-hit-lower" in filtered["ids"][0]

    def test_hybrid_search_demotes_precompact_and_quarantined_tagged_chunks(self, store):
        query_embedding = _embed("noise-aware rerank control")

        _insert_chunk(
            store,
            chunk_id="precompact-noise",
            content="noise-aware rerank control with direct match",
            embedding=query_embedding,
            tags=["precompact-checkpoint"],
            created_at="2026-05-16T00:00:00Z",
            importance=1.0,
        )
        _insert_chunk(
            store,
            chunk_id="quarantined-noise",
            content="noise-aware rerank control with direct match",
            embedding=query_embedding,
            tags=["quarantined"],
            created_at="2026-05-16T00:00:00Z",
            importance=1.0,
        )
        _insert_chunk(
            store,
            chunk_id="clean-hit",
            content="noise-aware rerank control semantic anchor phrase",
            embedding=[value + 0.0002 for value in query_embedding],
            created_at="2026-05-17T00:00:00Z",
            importance=10.0,
        )
        store.build_binary_index()

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="noise-aware rerank control",
            n_results=3,
            include_archived=False,
        )

        ids = results["ids"][0]
        assert ids[0] == "clean-hit", ids
        assert set(ids[1:]) == {"precompact-noise", "quarantined-noise"}, ids

    def test_hybrid_search_demotes_chunks_with_quarantine_metadata(self, store):
        query_embedding = _embed("metadata noise rerank")
        clean_content = "metadata noise rerank control anchor"
        noisy_content = "metadata noise rerank control anchor"

        _insert_chunk(
            store,
            chunk_id="meta-hit",
            content=clean_content,
            embedding=[value + 0.00002 for value in query_embedding],
            created_at="2026-05-17T00:00:00Z",
            importance=10.0,
        )
        _insert_chunk(
            store,
            chunk_id="meta-flag-noise",
            content=noisy_content,
            embedding=query_embedding,
            created_at="2026-05-16T00:00:00Z",
            importance=10.0,
            metadata_obj={"quarantine": True},
            tags=["search"],
        )
        store.build_binary_index()

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="metadata noise rerank",
            n_results=2,
            include_archived=False,
        )

        ids = results["ids"][0]
        assert ids[0] == "meta-hit", ids
        assert ids[1] == "meta-flag-noise", ids

    def test_noise_demoter_does_not_treat_arbitrary_true_metadata_as_quarantine(self):
        assert not _contains_precompact_or_quarantined_meta({"feature_enabled": "true"}, "ordinary content")

    def test_recency_intent_fallback_preserves_entity_filter(self, store):
        store.upsert_entity("person-a", "person", "Person A")
        store.upsert_entity("person-b", "person", "Person B")
        _insert_chunk(
            store,
            chunk_id="recent-a",
            content="fresh unrelated notes for person a",
            embedding=_embed("fresh a"),
            created_at="2999-01-01T00:00:00Z",
        )
        _insert_chunk(
            store,
            chunk_id="recent-b",
            content="fresh unrelated notes for person b",
            embedding=_embed("fresh b"),
            created_at="2999-01-01T00:00:00Z",
        )
        store.link_entity_chunk("person-a", "recent-a")
        store.link_entity_chunk("person-b", "recent-b")

        results = store.hybrid_search(
            query_embedding=_embed("latest work"),
            query_text="latest work",
            n_results=5,
            entity_id="person-a",
        )

        assert "recent-a" in results["ids"][0]
        assert "recent-b" not in results["ids"][0]

    def test_recency_intent_fallback_preserves_sentiment_filter(self, store):
        _insert_chunk(
            store,
            chunk_id="recent-positive",
            content="fresh unrelated positive notes",
            embedding=_embed("fresh positive"),
            created_at="2999-01-01T00:00:00Z",
        )
        _insert_chunk(
            store,
            chunk_id="recent-negative",
            content="fresh unrelated negative notes",
            embedding=_embed("fresh negative"),
            created_at="2999-01-01T00:00:00Z",
        )
        cursor = store.conn.cursor()
        cursor.execute("UPDATE chunks SET sentiment_label = 'positive' WHERE id = 'recent-positive'")
        cursor.execute("UPDATE chunks SET sentiment_label = 'negative' WHERE id = 'recent-negative'")

        results = store.hybrid_search(
            query_embedding=_embed("latest work"),
            query_text="latest work",
            n_results=5,
            sentiment_filter="positive",
        )

        assert "recent-positive" in results["ids"][0]
        assert "recent-negative" not in results["ids"][0]

    def test_recency_intent_fallback_preserves_content_type_filter(self, store, monkeypatch):
        _insert_chunk(
            store,
            chunk_id="recent-ai-code",
            content="fresh unrelated code notes",
            embedding=_embed("fresh code"),
            created_at="2999-01-01T00:00:00Z",
            content_type="ai_code",
        )
        _insert_chunk(
            store,
            chunk_id="recent-assistant",
            content="fresh unrelated assistant notes",
            embedding=_embed("fresh assistant"),
            created_at="2999-01-01T00:00:00Z",
            content_type="assistant_text",
        )
        monkeypatch.setattr(
            store,
            "search",
            lambda **_kwargs: {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]},
        )

        results = store.hybrid_search(
            query_embedding=_embed("latest work"),
            query_text="latest work",
            n_results=5,
            content_type_filter="ai_code",
        )

        assert "recent-ai-code" in results["ids"][0]
        assert "recent-assistant" not in results["ids"][0]

    def test_recency_intent_uses_term_boundaries(self):
        assert _has_recency_intent("latest work progress")
        assert _has_recency_intent("what changed this week")
        assert not _has_recency_intent("concurrent writer arbitration")
        assert not _has_recency_intent("recurrent pattern analysis")

    def test_recency_intent_fallback_compares_created_at_as_datetime(self, store, monkeypatch):
        cutoff_date = store.conn.cursor().execute("SELECT date('now', '-7 days')").fetchone()[0]
        _insert_chunk(
            store,
            chunk_id="stale-boundary",
            content="stale boundary payload",
            embedding=_embed("stale boundary"),
            created_at=f"{cutoff_date}T00:00:00Z",
        )
        monkeypatch.setattr(
            store,
            "search",
            lambda **_kwargs: {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]},
        )

        results = store.hybrid_search(
            query_embedding=_embed("latest unrelated"),
            query_text="latest unrelated",
            n_results=5,
        )

        assert "stale-boundary" not in results["ids"][0]

    def test_recency_intent_fallback_respects_date_to(self, store, monkeypatch):
        cursor = store.conn.cursor()
        before_date_to = cursor.execute("SELECT datetime('now', '-2 days')").fetchone()[0].replace(" ", "T") + "Z"
        date_to = cursor.execute("SELECT datetime('now', '-1 days')").fetchone()[0].replace(" ", "T") + "Z"
        after_date_to = cursor.execute("SELECT datetime('now')").fetchone()[0].replace(" ", "T") + "Z"
        _insert_chunk(
            store,
            chunk_id="recent-before-date-to",
            content="fresh before date_to payload",
            embedding=_embed("fresh before date to"),
            created_at=before_date_to,
        )
        _insert_chunk(
            store,
            chunk_id="recent-after-date-to",
            content="fresh after date_to payload",
            embedding=_embed("fresh after date to"),
            created_at=after_date_to,
        )
        monkeypatch.setattr(
            store,
            "search",
            lambda **_kwargs: {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]},
        )

        results = store.hybrid_search(
            query_embedding=_embed("latest unrelated"),
            query_text="latest unrelated",
            n_results=5,
            date_to=date_to,
        )

        assert "recent-before-date-to" in results["ids"][0]
        assert "recent-after-date-to" not in results["ids"][0]

    def test_recency_intent_fallback_retries_busy_query(self, store, monkeypatch):
        _insert_chunk(
            store,
            chunk_id="recent-after-busy",
            content="fresh after transient busy payload",
            embedding=_embed("fresh busy"),
            created_at="2999-01-01T00:00:00Z",
        )
        monkeypatch.setattr(
            store,
            "search",
            lambda **_kwargs: {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]},
        )
        real_cursor = store.conn.cursor()

        class BusyOnceCursor:
            def __init__(self):
                self.busy_count = 0

            def execute(self, sql, params=()):
                if "WHERE datetime(created_at) >= datetime('now', '-7 days')" in str(sql) and self.busy_count == 0:
                    self.busy_count += 1
                    raise apsw.BusyError("database is locked")
                return real_cursor.execute(sql, params)

        busy_cursor = BusyOnceCursor()
        sleeps: list[float] = []
        monkeypatch.setattr(store, "_read_cursor", lambda: busy_cursor)
        monkeypatch.setattr("time.sleep", sleeps.append)

        results = store.hybrid_search(
            query_embedding=_embed("latest unrelated"),
            query_text="latest unrelated",
            n_results=5,
        )

        assert busy_cursor.busy_count == 1
        assert sleeps == [0.05]
        assert "recent-after-busy" in results["ids"][0]

    def test_mmr_rerank_dedupes_near_duplicates(self, store, monkeypatch):
        monkeypatch.setattr("brainlayer.search_repo._MMR_LAMBDA", 0.65)

        def embedding(primary: float, secondary: float = 0.0) -> list[float]:
            vector = [0.0] * 1024
            vector[0] = primary
            vector[1] = secondary
            return vector

        _insert_chunk(
            store,
            chunk_id="dup-primary",
            content="oauth token rotation incident rollback and session repair",
            embedding=embedding(1.0, 0.0),
            importance=6.0,
        )
        _insert_chunk(
            store,
            chunk_id="dup-secondary",
            content="oauth token rotation incident rollback and session repair duplicate notes",
            embedding=embedding(0.999, 0.001),
            importance=6.0,
        )
        _insert_chunk(
            store,
            chunk_id="distinct-relevant",
            content="oauth token rotation migration checklist and recovery guide",
            embedding=embedding(0.72, 0.69),
            importance=5.5,
        )
        _insert_chunk(
            store,
            chunk_id="distinct-supporting",
            content="oauth token rotation audit trail and operator runbook",
            embedding=embedding(0.65, 0.75),
            importance=5.0,
        )

        scored = [
            (0.99, "dup-primary", "oauth token rotation incident rollback and session repair", {}, 0.01),
            (
                0.98,
                "dup-secondary",
                "oauth token rotation incident rollback and session repair duplicate notes",
                {},
                0.02,
            ),
            (0.94, "distinct-relevant", "oauth token rotation migration checklist and recovery guide", {}, 0.12),
            (0.9, "distinct-supporting", "oauth token rotation audit trail and operator runbook", {}, 0.15),
        ]

        reranked = store._mmr_rerank_scored_results(scored, n_results=3)
        ids = [item[1] for item in reranked[:3]]

        assert ids[0] == "dup-primary", ids
        assert "distinct-relevant" in ids[:2], ids
        assert set(ids[:2]) != {"dup-primary", "dup-secondary"}, ids

    def test_mmr_rerank_keeps_nonvector_hits_in_original_score_slots(self, store, monkeypatch):
        monkeypatch.setattr("brainlayer.search_repo._MMR_LAMBDA", 0.65)

        def embedding(primary: float, secondary: float = 0.0) -> list[float]:
            vector = [0.0] * 1024
            vector[0] = primary
            vector[1] = secondary
            return vector

        cursor = store.conn.cursor()
        cursor.execute(
            """
            INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type,
                char_count, source, created_at
            ) VALUES (?, ?, '{}', 'test.jsonl', 'hybrid-test', 'assistant_text', ?, 'claude_code', ?)
            """,
            (
                "lexical-only",
                "oauth token rotation troubleshooting transcript from a manual note",
                60,
                "2026-04-05T00:00:00Z",
            ),
        )
        _insert_chunk(
            store,
            chunk_id="dup-primary",
            content="oauth token rotation incident rollback and session repair",
            embedding=embedding(1.0, 0.0),
            importance=6.0,
        )
        _insert_chunk(
            store,
            chunk_id="dup-secondary",
            content="oauth token rotation incident rollback and session repair duplicate notes",
            embedding=embedding(0.999, 0.001),
            importance=6.0,
        )
        _insert_chunk(
            store,
            chunk_id="distinct-relevant",
            content="oauth token rotation migration checklist and recovery guide",
            embedding=embedding(0.72, 0.69),
            importance=5.5,
        )

        scored = [
            (1.0, "lexical-only", "oauth token rotation troubleshooting transcript from a manual note", {}, 0.0),
            (0.99, "dup-primary", "oauth token rotation incident rollback and session repair", {}, 0.01),
            (
                0.98,
                "dup-secondary",
                "oauth token rotation incident rollback and session repair duplicate notes",
                {},
                0.02,
            ),
            (0.94, "distinct-relevant", "oauth token rotation migration checklist and recovery guide", {}, 0.12),
        ]

        reranked = store._mmr_rerank_scored_results(scored, n_results=2)
        ids = [item[1] for item in reranked[:2]]

        assert ids[0] == "lexical-only", ids
        assert "dup-secondary" not in ids, ids
