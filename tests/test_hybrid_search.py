"""Tests for binary-quantized RRF hybrid search."""

import json
import uuid

import pytest

from brainlayer._helpers import serialize_f32
from brainlayer.search_repo import _hybrid_cache
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
    summary: str | None = None,
    tags: list[str] | None = None,
    resolved_query: str | None = None,
    importance: float | None = None,
    created_at: str | None = "2026-04-05T00:00:00Z",
    project: str = "hybrid-test",
):
    """Insert a chunk and its float vector directly."""
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, summary, tags, resolved_query, importance, created_at
        ) VALUES (?, ?, '{}', 'test.jsonl', ?, 'assistant_text', ?, 'claude_code', ?, ?, ?, ?, ?)""",
        (
            chunk_id,
            content,
            project,
            len(content),
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

    def test_mmr_rerank_dedupes_near_duplicates(self, store):
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

    def test_mmr_rerank_keeps_nonvector_hits_in_original_score_slots(self, store):
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
