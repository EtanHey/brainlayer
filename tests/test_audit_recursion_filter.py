import json

from brainlayer._helpers import serialize_f32
from brainlayer.vector_store import VectorStore


def _insert_chunk(store: VectorStore, chunk_id: str, content: str, tags: list[str], embedding: list[float]) -> None:
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (id, content, metadata, source_file, project,
           content_type, char_count, source, tags)
           VALUES (?, ?, '{}', 'audit-filter-test.jsonl', 'brainlayer',
                   'assistant_text', ?, 'claude_code', ?)""",
        (chunk_id, content, len(content), json.dumps(tags)),
    )
    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_f32(embedding)),
    )


def test_hybrid_search_excludes_audit_recursion_by_default(tmp_path):
    store = VectorStore(tmp_path / "audit-filter.db")
    try:
        query_embedding = [0.01] * 1024
        _insert_chunk(
            store,
            "audit-recursion-source",
            "why restart BrainBar audit recursion contamination exact match",
            ["r02", "audit"],
            query_embedding,
        )
        _insert_chunk(
            store,
            "ordinary-brainbar-memory",
            "why restart BrainBar because launchd replaced the old degraded binary",
            ["brainbar", "reliability"],
            [0.02] * 1024,
        )

        default_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="why restart BrainBar",
            n_results=3,
        )
        default_ids = default_results["ids"][0]

        assert "audit-recursion-source" not in default_ids
        assert "ordinary-brainbar-memory" in default_ids

        audit_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="why restart BrainBar",
            n_results=3,
            include_audit=True,
        )

        assert "audit-recursion-source" in audit_results["ids"][0]
    finally:
        store.close()


def test_hybrid_search_does_not_exclude_r0x_substrings_inside_normal_tags(tmp_path):
    store = VectorStore(tmp_path / "audit-filter-substring.db")
    try:
        query_embedding = [0.03] * 1024
        _insert_chunk(
            store,
            "ordinary-mirror07-memory",
            "mirror07 normal operational memory should remain searchable",
            ["mirror07", "reliability"],
            query_embedding,
        )

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="mirror07 normal operational memory",
            n_results=3,
        )

        assert "ordinary-mirror07-memory" in results["ids"][0]
    finally:
        store.close()
