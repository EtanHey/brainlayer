import json

from brainlayer._helpers import serialize_f32
from brainlayer.vector_store import VectorStore


def _embed(seed_text: str) -> list[float]:
    seed = (sum(ord(ch) for ch in seed_text[:40]) % 97) / 1000.0
    return [seed + (i / 10000.0) for i in range(1024)]


def _insert_chunk(store: VectorStore, *, chunk_id: str, content: str) -> None:
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, summary, tags, created_at
        ) VALUES (?, ?, '{}', 'test.jsonl', 'brainlayer', 'assistant_text', ?, 'manual', ?, ?, ?)""",
        (
            chunk_id,
            content,
            len(content),
            content,
            json.dumps(["fts"]),
            "2026-04-30T10:00:00Z",
        ),
    )
    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_f32(_embed("distant vector"))),
    )


def test_vector_store_creates_trigram_fts_table(tmp_path):
    store = VectorStore(tmp_path / "trigram.db")
    try:
        sql = store.conn.cursor().execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts_trigram'").fetchone()
        assert sql is not None
        assert "tokenize='trigram'" in sql[0]
    finally:
        store.close()


def test_fts_update_triggers_delete_by_mapped_rowid(tmp_path):
    store = VectorStore(tmp_path / "trigram-rowid.db")
    try:
        _insert_chunk(store, chunk_id="chunk-rowid", content="stalker-golem queue note")

        trigger_sql = {
            row[0]: row[1]
            for row in store.conn.cursor().execute(
                """
                SELECT name, sql FROM sqlite_master
                WHERE type = 'trigger' AND name IN ('chunks_fts_update', 'chunks_fts_trigram_update')
                """
            )
        }
        assert "DELETE FROM chunks_fts WHERE chunk_id = old.id" not in trigger_sql["chunks_fts_update"]
        assert "DELETE FROM chunks_fts_trigram WHERE chunk_id = old.id" not in trigger_sql["chunks_fts_trigram_update"]
        assert "chunk_fts_rowids" in trigger_sql["chunks_fts_update"]
        assert "chunk_fts_rowids" in trigger_sql["chunks_fts_trigram_update"]

        store.update_enrichment("chunk-rowid", summary="fresh rowid summary", tags=["rowid"])

        cursor = store.conn.cursor()
        assert cursor.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'chunk-rowid'").fetchone()[0] == 1
        assert (
            cursor.execute("SELECT COUNT(*) FROM chunks_fts_trigram WHERE chunk_id = 'chunk-rowid'").fetchone()[0] == 1
        )
        assert cursor.execute("SELECT summary FROM chunks_fts WHERE chunk_id = 'chunk-rowid'").fetchone()[0] == (
            "fresh rowid summary"
        )
    finally:
        store.close()


def test_hybrid_search_uses_trigram_fts_for_identifier_substrings(tmp_path):
    store = VectorStore(tmp_path / "trigram-search.db")
    try:
        _insert_chunk(store, chunk_id="chunk-trigram-hit", content="stalker-golem queue note")
        store.build_binary_index()
        cursor = store.conn.cursor()
        cursor.execute("DELETE FROM chunk_vectors")
        cursor.execute("DELETE FROM chunk_vectors_binary")

        results = store.hybrid_search(
            query_embedding=_embed("nothing close"),
            query_text="alker-go",
            n_results=5,
        )

        assert "chunk-trigram-hit" in results["ids"][0]
    finally:
        store.close()


def test_vector_store_repairs_partial_trigram_backfill_only_on_explicit_repair(tmp_path):
    db_path = tmp_path / "trigram-repair.db"
    store = VectorStore(db_path)
    try:
        _insert_chunk(store, chunk_id="chunk-a", content="stalker-golem queue note")
        _insert_chunk(store, chunk_id="chunk-b", content="brainbar queue fallback note")
        store.conn.cursor().execute("DELETE FROM chunks_fts_trigram WHERE chunk_id = ?", ("chunk-a",))
    finally:
        store.close()

    opened = VectorStore(db_path)
    try:
        trigram_count = opened.conn.cursor().execute("SELECT COUNT(*) FROM chunks_fts_trigram").fetchone()[0]
        chunk_count = opened.conn.cursor().execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert trigram_count == 1
        assert chunk_count == 2

        repaired = opened.repair_fts()
        trigram_count = opened.conn.cursor().execute("SELECT COUNT(*) FROM chunks_fts_trigram").fetchone()[0]
        assert repaired["chunks_fts_trigram"] == trigram_count == chunk_count == 2
    finally:
        opened.close()
