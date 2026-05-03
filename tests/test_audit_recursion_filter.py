import json

from brainlayer._helpers import serialize_f32
from brainlayer.engine import recall, think
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
    cursor.executemany(
        "INSERT OR IGNORE INTO chunk_tags (chunk_id, tag) VALUES (?, ?)",
        [(chunk_id, tag) for tag in tags],
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
        store.build_binary_index()

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
        store.build_binary_index()

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="mirror07 normal operational memory",
            n_results=3,
        )

        assert "ordinary-mirror07-memory" in results["ids"][0]
    finally:
        store.close()


def test_readonly_legacy_db_without_chunk_tags_uses_json_tag_fallback(tmp_path):
    db_path = tmp_path / "legacy-readonly-audit-filter.db"
    store = VectorStore(db_path)
    try:
        query_embedding = [0.04] * 1024
        _insert_chunk(
            store,
            "legacy-audit-source",
            "legacy readonly audit memory should be filtered",
            ["r02", "audit"],
            query_embedding,
        )
        _insert_chunk(
            store,
            "legacy-ordinary-memory",
            "legacy readonly ordinary memory should be searchable",
            ["brainbar", "reliability"],
            [0.05] * 1024,
        )
        store.build_binary_index()
        cursor = store.conn.cursor()
        for trigger in (
            "chunk_tags_insert",
            "chunk_tags_update",
            "chunk_tags_update_clear",
            "chunk_tags_delete",
        ):
            cursor.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        cursor.execute("DROP TABLE chunk_tags")
    finally:
        store.close()

    db_path.chmod(0o444)
    readonly_store = VectorStore(db_path)
    try:
        assert readonly_store._chunk_tags_available is False
        results = readonly_store.hybrid_search(
            query_embedding=query_embedding,
            query_text="legacy readonly memory",
            n_results=3,
        )
        ids = results["ids"][0]
        assert "legacy-audit-source" not in ids
        assert "legacy-ordinary-memory" in ids
    finally:
        readonly_store.close()
        db_path.chmod(0o644)


def test_hybrid_search_overfetches_when_audit_chunks_dominate_knn(tmp_path):
    store = VectorStore(tmp_path / "audit-filter-overfetch.db")
    try:
        query_embedding = [0.06] * 1024
        for index in range(60):
            _insert_chunk(
                store,
                f"audit-neighbor-{index}",
                f"audit recursion neighbor {index}",
                ["r02", "audit"],
                query_embedding,
            )
        _insert_chunk(
            store,
            "ordinary-after-audit-neighbors",
            "ordinary BrainBar restart decision should survive audit-heavy nearest neighbors",
            ["brainbar", "reliability"],
            [0.061] * 1024,
        )
        store.build_binary_index()

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="ordinary BrainBar restart decision",
            n_results=3,
        )

        assert "ordinary-after-audit-neighbors" in results["ids"][0]
        assert all(not chunk_id.startswith("audit-neighbor-") for chunk_id in results["ids"][0])
    finally:
        store.close()


def test_exact_r0x_tag_is_filtered_as_audit_shorthand(tmp_path):
    store = VectorStore(tmp_path / "audit-filter-r0x.db")
    try:
        query_embedding = [0.07] * 1024
        _insert_chunk(
            store,
            "audit-r0x-source",
            "r0x audit shorthand memory should be filtered",
            ["r0x"],
            query_embedding,
        )
        _insert_chunk(
            store,
            "ordinary-r0x-control",
            "ordinary control memory should remain searchable",
            ["brainbar"],
            [0.071] * 1024,
        )
        store.build_binary_index()

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="ordinary control memory",
            n_results=3,
        )

        assert "audit-r0x-source" not in results["ids"][0]
        assert "ordinary-r0x-control" in results["ids"][0]
    finally:
        store.close()


def test_engine_think_and_recall_forward_include_audit():
    class MockStore:
        def __init__(self):
            self.calls = []

        def hybrid_search(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "documents": [["ordinary memory"]],
                "metadatas": [[{"intent": "decision", "project": "brainlayer"}]],
            }

        def get_file_timeline(self, *_args, **_kwargs):
            return []

    mock_store = MockStore()
    think(
        "think about audit history",
        store=mock_store,
        embed_fn=lambda _text: [0.1] * 1024,
        include_audit=True,
    )
    recall(
        store=mock_store,
        embed_fn=lambda _text: [0.1] * 1024,
        topic="audit history",
        include_audit=True,
    )

    assert mock_store.calls[0]["include_audit"] is True
    assert mock_store.calls[1]["include_audit"] is True
