import json

import apsw
import pytest

from brainlayer._helpers import serialize_f32
from brainlayer.engine import recall, think
from brainlayer.mcp.search_handler import _brain_search, _kg_facts_sql
from brainlayer.search_repo import _audit_recursion_exclusion_sql
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


def _insert_context_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    content: str,
    tags: list[str],
    position: int,
) -> None:
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, tags, conversation_id, position
        ) VALUES (?, ?, '{}', 'audit-context.jsonl', 'brainlayer',
                  'assistant_text', ?, 'claude_code', ?, 'audit-context-session', ?)""",
        (chunk_id, content, len(content), json.dumps(tags), position),
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


def test_hybrid_search_does_not_exclude_normal_audit_word_tags(tmp_path):
    store = VectorStore(tmp_path / "audit-filter-normal-audit-tag.db")
    try:
        query_embedding = [0.031] * 1024
        _insert_chunk(
            store,
            "ordinary-security-audit-memory",
            "security audit finding with real operational content should remain searchable",
            ["security-audit", "reliability"],
            query_embedding,
        )
        store.build_binary_index()

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="security audit operational content",
            n_results=3,
        )

        assert "ordinary-security-audit-memory" in results["ids"][0]
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


def test_audit_recursion_count_uses_cached_value_on_busy_retry_exhaustion(tmp_path, monkeypatch):
    store = VectorStore(tmp_path / "audit-filter-count-cache.db")
    try:
        query_embedding = [0.065] * 1024
        _insert_chunk(
            store,
            "audit-count-source",
            "audit count source",
            ["audit"],
            query_embedding,
        )

        assert store._audit_recursion_count() == 1

        attempts = 0

        def busy_read_cursor():
            nonlocal attempts
            attempts += 1
            raise apsw.BusyError("database is locked")

        monkeypatch.setattr(store, "_checkpoint_cache_data_version", lambda: 999)
        monkeypatch.setattr(store, "_read_cursor", busy_read_cursor)

        assert store._audit_recursion_count() == 1
        assert attempts == 3
    finally:
        store.close()


def test_audit_overfetch_scales_with_filtered_row_count(tmp_path, monkeypatch):
    store = VectorStore(tmp_path / "audit-filter-overfetch-size.db")
    try:
        monkeypatch.setattr(store, "_audit_recursion_count", lambda: 1500)

        assert store._effective_knn_k(3, needs_overfetch=False, include_checkpoints=True, include_audit=False) == 1503
    finally:
        store.close()


def test_audit_count_cache_invalidates_after_same_connection_upsert(tmp_path):
    store = VectorStore(tmp_path / "audit-filter-cache-invalidation.db")
    try:
        query_embedding = [0.066] * 1024

        assert store._audit_recursion_count() == 0
        assert store._audit_recursion_count_cache == 0

        audit_chunks = [
            {
                "id": f"audit-cache-neighbor-{index}",
                "content": f"audit cache neighbor {index}",
                "metadata": {},
                "source_file": "audit-cache.jsonl",
                "project": "brainlayer",
                "content_type": "assistant_text",
                "source": "claude_code",
                "char_count": len(f"audit cache neighbor {index}"),
                "tags": ["audit"],
            }
            for index in range(30)
        ]
        normal_content = "ordinary same connection audit cache invalidation target"
        normal_chunk = {
            "id": "ordinary-after-same-connection-audit-cache",
            "content": normal_content,
            "metadata": {},
            "source_file": "audit-cache.jsonl",
            "project": "brainlayer",
            "content_type": "assistant_text",
            "source": "claude_code",
            "char_count": len(normal_content),
            "tags": ["brainbar"],
        }
        store.upsert_chunks(audit_chunks + [normal_chunk], [query_embedding] * 30 + [[0.067] * 1024])

        assert store._audit_recursion_count_cache is None

        results = store.search(query_embedding=query_embedding, n_results=3)
        assert "ordinary-after-same-connection-audit-cache" in results["ids"][0]
        assert all(not chunk_id.startswith("audit-cache-neighbor-") for chunk_id in results["ids"][0])
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


def test_recursive_mcp_output_content_is_filtered_even_without_audit_tags(tmp_path):
    store = VectorStore(tmp_path / "recursive-mcp-output-filter.db")
    try:
        query_embedding = [0.08] * 1024
        _insert_chunk(
            store,
            "recursive-jsonrpc-output",
            'MCP BrainLayer Memory: Invalid JSON-RPC message: {"jsonrpc":"2.0","id":24}',
            ["correction:factual", "auto-detected"],
            query_embedding,
        )
        _insert_chunk(
            store,
            "recursive-brain-search-box",
            '┌─ brain_search: "BrainLayer audit recursion" ─ 1 result\n│ recursive output',
            ["auto-detected"],
            query_embedding,
        )
        _insert_chunk(
            store,
            "recursive-entity-search-box",
            "┌─ Entity search: BrainLayer\n│ recursive entity output",
            ["auto-detected"],
            query_embedding,
        )
        _insert_chunk(
            store,
            "ordinary-mcp-memory",
            "BrainLayer MCP timeout investigation produced a real operational fix",
            ["brainlayer", "mcp"],
            [0.081] * 1024,
        )
        store.build_binary_index()

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="BrainLayer MCP",
            n_results=5,
        )
        ids = results["ids"][0]

        assert "ordinary-mcp-memory" in ids
        assert "recursive-jsonrpc-output" not in ids
        assert "recursive-brain-search-box" not in ids
        assert "recursive-entity-search-box" not in ids

        audit_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="BrainLayer MCP",
            n_results=5,
            include_audit=True,
        )

        assert "recursive-jsonrpc-output" in audit_results["ids"][0]
        assert "recursive-brain-search-box" in audit_results["ids"][0]
        assert "recursive-entity-search-box" in audit_results["ids"][0]
    finally:
        store.close()


def test_formatted_jsonrpc_content_is_filtered_by_sql_paths(tmp_path):
    store = VectorStore(tmp_path / "recursive-formatted-jsonrpc-filter.db")
    try:
        query_embedding = [0.09] * 1024
        _insert_chunk(
            store,
            "recursive-formatted-jsonrpc-output",
            'MCP output payload: {"jsonrpc" :\n\t"2.0", "id": 24}',
            ["auto-detected"],
            query_embedding,
        )
        _insert_chunk(
            store,
            "ordinary-formatted-jsonrpc-control",
            "BrainLayer MCP guard should keep ordinary results visible",
            ["brainlayer", "mcp"],
            [0.091] * 1024,
        )
        store.build_binary_index()

        vector_results = store.search(query_embedding=query_embedding, n_results=5)
        hybrid_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="BrainLayer MCP guard",
            n_results=5,
        )

        assert "recursive-formatted-jsonrpc-output" not in vector_results["ids"][0]
        assert "ordinary-formatted-jsonrpc-control" in vector_results["ids"][0]
        assert "recursive-formatted-jsonrpc-output" not in hybrid_results["ids"][0]
        assert "ordinary-formatted-jsonrpc-control" in hybrid_results["ids"][0]
    finally:
        store.close()


def test_brain_search_box_with_leading_non_space_whitespace_is_filtered_by_sql_paths(tmp_path):
    store = VectorStore(tmp_path / "recursive-leading-whitespace-filter.db")
    try:
        query_embedding = [0.092] * 1024
        _insert_chunk(
            store,
            "recursive-leading-whitespace-box",
            '\n\t┌─ brain_search: "BrainLayer audit recursion"\n│ recursive output',
            ["auto-detected"],
            query_embedding,
        )
        _insert_chunk(
            store,
            "ordinary-leading-whitespace-control",
            "BrainLayer MCP guard should still return ordinary memories",
            ["brainlayer", "mcp"],
            [0.093] * 1024,
        )
        store.build_binary_index()

        vector_results = store.search(query_embedding=query_embedding, n_results=5)
        hybrid_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="BrainLayer MCP guard",
            n_results=5,
        )

        assert "recursive-leading-whitespace-box" not in vector_results["ids"][0]
        assert "ordinary-leading-whitespace-control" in vector_results["ids"][0]
        assert "recursive-leading-whitespace-box" not in hybrid_results["ids"][0]
        assert "ordinary-leading-whitespace-control" in hybrid_results["ids"][0]
    finally:
        store.close()


def test_audit_recursion_sql_accepts_explicit_content_expression():
    clause = _audit_recursion_exclusion_sql(
        "c.id",
        "audit_tags.tags",
        content_expr="c.content",
        use_chunk_tags=False,
    )

    assert "c.content" in clause
    assert "audit_tags.content" not in clause


def test_kg_facts_exclude_audit_sourced_relations_by_default(tmp_path):
    store = VectorStore(tmp_path / "audit-filter-kg-facts.db")
    try:
        query_embedding = [0.094] * 1024
        _insert_chunk(
            store,
            "audit-fact-chunk",
            "Etan stores recursive audit output",
            ["audit"],
            query_embedding,
        )
        _insert_chunk(
            store,
            "normal-fact-chunk",
            "Etan stores durable memory",
            ["brainlayer"],
            [0.095] * 1024,
        )
        store.upsert_entity("person-etan", "person", "Etan")
        store.upsert_entity("project-audit", "project", "Audit Project")
        store.upsert_entity("project-normal", "project", "Normal Project")
        store.add_relation(
            "rel-audit",
            "person-etan",
            "project-audit",
            "mentions",
            fact="audit-sourced fact",
            source_chunk_id="audit-fact-chunk",
        )
        store.add_relation(
            "rel-normal",
            "person-etan",
            "project-normal",
            "maintains",
            fact="normal fact",
            source_chunk_id="normal-fact-chunk",
        )

        sql_default_facts = _kg_facts_sql(store, "Etan")
        sql_audit_facts = _kg_facts_sql(store, "Etan", include_audit=True)
        hybrid_default = store.kg_hybrid_search(
            query_embedding=query_embedding,
            query_text="Etan",
            n_results=10,
            entity_name="Etan",
        )
        hybrid_with_audit = store.kg_hybrid_search(
            query_embedding=query_embedding,
            query_text="Etan",
            n_results=10,
            entity_name="Etan",
            include_audit=True,
        )

        assert {fact["target"] for fact in sql_default_facts} == {"Normal Project"}
        assert {fact["target"] for fact in sql_audit_facts} == {"Audit Project", "Normal Project"}
        assert {fact["target_entity"]["name"] for fact in hybrid_default["facts"]} == {"Normal Project"}
        assert {fact["target_entity"]["name"] for fact in hybrid_with_audit["facts"]} == {
            "Audit Project",
            "Normal Project",
        }
    finally:
        store.close()


@pytest.mark.asyncio
async def test_chunk_context_filters_audit_neighbors_by_default(tmp_path, monkeypatch):
    store = VectorStore(tmp_path / "audit-context-filter.db")
    try:
        recursive_content = 'MCP BrainLayer Memory: Invalid JSON-RPC message: {"jsonrpc":"2.0"}'
        _insert_context_chunk(
            store,
            chunk_id="audit-context-neighbor",
            content=recursive_content,
            tags=["audit"],
            position=1,
        )
        _insert_context_chunk(
            store,
            chunk_id="normal-context-target",
            content="Normal context target",
            tags=["brainlayer"],
            position=2,
        )
        _insert_context_chunk(
            store,
            chunk_id="normal-context-neighbor",
            content="Normal context neighbor",
            tags=["brainlayer"],
            position=3,
        )
        monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)

        default_content = await _brain_search(query="ignored", chunk_id="normal-context-target", before=1, after=1)
        audit_content = await _brain_search(
            query="ignored",
            chunk_id="normal-context-target",
            before=1,
            after=1,
            include_audit=True,
        )

        assert recursive_content not in default_content[0].text
        assert "Normal context target" in default_content[0].text
        assert "Normal context neighbor" in default_content[0].text
        assert recursive_content in audit_content[0].text
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
