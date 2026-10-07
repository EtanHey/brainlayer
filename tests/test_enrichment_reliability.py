"""Tests for enrichment reliability — retry, circuit breaker, timeout config."""

from brainlayer.pipeline import enrichment_results as enrichment


def test_parse_enrichment_uses_hybrid_tag_taxonomy_by_default(monkeypatch):
    monkeypatch.delenv("BRAINLAYER_ENRICHMENT_TAG_MODE", raising=False)
    parsed = enrichment.parse_enrichment(
        """
        {
          "summary": "BrainLayer Track B normalizes enrichment tags before backlog re-enrichment.",
          "tags": [
            "Project/BrainLayer",
            "tech/debug/investigation",
            "React.js",
            "reactjs",
            "tech/debug/resolution",
            "one-off-singleton-from-model",
            "PM/Decision"
          ],
          "importance": 8
        }
        """
    )

    assert parsed is not None
    assert parsed["tags"] == [
        "project/brainlayer",
        "tech/debug/investigation",
        "react",
        "tech/debug/resolution",
        "one-off-singleton-from-model",
        "pm/decision",
    ]


def test_parse_enrichment_can_roll_back_to_taxonomy_whitelist(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_TAG_MODE", "taxonomy")
    parsed = enrichment.parse_enrichment(
        """
        {
          "summary": "BrainLayer Track B can still A/B the old taxonomy whitelist mode.",
          "tags": ["Project/BrainLayer", "React.js", "tech/debug/resolution", "PM/Decision"],
          "importance": 8
        }
        """
    )

    assert parsed is not None
    assert parsed["tags"] == ["project/brainlayer", "pm/decision"]


def test_parse_enrichment_version_stamps_outputs(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_RUN_ID", "test-run")
    parsed = enrichment.parse_enrichment(
        """
        {
          "summary": "Hybrid taxonomy outputs carry provenance metadata for eval reproducibility.",
          "tags": ["React.js"],
          "importance": 6
        }
        """
    )

    assert parsed is not None
    metadata = parsed["enrichment_metadata"]
    assert metadata["prompt_version"]
    assert metadata["tag_mode"] == "hybrid"
    assert metadata["taxonomy_git_sha"]
    assert metadata["taxonomy_content_sha"]
    assert metadata["run_id"] == "test-run"


def test_build_prompt_switches_tag_rules_with_env(monkeypatch):
    from brainlayer.pipeline import enrichment

    chunk = {
        "content": "BrainLayer React.js tag normalization decision.",
        "project": "brainlayer",
        "content_type": "assistant_text",
    }

    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_TAG_MODE", "hybrid")
    hybrid_prompt = enrichment.build_prompt(chunk)
    assert "HYBRID TAG MODE" in hybrid_prompt
    assert "React.js/reactjs/React -> react" in hybrid_prompt

    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_TAG_MODE", "taxonomy")
    taxonomy_prompt = enrichment.build_prompt(chunk)
    assert "TAXONOMY WHITELIST MODE" in taxonomy_prompt
    assert "Do NOT invent free-form singleton tags" in taxonomy_prompt


def test_parse_enrichment_tag_whitelist_is_forward_only_for_existing_rows(tmp_path, monkeypatch):
    from brainlayer.vector_store import VectorStore

    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_TAG_MODE", "taxonomy")
    store = VectorStore(tmp_path / "forward-tags.db")
    try:
        cursor = store.conn.cursor()
        cursor.execute(
            """
            INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type,
                value_type, char_count, source, tags
            ) VALUES (?, ?, '{}', 'test', 'brainlayer', 'note', 'high', 80, 'manual', ?)
            """,
            (
                "legacy-tags",
                "Existing stored rows should not be rewritten by parsing a new enrichment response.",
                '["legacy-freeform", "project/brainlayer"]',
            ),
        )

        parsed = enrichment.parse_enrichment(
            """
            {
              "summary": "New enrichment output must be taxonomy-only.",
              "tags": ["project/brainlayer", "new-model-singleton", "tech/debug/investigation"],
              "importance": 7
            }
            """
        )

        stored_tags = cursor.execute("SELECT tags FROM chunks WHERE id = 'legacy-tags'").fetchone()[0]
    finally:
        store.close()

    assert parsed is not None
    assert parsed["tags"] == ["project/brainlayer", "tech/debug/investigation"]
    assert stored_tags == '["legacy-freeform", "project/brainlayer"]'


def test_tombstone_singleton_tags_preserves_taxonomy_tags(tmp_path):
    from brainlayer.tag_normalization import tombstone_singleton_tags
    from brainlayer.vector_store import VectorStore

    store = VectorStore(tmp_path / "tags.db")
    try:
        cursor = store.conn.cursor()
        rows = [
            ("c1", '["project/brainlayer", "singleton-sprawl"]'),
            ("c2", '["shared-sprawl"]'),
            ("c3", '["shared-sprawl"]'),
        ]
        for chunk_id, tags_json in rows:
            cursor.execute(
                """
                INSERT INTO chunks (
                    id, content, metadata, source_file, project, content_type,
                    value_type, char_count, source, tags
                ) VALUES (?, ?, '{}', 'test', 'brainlayer', 'note', 'high', 80, 'manual', ?)
                """,
                (chunk_id, f"content for {chunk_id} with enough text to store", tags_json),
            )

        result = tombstone_singleton_tags(store.conn)

        assert result.tombstoned == 1
        assert result.updated_chunks == 1
        assert cursor.execute("SELECT reason FROM tag_tombstones WHERE tag = 'singleton-sprawl'").fetchone()[0] == (
            "singleton-non-taxonomy"
        )
        assert cursor.execute("SELECT 1 FROM tag_tombstones WHERE tag = 'project/brainlayer'").fetchone() is None
        assert cursor.execute("SELECT tags FROM chunks WHERE id = 'c1'").fetchone()[0] == '["project/brainlayer"]'
        assert sorted(row[0] for row in cursor.execute("SELECT tag FROM chunk_tags")) == [
            "project/brainlayer",
            "shared-sprawl",
            "shared-sprawl",
        ]
    finally:
        store.close()


def test_store_memory_uses_lowercase_value_type():
    import tempfile
    from pathlib import Path

    from brainlayer.store import store_memory
    from brainlayer.vector_store import VectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store = VectorStore(Path(tmpdir) / "value-type.db")
        try:
            result = store_memory(
                store=vector_store,
                embed_fn=None,
                content="Value type casing should match the lower-case ContentValue enum.",
                memory_type="note",
                project="brainlayer",
            )
            assert (
                vector_store.conn.cursor()
                .execute(
                    "SELECT value_type FROM chunks WHERE id = ?",
                    (result["id"],),
                )
                .fetchone()[0]
                == "high"
            )
        finally:
            vector_store.close()
