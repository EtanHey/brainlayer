"""Phase 3: Brain Digest + Brain Entity tests."""

from typing import List

from brainlayer.vector_store import VectorStore


def _dummy_embed(text):  # noqa: ARG001
    """Dummy embedding function for tests."""
    return [0.1] * 1024


def _insert_chunks(
    store: VectorStore, ids: List[str], documents: List[str], metadatas: List[dict], embeddings: List[list]
) -> None:
    """Helper to insert test chunks using upsert_chunks API."""
    chunks = []
    for cid, doc, meta in zip(ids, documents, metadatas):
        chunks.append(
            {
                "id": cid,
                "content": doc,
                "metadata": meta,
                "source_file": meta.get("source_file", "test.jsonl"),
                "project": meta.get("project"),
                "content_type": meta.get("content_type", "user_message"),
                "char_count": len(doc),
                "source": meta.get("source", "claude_code"),
            }
        )
    store.upsert_chunks(chunks, embeddings)


# --- Task 1: Schema — user_verified column ---


def test_user_verified_column_on_kg_entities(tmp_path):
    """kg_entities has user_verified column."""
    store = VectorStore(tmp_path / "test.db")
    cursor = store.conn.cursor()
    cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entities)")}
    assert "user_verified" in cols


def test_user_verified_column_on_kg_relations(tmp_path):
    """kg_relations has user_verified column."""
    store = VectorStore(tmp_path / "test.db")
    cursor = store.conn.cursor()
    cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_relations)")}
    assert "user_verified" in cols


def test_user_verified_defaults_to_false(tmp_path):
    """user_verified defaults to 0 (false) on new entities."""
    store = VectorStore(tmp_path / "test.db")
    eid = store.upsert_entity("test-ent-1", "person", "Test Person")
    cursor = store.conn.cursor()
    row = list(cursor.execute("SELECT user_verified FROM kg_entities WHERE id = ?", [eid]))[0]
    assert row[0] == 0


# --- Task 2: Digest pipeline ---


def test_digest_content_returns_structured_result(tmp_path):
    """digest_content returns DigestResult with entities and sentiment."""
    from brainlayer.pipeline.digest import digest_content

    store = VectorStore(tmp_path / "test.db")
    dummy_embed = _dummy_embed

    result = digest_content(
        content="Etan met with Dor Zohar to discuss the Domica project. They decided to use React Native.",
        store=store,
        embed_fn=dummy_embed,
        title="Meeting notes",
        project="domica",
        participants=["Etan Heyman", "Dor Zohar"],
    )

    assert result["digest_id"] is not None
    assert result["summary"] != ""
    assert isinstance(result["entities"], list)
    assert isinstance(result["relations"], list)
    assert isinstance(result["sentiment"], dict)
    assert "label" in result["sentiment"]
    assert isinstance(result["stats"], dict)
    assert result["stats"]["entities_found"] >= 0


def test_digest_content_creates_chunk(tmp_path):
    """digest_content stores a new chunk in the DB."""
    from brainlayer.pipeline.digest import digest_content

    store = VectorStore(tmp_path / "test.db")
    dummy_embed = _dummy_embed

    result = digest_content(
        content="Some meeting notes about testing.",
        store=store,
        embed_fn=dummy_embed,
    )

    # Verify chunk exists
    cursor = store.conn.cursor()
    row = list(cursor.execute("SELECT id, source, content_type FROM chunks WHERE id = ?", [result["digest_id"]]))
    assert len(row) == 1
    assert row[0][1] == "digest"
    assert row[0][2] == "user_message"


def test_digest_content_extracts_entities(tmp_path):
    """digest_content extracts entities from seed list."""
    from brainlayer.pipeline.digest import digest_content

    store = VectorStore(tmp_path / "test.db")
    dummy_embed = _dummy_embed

    result = digest_content(
        content="Etan Heyman discussed brainlayer architecture with Dor Zohar at Cantaloupe AI.",
        store=store,
        embed_fn=dummy_embed,
        participants=["Etan Heyman", "Dor Zohar"],
    )

    entity_names = [e["name"] for e in result["entities"]]
    # At least seed entities should be found
    assert any("Etan" in n for n in entity_names) or any("Dor" in n for n in entity_names)


def test_digest_content_applies_sentiment(tmp_path):
    """digest_content includes sentiment analysis."""
    from brainlayer.pipeline.digest import digest_content

    store = VectorStore(tmp_path / "test.db")
    dummy_embed = _dummy_embed

    result = digest_content(
        content="This is amazing! Everything works perfectly!",
        store=store,
        embed_fn=dummy_embed,
    )

    assert result["sentiment"]["label"] == "positive"
    assert result["sentiment"]["score"] > 0


def test_digest_content_confidence_tiers(tmp_path):
    """Entities are categorized by confidence tier in stats."""
    from brainlayer.pipeline.digest import digest_content

    store = VectorStore(tmp_path / "test.db")
    dummy_embed = _dummy_embed

    result = digest_content(
        content="Etan Heyman works at Cantaloupe AI on the brainlayer project.",
        store=store,
        embed_fn=dummy_embed,
        participants=["Etan Heyman"],
    )

    stats = result["stats"]
    assert "high_confidence" in stats
    assert "needs_review" in stats


def test_digest_content_empty_raises(tmp_path):
    """Empty content raises ValueError."""
    from brainlayer.pipeline.digest import digest_content

    store = VectorStore(tmp_path / "test.db")
    dummy_embed = _dummy_embed

    import pytest

    with pytest.raises(ValueError, match="content must be non-empty"):
        digest_content(content="", store=store, embed_fn=dummy_embed)


def test_digest_extracts_action_items(tmp_path):
    """digest_content extracts action items from content."""
    from brainlayer.pipeline.digest import digest_content

    store = VectorStore(tmp_path / "test.db")
    dummy_embed = _dummy_embed

    result = digest_content(
        content="Action items: 1. Send the proposal to Avi by Friday. 2. Schedule a follow-up meeting with Dor.",
        store=store,
        embed_fn=dummy_embed,
    )

    assert isinstance(result["action_items"], list)
    assert isinstance(result["decisions"], list)
    assert isinstance(result["questions"], list)


def test_digest_content_applies_faceted_enrichment_and_marks_chunk_enriched(tmp_path):
    """digest_content writes faceted Gemini tags into the chunk enrichment fields."""
    from brainlayer.pipeline.digest import digest_content

    store = VectorStore(tmp_path / "test.db")

    def fake_faceted_enrich(*, content, project, title, participants):  # noqa: ARG001
        return {
            "topics": ["brainlayer-search-quality", "entity-memory-scope"],
            "activity": "act:designing",
            "domains": ["dom:mcp", "dom:python"],
            "confidence": 0.91,
            "provider": "gemini",
            "model": "gemini-2.5-flash-lite",
        }

    result = digest_content(
        content="We decided BrainLayer digest should add faceted enrichment through Gemini.",
        store=store,
        embed_fn=_dummy_embed,
        project="brainlayer",
        faceted_enrich_fn=fake_faceted_enrich,
    )

    assert result["tags"] == [
        "brainlayer-search-quality",
        "entity-memory-scope",
        "act:designing",
        "dom:mcp",
        "dom:python",
    ]
    assert result["enrichment"]["status"] == "enriched"
    assert result["enrichment"]["confidence"] == 0.91

    cursor = store.conn.cursor()
    row = list(
        cursor.execute(
            "SELECT tags, intent, summary, enriched_at FROM chunks WHERE id = ?",
            [result["digest_id"]],
        )
    )[0]

    assert row[0] is not None
    assert "act:designing" in row[0]
    assert row[1] == "designing"
    assert row[2] == result["summary"]
    assert row[3] is not None


# --- Task 3: brain_digest MCP tool schema ---


def test_brain_digest_tool_exists():
    """brain_digest tool is registered in MCP server."""
    import asyncio

    from brainlayer.mcp import list_tools

    tools = asyncio.run(list_tools())
    tool_names = [t.name for t in tools]
    assert "brain_digest" in tool_names


def test_brain_digest_schema_has_required_fields():
    """brain_digest tool exposes digest fields and mode-based enrich controls."""
    import asyncio

    from brainlayer.mcp import list_tools

    tools = asyncio.run(list_tools())
    digest = next(t for t in tools if t.name == "brain_digest")
    props = digest.inputSchema.get("properties", {})
    assert "content" in props
    assert "title" in props
    assert "participants" in props
    assert "mode" in props
    assert "limit" in props


def test_brain_digest_description_teaches_routing():
    """brain_digest description explains when to use it and how it differs from brain_store."""
    import asyncio

    from brainlayer.mcp import list_tools

    tools = asyncio.run(list_tools())
    digest = next(t for t in tools if t.name == "brain_digest")
    desc = digest.description.lower()

    assert "brain_store" in desc
    assert "large" in desc  # "large text content"
    assert "entities" in desc
    assert "knowledge graph" in desc
    assert "digest" in desc
    assert "connect" in desc
    assert "enrich" in desc


# --- Task 4: brain_entity MCP tool ---


def test_brain_entity_tool_exists():
    """brain_entity tool is registered in MCP server."""
    import asyncio

    from brainlayer.mcp import list_tools

    tools = asyncio.run(list_tools())
    tool_names = [t.name for t in tools]
    assert "brain_entity" in tool_names


def test_brain_entity_schema():
    """brain_entity keeps query in properties even when optional."""
    import asyncio

    from brainlayer.mcp import list_tools

    tools = asyncio.run(list_tools())
    entity_tool = next(t for t in tools if t.name == "brain_entity")
    props = entity_tool.inputSchema.get("properties", {})
    assert "query" in props


def test_entity_lookup_returns_structured_data(tmp_path):
    """_entity_lookup returns entity with relations and evidence."""
    from brainlayer.pipeline.digest import entity_lookup

    store = VectorStore(tmp_path / "test.db")
    dummy_embed = _dummy_embed

    # Create an entity with a chunk
    eid = store.upsert_entity("person-etan", "person", "Etan Heyman", embedding=dummy_embed("Etan Heyman"))
    _insert_chunks(
        store,
        ["c1"],
        ["Etan discussed brainlayer"],
        [{"source_file": "t.jsonl", "project": "test"}],
        [dummy_embed("test")],
    )
    store.link_entity_chunk(eid, "c1", relevance=0.9, context="discussed brainlayer")

    result = entity_lookup("Etan", store, dummy_embed)
    assert result is not None
    assert result["name"] == "Etan Heyman"
    assert result["entity_type"] == "person"
    assert isinstance(result["relations"], list)
    assert isinstance(result["evidence"], list)
    assert len(result["evidence"]) >= 1


def test_entity_lookup_not_found(tmp_path):
    """entity_lookup returns None for unknown entities."""
    from brainlayer.pipeline.digest import entity_lookup

    store = VectorStore(tmp_path / "test.db")
    dummy_embed = _dummy_embed

    result = entity_lookup("Nonexistent Person", store, dummy_embed)
    assert result is None


def test_entity_lookup_merges_case_duplicate_entities(tmp_path):
    """Lookup should collapse case-only dupes and return the entity with linked evidence."""
    from brainlayer.pipeline.digest import entity_lookup

    store = VectorStore(tmp_path / "test.db")
    dummy_embed = _dummy_embed

    rich_id = store.upsert_entity("project-rich", "project", "brainlayer", embedding=dummy_embed("brainlayer"))
    sparse_id = "project-sparse"
    store.conn.cursor().execute(
        """
        INSERT INTO kg_entities (id, entity_type, name, metadata, canonical_name, created_at, updated_at)
        VALUES (?, 'project', 'BrainLayer', '{}', NULL, '2026-04-13T00:00:00Z', '2026-04-13T00:00:00Z')
        """,
        (sparse_id,),
    )
    sqlite_id = store.upsert_entity("tech-sqlite", "technology", "SQLite", embedding=dummy_embed("sqlite"))
    store.add_relation("rel-uses", rich_id, sqlite_id, "uses")
    _insert_chunks(
        store,
        ["c-rich"],
        ["brainlayer uses SQLite for local-first search"],
        [{"source_file": "t.jsonl", "project": "test"}],
        [dummy_embed("brainlayer uses sqlite")],
    )
    store.link_entity_chunk(rich_id, "c-rich", relevance=0.9, context="architecture note")

    result = entity_lookup("BrainLayer", store, dummy_embed, entity_type="project")

    assert result is not None
    assert len(result["evidence"]) == 1
    assert any(relation["target_name"] == "SQLite" for relation in result["relations"])
    remaining = list(
        store._read_cursor().execute(
            "SELECT id FROM kg_entities WHERE entity_type = 'project' AND LOWER(name) = 'brainlayer'"
        )
    )
    assert remaining == [(rich_id,)]


def test_entity_lookup_adds_flowbar_voicebar_rename_relation(tmp_path):
    """Lookup should seed the FlowBar -> VoiceBar rename edge when both entities exist."""
    from brainlayer.pipeline.digest import entity_lookup

    store = VectorStore(tmp_path / "test.db")
    dummy_embed = _dummy_embed

    flowbar_id = store.upsert_entity("project-flowbar", "project", "FlowBar", embedding=dummy_embed("FlowBar"))
    voicebar_id = store.upsert_entity("project-voicebar", "project", "VoiceBar", embedding=dummy_embed("VoiceBar"))

    result = entity_lookup("VoiceBar", store, dummy_embed, entity_type="project")

    assert result is not None
    row = (
        store._read_cursor()
        .execute(
            """
        SELECT relation_type FROM kg_relations
        WHERE source_id = ? AND target_id = ? AND relation_type = 'RENAMED_FROM'
        """,
            (flowbar_id, voicebar_id),
        )
        .fetchone()
    )
    assert row == ("RENAMED_FROM",)


# --- Task 5: Integration test ---


def test_full_digest_pipeline(tmp_path):
    """End-to-end: digest content -> entities in KG -> entity lookup."""
    from brainlayer.pipeline.digest import digest_content, entity_lookup

    store = VectorStore(tmp_path / "test.db")
    dummy_embed = _dummy_embed

    # Digest some content
    result = digest_content(
        content="Etan Heyman and Dor Zohar discussed the Domica project at Cantaloupe AI headquarters.",
        store=store,
        embed_fn=dummy_embed,
        title="Team meeting",
        project="domica",
        participants=["Etan Heyman", "Dor Zohar"],
    )

    assert result["stats"]["entities_found"] >= 2

    # Now look up an entity
    entity = entity_lookup("Etan Heyman", store, dummy_embed)
    assert entity is not None
    assert entity["name"] == "Etan Heyman"
    # Should have evidence from the digest chunk
    assert len(entity["evidence"]) >= 1
