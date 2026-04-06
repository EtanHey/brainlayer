"""Tests for brain_entity type sync and list action.

Verifies:
1. MCP enum matches entity_type_hierarchy seed types
2. brain_entity schema includes action param
3. brain_entity schema includes limit/offset params
4. KGMixin.list_entities works with temp DB
5. KGMixin.list_entities filters by entity_type
"""

import uuid

import pytest

from brainlayer.mcp.entity_handler import _format_entity_list
from brainlayer.vector_store import VectorStore

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore for testing."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


def _get_brain_entity_tool():
    """Extract the brain_entity Tool object from the MCP server tool list."""
    import asyncio

    from brainlayer.mcp import list_tools

    tools = asyncio.run(list_tools())
    for tool in tools:
        if tool.name == "brain_entity":
            return tool
    pytest.fail("brain_entity tool not found in MCP server tool list")


# ── Hierarchy Seed Types ────────────────────────────────────────

# These are the 17 types seeded in vector_store.py _init_db()
_HIERARCHY_SEED_TYPES = {
    "agent",
    "person",
    "tool",
    "project",
    "concept",
    "topic",
    "protocol",
    "community",
    "health_metric",
    "workflow",
    "device",
    "event",
    "organization",
    "golem",
    "platform",
    "skill",
    "decision",
}

# Additional types that exist in production DB but not in hierarchy seed
_EXTRA_DB_TYPES = {"technology", "library", "company", "location"}


# ── Schema Tests ────────────────────────────────────────────────


class TestEntityTypeEnum:
    """Verify MCP entity_type enum matches hierarchy seed."""

    def test_entity_type_enum_matches_hierarchy(self):
        """Every type in hierarchy seed must exist in the MCP enum."""
        tool = _get_brain_entity_tool()
        schema = tool.inputSchema
        enum_values = set(schema["properties"]["entity_type"]["enum"])
        missing = _HIERARCHY_SEED_TYPES - enum_values
        assert not missing, f"Hierarchy types missing from MCP enum: {missing}"

    def test_entity_type_enum_includes_extra_db_types(self):
        """Extra DB types (technology, library, company, location) must be in enum."""
        tool = _get_brain_entity_tool()
        schema = tool.inputSchema
        enum_values = set(schema["properties"]["entity_type"]["enum"])
        missing = _EXTRA_DB_TYPES - enum_values
        assert not missing, f"Extra DB types missing from MCP enum: {missing}"


class TestEntitySchemaParams:
    """Verify brain_entity schema has action, limit, offset params."""

    def test_brain_entity_schema_has_action_param(self):
        tool = _get_brain_entity_tool()
        props = tool.inputSchema["properties"]
        assert "action" in props, "action param missing from brain_entity schema"
        assert props["action"]["enum"] == ["lookup", "list"]
        assert props["action"]["default"] == "lookup"

    def test_brain_entity_schema_has_limit_offset(self):
        tool = _get_brain_entity_tool()
        props = tool.inputSchema["properties"]
        assert "limit" in props, "limit param missing from brain_entity schema"
        assert props["limit"]["type"] == "integer"
        assert props["limit"]["default"] == 20
        assert props["limit"]["minimum"] == 1
        assert props["limit"]["maximum"] == 100

        assert "offset" in props, "offset param missing from brain_entity schema"
        assert props["offset"]["type"] == "integer"
        assert props["offset"]["default"] == 0
        assert props["offset"]["minimum"] == 0

    def test_brain_entity_query_not_required(self):
        """query should not be required (list action doesn't need it)."""
        tool = _get_brain_entity_tool()
        required = tool.inputSchema.get("required", [])
        assert "query" not in required, "query should not be required (list action doesn't use it)"


# ── DB Tests: list_entities ─────────────────────────────────────


class TestListEntities:
    """Test KGMixin.list_entities with a temp DB."""

    def _add_entity(self, store, name, entity_type, importance=0.5, description=None):
        """Helper to insert an entity."""
        eid = str(uuid.uuid4())
        store.upsert_entity(
            entity_id=eid,
            entity_type=entity_type,
            name=name,
            importance=importance,
            description=description,
        )
        return eid

    def test_list_entities_method(self, store):
        """list_entities returns correct structure with seeded data."""
        self._add_entity(store, "Python", "technology", importance=0.9, description="Programming language")
        self._add_entity(store, "BrainLayer", "project", importance=0.8, description="Memory system")
        self._add_entity(store, "Etan", "person", importance=0.7, description="Developer")

        result = store.list_entities()
        assert "total" in result
        assert "entities" in result
        assert result["total"] == 3
        assert len(result["entities"]) == 3
        # Sorted by importance DESC
        assert result["entities"][0]["name"] == "Python"
        assert result["entities"][1]["name"] == "BrainLayer"
        assert result["entities"][2]["name"] == "Etan"

    def test_list_entities_with_type_filter(self, store):
        """list_entities filters by entity_type correctly."""
        self._add_entity(store, "Python", "technology", importance=0.9)
        self._add_entity(store, "TypeScript", "technology", importance=0.8)
        self._add_entity(store, "BrainLayer", "project", importance=0.7)

        result = store.list_entities(entity_type="technology")
        assert result["total"] == 2
        assert len(result["entities"]) == 2
        assert all(e["entity_type"] == "technology" for e in result["entities"])

    def test_list_entities_includes_legacy_null_status_rows(self, store):
        """list_entities includes pre-migration rows whose status was never backfilled."""
        entity_id = self._add_entity(store, "Legacy", "topic", importance=0.4)
        store.conn.cursor().execute("UPDATE kg_entities SET status = NULL WHERE id = ?", (entity_id,))

        result = store.list_entities()

        assert result["total"] == 1
        assert result["entities"][0]["name"] == "Legacy"

    def test_list_entities_pagination(self, store):
        """list_entities respects limit and offset."""
        for i in range(5):
            self._add_entity(store, f"Entity{i}", "topic", importance=0.5 + i * 0.01)

        page1 = store.list_entities(limit=2, offset=0)
        assert page1["total"] == 5
        assert len(page1["entities"]) == 2

        page2 = store.list_entities(limit=2, offset=2)
        assert page2["total"] == 5
        assert len(page2["entities"]) == 2

        # No overlap between pages
        page1_names = {e["name"] for e in page1["entities"]}
        page2_names = {e["name"] for e in page2["entities"]}
        assert page1_names.isdisjoint(page2_names)

    def test_list_entities_empty(self, store):
        """list_entities returns empty when no entities exist."""
        result = store.list_entities()
        assert result["total"] == 0
        assert result["entities"] == []

    def test_list_entities_entity_fields(self, store):
        """list_entities returns all expected fields per entity."""
        self._add_entity(store, "TestEntity", "tool", importance=0.6, description="A test tool")

        result = store.list_entities()
        entity = result["entities"][0]
        assert "id" in entity
        assert entity["entity_type"] == "tool"
        assert entity["name"] == "TestEntity"
        assert entity["description"] == "A test tool"
        assert entity["importance"] == 0.6
        assert "created_at" in entity


def test_format_entity_list_shows_zero_importance():
    """importance=0 should render instead of being treated as missing."""
    formatted = _format_entity_list(
        {
            "total": 1,
            "entities": [
                {
                    "entity_type": "topic",
                    "name": "Zero",
                    "importance": 0,
                    "description": "Zero importance still matters for display",
                }
            ],
        }
    )

    assert "(importance=0)" in formatted
