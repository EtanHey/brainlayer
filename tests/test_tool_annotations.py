"""Tests for B6: ToolAnnotations on all 12 MCP tools + B7: agent_id scoping on brain_store."""

import asyncio

import pytest


# ── B6: ToolAnnotations ────────────────────────────────────────────


EXPECTED_TOOLS = [
    "brain_search",
    "brain_store",
    "brain_recall",
    "brain_digest",
    "brain_entity",
    "brain_expand",
    "brain_update",
    "brain_tags",
    "brain_get_person",
    "brain_supersede",
    "brain_archive",
    "brain_enrich",
]

# Tools that only read data — no writes, no side effects
READ_ONLY_TOOLS = {
    "brain_search",
    "brain_recall",
    "brain_entity",
    "brain_expand",
    "brain_tags",
    "brain_get_person",
}

# Tools that write/modify data
WRITE_TOOLS = {
    "brain_store",
    "brain_digest",
    "brain_update",
    "brain_supersede",
    "brain_archive",
    "brain_enrich",
}

# Tools whose repeated calls produce the same result (safe to retry)
IDEMPOTENT_TOOLS = {
    "brain_search",
    "brain_recall",
    "brain_entity",
    "brain_expand",
    "brain_tags",
    "brain_get_person",
    "brain_update",  # updating to same values is idempotent
}

# Tools that destroy/remove data
DESTRUCTIVE_TOOLS = {
    "brain_archive",   # soft-deletes
    "brain_supersede",  # marks as superseded
}


class TestToolAnnotationsPresent:
    """Every tool MUST have ToolAnnotations with readOnlyHint, destructiveHint, idempotentHint."""

    def _get_tools(self):
        from brainlayer.mcp import list_tools
        return asyncio.run(list_tools())

    def test_all_12_tools_have_annotations(self):
        """Every one of the 12 tools must have a non-None annotations field."""
        tools = self._get_tools()
        tool_map = {t.name: t for t in tools}
        for name in EXPECTED_TOOLS:
            assert name in tool_map, f"Tool {name} missing from list_tools"
            tool = tool_map[name]
            assert tool.annotations is not None, f"Tool {name} has no annotations"

    def test_annotations_have_all_three_hints(self):
        """Each tool's annotations must specify all three hint fields."""
        tools = self._get_tools()
        for tool in tools:
            if tool.name not in EXPECTED_TOOLS:
                continue
            ann = tool.annotations
            assert ann is not None, f"{tool.name}: annotations is None"
            assert ann.readOnlyHint is not None, f"{tool.name}: readOnlyHint is None"
            assert ann.destructiveHint is not None, f"{tool.name}: destructiveHint is None"
            assert ann.idempotentHint is not None, f"{tool.name}: idempotentHint is None"

    @pytest.mark.parametrize("tool_name", sorted(READ_ONLY_TOOLS))
    def test_read_only_tools(self, tool_name):
        """Read-only tools must have readOnlyHint=True."""
        tools = self._get_tools()
        tool = next(t for t in tools if t.name == tool_name)
        assert tool.annotations.readOnlyHint is True, f"{tool_name} should be readOnly"

    @pytest.mark.parametrize("tool_name", sorted(WRITE_TOOLS))
    def test_write_tools(self, tool_name):
        """Write tools must have readOnlyHint=False."""
        tools = self._get_tools()
        tool = next(t for t in tools if t.name == tool_name)
        assert tool.annotations.readOnlyHint is False, f"{tool_name} should not be readOnly"

    @pytest.mark.parametrize("tool_name", sorted(DESTRUCTIVE_TOOLS))
    def test_destructive_tools(self, tool_name):
        """Destructive tools must have destructiveHint=True."""
        tools = self._get_tools()
        tool = next(t for t in tools if t.name == tool_name)
        assert tool.annotations.destructiveHint is True, f"{tool_name} should be destructive"

    @pytest.mark.parametrize("tool_name", sorted(IDEMPOTENT_TOOLS))
    def test_idempotent_tools(self, tool_name):
        """Idempotent tools must have idempotentHint=True."""
        tools = self._get_tools()
        tool = next(t for t in tools if t.name == tool_name)
        assert tool.annotations.idempotentHint is True, f"{tool_name} should be idempotent"


# ── B7: agent_id scoping on brain_store ────────────────────────────


class TestAgentIdScoping:
    """brain_store must accept agent_id parameter for per-agent tagging."""

    def test_brain_store_has_agent_id_param(self):
        """brain_store inputSchema must include agent_id as optional param."""
        from brainlayer.mcp import list_tools
        tools = asyncio.run(list_tools())
        store_tool = next(t for t in tools if t.name == "brain_store")
        props = store_tool.inputSchema["properties"]
        assert "agent_id" in props, "brain_store must have agent_id parameter"
        assert props["agent_id"]["type"] == "string"

    def test_agent_id_not_required(self):
        """agent_id should be optional — not in required list."""
        from brainlayer.mcp import list_tools
        tools = asyncio.run(list_tools())
        store_tool = next(t for t in tools if t.name == "brain_store")
        required = store_tool.inputSchema.get("required", [])
        assert "agent_id" not in required
