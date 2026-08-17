"""Tests for MCP input schema length limits."""

import asyncio
from typing import Any

from mcp.types import TextContent

import brainlayer.mcp as mcp_module
from brainlayer.mcp import _full_tool_definitions
from brainlayer.mcp.palette import ToolPalette


def _get_tools():
    return _full_tool_definitions()


def _iter_string_fields(schema: dict[str, Any], path: str = ""):
    schema_type = schema.get("type")
    if schema_type == "string":
        yield path, schema
        return

    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, dict) and items.get("type") == "string":
            yield f"{path}[]", items
        return

    if schema_type == "object":
        for prop_name, prop_schema in schema.get("properties", {}).items():
            next_path = f"{path}.{prop_name}" if path else prop_name
            yield from _iter_string_fields(prop_schema, next_path)


def _iter_string_arrays(schema: dict[str, Any], path: str = ""):
    schema_type = schema.get("type")
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, dict) and items.get("type") == "string":
            yield path, schema, items
        return

    if schema_type == "object":
        for prop_name, prop_schema in schema.get("properties", {}).items():
            next_path = f"{path}.{prop_name}" if path else prop_name
            yield from _iter_string_arrays(prop_schema, next_path)


def test_all_string_input_fields_have_max_length_and_string_arrays_have_max_items():
    for tool in _get_tools():
        schema = tool.inputSchema
        for field_path, string_schema in _iter_string_fields(schema):
            assert "maxLength" in string_schema, f"{tool.name}.{field_path} is missing maxLength"

        for field_path, array_schema, item_schema in _iter_string_arrays(schema):
            assert "maxItems" in array_schema, f"{tool.name}.{field_path} is missing maxItems"
            assert "maxLength" in item_schema, f"{tool.name}.{field_path}[] is missing maxLength"


def test_brain_digest_schema_limits_content_length():
    digest_tool = next(tool for tool in _get_tools() if tool.name == "brain_digest")
    content_schema = digest_tool.inputSchema["properties"]["content"]
    assert content_schema["maxLength"] == 200_000


def test_mcp_v2_adapter_preserves_combined_tool_results(monkeypatch):
    async def fake_store_new(**_kwargs):
        return (
            [TextContent(type="text", text="stored")],
            {"chunk_id": "chunk-1", "related": []},
        )

    monkeypatch.setattr(mcp_module, "_tool_palette", ToolPalette("full"))
    monkeypatch.setattr(mcp_module, "_store_new", fake_store_new)

    async def call_store():
        return await mcp_module.call_tool("brain_store", {"content": "remember this"})

    result = asyncio.run(call_store())
    assert isinstance(result, tuple)
    assert result[0][0].text == "stored"
    assert result[1] == {"chunk_id": "chunk-1", "related": []}


def test_mcp_v2_adapter_preserves_brain_store_error_message(monkeypatch):
    async def failing_store_new(**_kwargs):
        return mcp_module._error_result("Store failed: database is locked")

    monkeypatch.setattr(mcp_module, "_tool_palette", ToolPalette("full"))
    monkeypatch.setattr(mcp_module, "_store_new", failing_store_new)

    async def call_store():
        return await mcp_module.call_tool("brain_store", {"content": "remember this"})

    result = asyncio.run(call_store())

    assert result.is_error is True
    assert result.content[0].text == "Store failed: database is locked"
