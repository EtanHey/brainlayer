"""Tests for MCP input schema length limits."""

import asyncio
from typing import Any

from mcp.client import Client
from mcp.types import TextContent

import brainlayer.mcp as mcp_module
from brainlayer.mcp import _full_tool_definitions, server
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
        schema = tool.input_schema
        for field_path, string_schema in _iter_string_fields(schema):
            assert "maxLength" in string_schema, f"{tool.name}.{field_path} is missing maxLength"

        for field_path, array_schema, item_schema in _iter_string_arrays(schema):
            assert "maxItems" in array_schema, f"{tool.name}.{field_path} is missing maxItems"
            assert "maxLength" in item_schema, f"{tool.name}.{field_path}[] is missing maxLength"


async def _call_brain_digest(arguments: dict[str, Any]):
    async with Client(server, mode="legacy") as client:
        return await client.call_tool("brain_digest", arguments)


def test_brain_digest_schema_rejects_oversized_content(monkeypatch):
    monkeypatch.setattr(mcp_module, "_tool_palette", ToolPalette("full"))
    result = asyncio.run(_call_brain_digest({"content": "x" * 200_001}))

    assert result.is_error is True
    assert result.content, "Expected error content to be non-empty"
    text = result.content[0].text
    assert "Input validation error:" in text
    assert "is too long" in text


def test_mcp_v2_adapter_preserves_combined_tool_results(monkeypatch):
    async def fake_store_new(**_kwargs):
        return (
            [TextContent(type="text", text="stored")],
            {"chunk_id": "chunk-1", "related": []},
        )

    monkeypatch.setattr(mcp_module, "_tool_palette", ToolPalette("full"))
    monkeypatch.setattr(mcp_module, "_store_new", fake_store_new)

    async def call_store():
        async with Client(server, mode="legacy") as client:
            return await client.call_tool("brain_store", {"content": "remember this"})

    result = asyncio.run(call_store())

    assert result.is_error is False
    assert result.content[0].text == "stored"
    assert result.structured_content == {"chunk_id": "chunk-1", "related": []}
