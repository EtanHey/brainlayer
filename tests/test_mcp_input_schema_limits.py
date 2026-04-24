"""Tests for MCP input schema length limits."""

import asyncio
from typing import Any

from mcp import types

from brainlayer.mcp import list_tools, server


def _get_tools():
    return asyncio.run(list_tools())


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


async def _call_brain_digest(arguments: dict[str, Any]):
    handler = server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(params=types.CallToolRequestParams(name="brain_digest", arguments=arguments))
    return await handler(request)


def test_brain_digest_schema_rejects_oversized_content():
    result = asyncio.run(_call_brain_digest({"content": "x" * 200_001})).root

    assert result.isError is True
    assert result.content, "Expected error content to be non-empty"
    text = result.content[0].text
    assert "Input validation error:" in text
    assert "is too long" in text
