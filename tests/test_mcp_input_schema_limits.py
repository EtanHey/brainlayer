"""Tests for MCP input schema length limits."""

import asyncio
from pathlib import Path
from typing import Any

from mcp.types import TextContent

import brainlayer.mcp as mcp_module
from brainlayer.mcp import _full_tool_definitions
from brainlayer.mcp.palette import ToolPalette


def _get_tools():
    return _full_tool_definitions()


def _tool_input_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "input_schema", None)
    if isinstance(schema, dict):
        return schema
    schema = getattr(tool, "inputSchema", None)
    if isinstance(schema, dict):
        return schema
    raise AssertionError(f"{tool.name} is missing MCP input schema payload")


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
        schema = _tool_input_schema(tool)
        for field_path, string_schema in _iter_string_fields(schema):
            assert "maxLength" in string_schema, f"{tool.name}.{field_path} is missing maxLength"

        for field_path, array_schema, item_schema in _iter_string_arrays(schema):
            assert "maxItems" in array_schema, f"{tool.name}.{field_path} is missing maxItems"
            assert "maxLength" in item_schema, f"{tool.name}.{field_path}[] is missing maxLength"


def test_brain_digest_schema_limits_content_length():
    """maxLength is schema metadata for BrainBar; Python library does not reject oversize input."""
    digest_tool = next(tool for tool in _get_tools() if tool.name == "brain_digest")
    content_schema = _tool_input_schema(digest_tool)["properties"]["content"]
    assert content_schema["maxLength"] == 200_000


def test_call_tool_does_not_enforce_schema_max_length_at_python_library_layer(monkeypatch):
    """Oversize rejection is enforced by BrainBar MCPRouter — see Swift gate below."""
    captured = {}

    async def fake_digest(**kwargs):
        captured.update(kwargs)
        return ([TextContent(type="text", text="ok")], {"entities": 0})

    monkeypatch.setattr(mcp_module, "_tool_palette", ToolPalette("full"))
    monkeypatch.setattr(mcp_module, "_brain_digest", fake_digest)

    oversized = "x" * 200_001
    result = asyncio.run(mcp_module.call_tool("brain_digest", {"content": oversized}))
    assert captured["content"] == oversized
    assert result[0][0].text == "ok"


def test_brain_digest_max_length_enforced_in_brainbar_mcp_router_tests():
    """Cite the enforcing-layer behavioral test (BrainBar Swift, not Python library)."""
    swift_tests = Path(__file__).resolve().parents[1] / "brain-bar" / "Tests" / "BrainBarTests" / "MCPRouterTests.swift"
    source = swift_tests.read_text(encoding="utf-8")
    assert "func testBrainDigestRejectsOversizedContentAtSchemaLayer" in source
    assert "content length 200001 exceeds maxLength 200000" in source


def test_call_tool_success_results_return_content_tuple(monkeypatch):
    async def fake_store_new(**_kwargs):
        return (
            [TextContent(type="text", text="stored")],
            {"chunk_id": "chunk-1", "related": []},
        )

    monkeypatch.setattr(mcp_module, "_tool_palette", ToolPalette("full"))
    monkeypatch.setattr(mcp_module, "_store_new", fake_store_new)

    result = asyncio.run(mcp_module.call_tool("brain_store", {"content": "remember this"}))
    assert isinstance(result, tuple)
    assert result[0][0].text == "stored"
    assert result[1] == {"chunk_id": "chunk-1", "related": []}


def test_call_tool_error_results_use_call_tool_result(monkeypatch):
    async def failing_store_new(**_kwargs):
        return mcp_module._error_result("Store failed: database is locked")

    monkeypatch.setattr(mcp_module, "_tool_palette", ToolPalette("full"))
    monkeypatch.setattr(mcp_module, "_store_new", failing_store_new)

    result = asyncio.run(mcp_module.call_tool("brain_store", {"content": "remember this"}))

    assert result.is_error is True
    assert result.content[0].text == "Store failed: database is locked"
