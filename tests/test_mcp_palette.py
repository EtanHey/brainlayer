"""Tests for the stateful BrainLayer MCP tool palette."""

import asyncio

import pytest

import brainlayer.mcp as mcp_module
from brainlayer.mcp import _full_tool_definitions, call_tool, list_tools
from brainlayer.mcp.palette import CORE_TOOL_NAMES, ToolPalette

CORE_WITH_CONTROL = (*CORE_TOOL_NAMES, "expand_palette")


@pytest.mark.parametrize("profile", [None, "", "   ", "core", "bogus"])
def test_core_profiles_fail_closed(monkeypatch, profile):
    monkeypatch.delenv("BRAINLAYER_MCP_PROFILE", raising=False)
    palette = ToolPalette(profile)

    assert tuple(tool.name for tool in palette.expose(_full_tool_definitions())) == CORE_WITH_CONTROL


@pytest.mark.parametrize("profile", ["full", "operator"])
def test_full_profiles_preserve_all_python_tools(profile):
    tools = ToolPalette(profile).expose(_full_tool_definitions())

    assert len(tools) == 13
    assert tuple(tool.name for tool in tools) == tuple(tool.name for tool in _full_tool_definitions())


def test_environment_profile_is_resolved_once(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MCP_PROFILE", "operator")
    palette = ToolPalette()
    monkeypatch.setenv("BRAINLAYER_MCP_PROFILE", "core")

    assert len(palette.expose(_full_tool_definitions())) == 13


def test_python_palette_expands_once_and_dispatches_deferred_tools(monkeypatch):
    palette = ToolPalette("core")
    monkeypatch.setattr(mcp_module, "_tool_palette", palette)

    assert tuple(tool.name for tool in asyncio.run(list_tools())) == CORE_WITH_CONTROL

    before = asyncio.run(call_tool("brain_tags", {}))
    assert before.is_error is True
    assert "not exposed" in before.content[0].text

    first = asyncio.run(call_tool("expand_palette", {}))
    assert first.is_error is False
    assert first.structured_content == {
        "expanded": True,
        "already_expanded": False,
        "registered_tools": [tool.name for tool in _full_tool_definitions() if tool.name not in CORE_TOOL_NAMES],
    }
    assert len(asyncio.run(list_tools())) == 13

    after = asyncio.run(call_tool("brain_tags", {}))
    assert "deprecated" in after.content[0].text

    second = asyncio.run(call_tool("expand_palette", {}))
    assert second.structured_content == {
        "expanded": False,
        "already_expanded": True,
        "registered_tools": [],
    }
    assert len(asyncio.run(list_tools())) == 13
