"""Per-session MCP tool palette selection."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from mcp.types import Tool

PROFILE_ENV = "BRAINLAYER_MCP_PROFILE"
CORE_TOOL_NAMES = (
    "brain_search",
    "brain_store",
    "brain_recall",
    "brain_expand",
)
EXPAND_TOOL_NAME = "expand_palette"

logger = logging.getLogger(__name__)


class ToolPalette:
    """Resolve one MCP session's visible tools and expansion state."""

    def __init__(self, profile: str | None = None) -> None:
        environment_profile = os.environ.get(PROFILE_ENV)
        raw_profile = environment_profile if profile is None else profile
        normalized = (raw_profile or "").strip().lower()

        self._full = normalized in {"full", "operator"}
        self._expanded = False
        if normalized not in {"", "core", "full", "operator"} and profile is None:
            logger.warning("Unknown %s=%r; using core profile", PROFILE_ENV, environment_profile)

    @property
    def is_full(self) -> bool:
        return self._full

    def expose(self, full_tools: Sequence[Tool]) -> list[Tool]:
        if self._full or self._expanded:
            return list(full_tools)

        tools_by_name = {tool.name: tool for tool in full_tools}
        core_tools = [tools_by_name[name] for name in CORE_TOOL_NAMES if name in tools_by_name]
        core_tools.append(
            Tool(
                name=EXPAND_TOOL_NAME,
                description="Expose all tools.",
                input_schema={"type": "object"},
            )
        )
        return core_tools

    def is_exposed(self, name: str) -> bool:
        return self._full or self._expanded or name in CORE_TOOL_NAMES or name == EXPAND_TOOL_NAME

    def expand(self, full_tool_names: Sequence[str]) -> dict[str, object]:
        if self._expanded:
            return {
                "expanded": False,
                "already_expanded": True,
                "registered_tools": [],
            }

        self._expanded = True
        return {
            "expanded": True,
            "already_expanded": False,
            "registered_tools": [name for name in full_tool_names if name not in CORE_TOOL_NAMES],
        }
