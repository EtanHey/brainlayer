"""Write-side guards for content that must never enter BrainLayer."""

from __future__ import annotations

import re

_JSONRPC_MESSAGE_RE = re.compile(r'"jsonrpc"\s*:\s*"2\.0"', re.IGNORECASE)
_INVALID_JSONRPC_MARKER = "mcp brainlayer memory: invalid json-rpc message"
_BRAIN_SEARCH_BOX_PREFIX = "┌─ brain_search:"


def recursive_mcp_output_reason(content: str | None) -> str | None:
    """Return a reason when content is BrainLayer MCP output being re-ingested."""
    if not content:
        return None

    stripped = str(content).lstrip()
    if stripped.startswith(_BRAIN_SEARCH_BOX_PREFIX):
        return "brain_search_output"

    folded = stripped.casefold()
    if _INVALID_JSONRPC_MARKER in folded:
        return "invalid_jsonrpc_mcp_output"
    if _JSONRPC_MESSAGE_RE.search(stripped):
        return "jsonrpc_message"

    return None


def reject_recursive_mcp_output(content: str | None) -> None:
    """Raise ValueError when content is recursive BrainLayer MCP output."""
    reason = recursive_mcp_output_reason(content)
    if reason:
        raise ValueError(f"recursive MCP output is not stored in BrainLayer: {reason}")
