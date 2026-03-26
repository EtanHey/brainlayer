"""Mock MCP servers for sandboxed behavioral agent testing.

Provides MockMcpServer base class with call logging and assertion helpers,
plus three domain mocks: mock-github, mock-brainlayer, mock-voicelayer.

Usage:
    server = MockGitHub()
    async with server.connect() as client:
        result = await client.call_tool("gh_pr_create", {"title": "Fix auth"})
        assert server.was_called("gh_pr_create")
        assert server.called_before("gh_pr_create", "gh_pr_merge")
"""

from .assertions import (
    assert_call_count,
    assert_call_sequence,
    assert_called_before,
    assert_called_between,
    assert_never_called,
)
from .base import MockMcpServer

__all__ = [
    "MockMcpServer",
    "assert_called_before",
    "assert_called_between",
    "assert_call_count",
    "assert_call_sequence",
    "assert_never_called",
]
