"""Base MockMcpServer class with call logging and in-memory transport.

Uses the low-level MCP Server API (not FastMCP) to avoid parameter
typing constraints. Tools are registered with JSON schemas and receive
raw argument dicts — matching how the real brainlayer MCP server works.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

from mcp.client.session import ClientSession
from mcp.server import Server
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent, Tool


@dataclass
class ToolCall:
    """Record of a single tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    timestamp: float = field(default_factory=time.monotonic)
    result: Any = None


class MockMcpServer:
    """Base class for mock MCP servers with call logging.

    Uses the low-level Server API with @server.list_tools() and
    @server.call_tool() handlers. Subclasses register tools via
    _register_tools() which populates _tools and _handlers dicts.

    Usage:
        server = MockMcpServer("test-server")
        server.register_tool("my_tool", {"type": "object", "properties": {}}, handler)
        async with server.connect() as client:
            await client.call_tool("my_tool", {})
            assert server.was_called("my_tool")
    """

    def __init__(self, name: str = "mock-server"):
        self._server = Server(name)
        self._call_log: list[ToolCall] = []
        self._tools: dict[str, Tool] = {}
        self._handlers: dict[str, Callable] = {}
        self._register_tools()
        self._setup_server_handlers()

    def _setup_server_handlers(self) -> None:
        """Wire up the MCP list_tools and call_tool handlers."""
        mock_ref = self

        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return list(mock_ref._tools.values())

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any] | None = None) -> list[TextContent]:
            args = arguments or {}
            handler = mock_ref._handlers.get(name)
            if handler:
                result = handler(args)
            else:
                result = json.dumps({"mock": True, "tool": name})

            if not isinstance(result, str):
                result = json.dumps(result)

            call = ToolCall(tool_name=name, arguments=args, result=result)
            mock_ref._call_log.append(call)

            return [TextContent(type="text", text=result)]

    @property
    def call_log(self) -> list[ToolCall]:
        """Full ordered call log."""
        return list(self._call_log)

    @property
    def call_names(self) -> list[str]:
        """Just the tool names in call order."""
        return [c.tool_name for c in self._call_log]

    def reset(self) -> None:
        """Clear call log for a fresh test scenario."""
        self._call_log.clear()

    def register_tool(
        self,
        name: str,
        schema: dict[str, Any],
        handler: Callable[..., Any] | None = None,
        description: str = "",
    ) -> None:
        """Register a mock tool with optional custom handler."""
        self._tools[name] = Tool(
            name=name,
            description=description or f"Mock {name}",
            inputSchema=schema,
        )
        if handler:
            self._handlers[name] = handler

    def _register_tools(self) -> None:
        """Override in subclasses to register domain-specific tools."""

    # --- Query methods ---

    def was_called(self, tool_name: str) -> bool:
        return any(c.tool_name == tool_name for c in self._call_log)

    def call_count(self, tool_name: str) -> int:
        return sum(1 for c in self._call_log if c.tool_name == tool_name)

    def get_calls(self, tool_name: str) -> list[ToolCall]:
        return [c for c in self._call_log if c.tool_name == tool_name]

    def get_call_args(self, tool_name: str, index: int = 0) -> dict[str, Any]:
        calls = self.get_calls(tool_name)
        if index >= len(calls):
            raise IndexError(f"{tool_name} called {len(calls)} times, requested index {index}")
        return calls[index].arguments

    def called_before(self, first: str, second: str) -> bool:
        """True if first tool was called before second tool."""
        first_idx = next((i for i, c in enumerate(self._call_log) if c.tool_name == first), None)
        second_idx = next((i for i, c in enumerate(self._call_log) if c.tool_name == second), None)
        if first_idx is None or second_idx is None:
            return False
        return first_idx < second_idx

    def called_between(self, before: str, middle: str, after: str) -> bool:
        """True if middle was called between before and after."""
        before_idx = next((i for i, c in enumerate(self._call_log) if c.tool_name == before), None)
        middle_idx = next((i for i, c in enumerate(self._call_log) if c.tool_name == middle), None)
        after_idx = next((i for i, c in enumerate(self._call_log) if c.tool_name == after), None)
        if any(idx is None for idx in [before_idx, middle_idx, after_idx]):
            return False
        return before_idx < middle_idx < after_idx

    # --- Connection ---

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[ClientSession, None]:
        """Create an in-memory client session connected to this mock server."""
        async with create_connected_server_and_client_session(self._server) as session:
            yield session
