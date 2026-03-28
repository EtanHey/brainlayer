"""Mock BrainLayer MCP server — simulates brain_search/store/recall.

Returns test data fixtures. Captures all stores for assertion.
In-memory storage so tests can verify what was stored and searched.

Usage:
    brain = MockBrainLayer()
    async with brain.connect() as client:
        await client.call_tool("brain_store", {"content": "Auth uses JWT", "tags": ["auth"]})
        result = await client.call_tool("brain_search", {"query": "authentication"})

        assert brain.call_count("brain_store") == 1
        assert brain.stored_items[0]["content"] == "Auth uses JWT"
"""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any

from .base import MockMcpServer

# --- Default fixtures ---

DEFAULT_SEARCH_RESULTS = [
    {
        "chunk_id": "chunk_001",
        "content": "Authentication uses JWT tokens with 24h expiry.",
        "score": 0.92,
        "project": "brainlayer",
        "content_type": "decision",
        "tags": ["auth", "jwt"],
        "importance": 8,
        "date": "2026-03-20",
    },
    {
        "chunk_id": "chunk_002",
        "content": "Rate limiting set to 100 req/min per API key.",
        "score": 0.85,
        "project": "brainlayer",
        "content_type": "architecture",
        "tags": ["api", "rate-limit"],
        "importance": 7,
        "date": "2026-03-18",
    },
]

DEFAULT_RECALL = {
    "mode": "context",
    "project": "brainlayer",
    "summary": "Working on mock MCP harness for behavioral testing.",
    "recent_decisions": ["Use InMemoryTransport for mock servers"],
    "active_files": ["tests/mock_mcp/base.py"],
}


class MockBrainLayer(MockMcpServer):
    """Mock BrainLayer MCP server with in-memory storage."""

    def __init__(
        self,
        search_fixture: list[dict[str, Any]] | None = None,
        recall_fixture: dict[str, Any] | None = None,
    ):
        self._search_fixture = search_fixture if search_fixture is not None else copy.deepcopy(DEFAULT_SEARCH_RESULTS)
        self._recall_fixture = recall_fixture if recall_fixture is not None else copy.deepcopy(DEFAULT_RECALL)
        self.stored_items: list[dict[str, Any]] = []
        super().__init__("mock-brainlayer")

    def _register_tools(self) -> None:
        self.register_tool(
            "brain_search",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "project": {"type": "string", "description": "Filter by project"},
                    "tag": {"type": "string", "description": "Filter by tag"},
                    "importance_min": {"type": "integer", "description": "Minimum importance"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
            handler=self._handle_search,
            description="Semantic search across indexed memories",
        )

        self.register_tool(
            "brain_store",
            {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to store"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization",
                    },
                    "importance": {"type": "integer", "description": "Importance 1-10"},
                    "type": {"type": "string", "description": "Memory type"},
                },
                "required": ["content"],
            },
            handler=self._handle_store,
            description="Store a memory (decision, learning, todo)",
        )

        self.register_tool(
            "brain_recall",
            {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Recall mode",
                        "enum": ["context", "sessions", "operations", "plan", "summary", "stats"],
                        "default": "context",
                    },
                },
            },
            handler=self._handle_recall,
            description="Recall session or operational context",
        )

    def _handle_search(self, args: dict[str, Any]) -> str:
        query = args.get("query", "")
        # Filter fixtures by query keyword match (simple substring)
        results = [r for r in self._search_fixture if query.lower() in r.get("content", "").lower()]
        # If no match, return all fixtures (behavioral testing cares about call sequence, not results)
        if not results:
            results = self._search_fixture

        return json.dumps(
            {
                "query": query,
                "total": len(results),
                "results": results,
            }
        )

    def _handle_store(self, args: dict[str, Any]) -> str:
        chunk_id = f"mock_{uuid.uuid4().hex[:8]}"
        item = {
            "chunk_id": chunk_id,
            "content": args.get("content", ""),
            "tags": args.get("tags", []),
            "importance": args.get("importance", 5),
            "type": args.get("type", "auto"),
        }
        self.stored_items.append(item)
        return json.dumps({"stored": True, "chunk_id": chunk_id})

    def _handle_recall(self, args: dict[str, Any]) -> str:
        return json.dumps(self._recall_fixture)

    # --- Test helpers ---

    def add_search_fixture(self, content: str, **kwargs: Any) -> None:
        """Add a search result to the fixture pool."""
        self._search_fixture.append(
            {
                "chunk_id": f"fixture_{uuid.uuid4().hex[:8]}",
                "content": content,
                "score": kwargs.get("score", 0.80),
                "project": kwargs.get("project", "test"),
                "content_type": kwargs.get("content_type", "note"),
                "tags": kwargs.get("tags", []),
                "importance": kwargs.get("importance", 5),
                "date": kwargs.get("date", "2026-03-26"),
            }
        )

    def clear_fixtures(self) -> None:
        """Clear search fixtures and stored items (does not clear call log)."""
        self._search_fixture.clear()
        self.stored_items.clear()

    def reset(self) -> None:
        """Clear call log, restore default fixtures, and clear stored items."""
        super().reset()
        self._search_fixture = copy.deepcopy(DEFAULT_SEARCH_RESULTS)
        self.stored_items.clear()
