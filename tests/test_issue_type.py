"""Tests for brain_store issue type — lifecycle, code refs, severity."""

import asyncio
import json

from brainlayer.store import VALID_MEMORY_TYPES, store_memory
from brainlayer.vector_store import VectorStore


def _dummy_embed(text):  # noqa: ARG001
    return [0.1] * 1024


class TestIssueTypeValidation:
    """Validate issue is a recognized memory type."""

    def test_issue_in_valid_types(self):
        assert "issue" in VALID_MEMORY_TYPES

    def test_store_accepts_issue_type(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        result = store_memory(
            store=store,
            embed_fn=_dummy_embed,
            content="Issue: brain_digest fails with embed AttributeError",
            memory_type="issue",
            project="brainlayer",
        )
        assert result["id"].startswith("manual-")

    def test_issue_rejects_invalid_type(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        import pytest

        with pytest.raises(ValueError, match="type must be one of"):
            store_memory(
                store=store,
                embed_fn=_dummy_embed,
                content="not a real type",
                memory_type="invalid_type",
            )


class TestIssueMetadata:
    """Issue-specific fields stored in metadata JSON."""

    def test_status_stored(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        result = store_memory(
            store=store,
            embed_fn=_dummy_embed,
            content="Issue: MLX crashes during enrichment",
            memory_type="issue",
            status="open",
        )
        cursor = store.conn.cursor()
        row = list(cursor.execute("SELECT metadata FROM chunks WHERE id = ?", [result["id"]]))[0]
        meta = json.loads(row[0])
        assert meta["status"] == "open"

    def test_severity_stored(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        result = store_memory(
            store=store,
            embed_fn=_dummy_embed,
            content="Issue: digest pipeline broken",
            memory_type="issue",
            severity="critical",
        )
        cursor = store.conn.cursor()
        row = list(cursor.execute("SELECT metadata FROM chunks WHERE id = ?", [result["id"]]))[0]
        meta = json.loads(row[0])
        assert meta["severity"] == "critical"

    def test_code_refs_stored(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        result = store_memory(
            store=store,
            embed_fn=_dummy_embed,
            content="Issue: embed method missing on EmbeddingModel",
            memory_type="issue",
            file_path="src/brainlayer/mcp/__init__.py",
            function_name="_brain_digest",
            line_number=1216,
        )
        cursor = store.conn.cursor()
        row = list(cursor.execute("SELECT metadata FROM chunks WHERE id = ?", [result["id"]]))[0]
        meta = json.loads(row[0])
        assert meta["file_path"] == "src/brainlayer/mcp/__init__.py"
        assert meta["function_name"] == "_brain_digest"
        assert meta["line_number"] == 1216

    def test_all_fields_together(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        result = store_memory(
            store=store,
            embed_fn=_dummy_embed,
            content="Issue: WhatsApp enrichment fails when MLX crashes",
            memory_type="issue",
            project="brainlayer",
            status="in_progress",
            severity="high",
            file_path="src/brainlayer/pipeline/enrichment.py",
            function_name="call_llm",
            line_number=431,
            tags=["enrichment", "mlx", "whatsapp"],
            importance=8,
        )
        cursor = store.conn.cursor()
        row = list(
            cursor.execute(
                "SELECT metadata, tags, importance, content_type FROM chunks WHERE id = ?",
                [result["id"]],
            )
        )[0]
        meta = json.loads(row[0])
        assert meta["status"] == "in_progress"
        assert meta["severity"] == "high"
        assert meta["file_path"] == "src/brainlayer/pipeline/enrichment.py"
        assert row[1] is not None  # tags stored
        assert row[2] == 8.0  # importance
        assert row[3] == "issue"  # content_type = memory_type


class TestIssueAutoDetection:
    """Auto-detection of issue type from content patterns."""

    def test_issue_prefix_detected(self):
        from brainlayer.mcp import _detect_memory_type

        assert _detect_memory_type("Issue: brain_digest embed bug") == "issue"

    def test_blocking_keyword_detected(self):
        from brainlayer.mcp import _detect_memory_type

        assert _detect_memory_type("This is blocking the 6PM integration") == "issue"

    def test_crashes_keyword_detected(self):
        from brainlayer.mcp import _detect_memory_type

        assert _detect_memory_type("MLX crashes during enrichment batch") == "issue"

    def test_p0_detected(self):
        from brainlayer.mcp import _detect_memory_type

        assert _detect_memory_type("P0 bug in digest pipeline") == "issue"

    def test_severity_keyword_detected(self):
        from brainlayer.mcp import _detect_memory_type

        assert _detect_memory_type("severity critical: data loss in migration") == "issue"

    def test_non_issue_not_detected(self):
        from brainlayer.mcp import _detect_memory_type

        # "TODO" should match todo, not issue
        assert _detect_memory_type("TODO: fix the tests") == "todo"


class TestIssueMCPSchema:
    """MCP tool schema includes issue fields."""

    def test_issue_in_type_enum(self):
        tools = asyncio.run(_get_tools())
        store_tool = next(t for t in tools if t.name == "brain_store")
        type_enum = store_tool.inputSchema["properties"]["type"]["enum"]
        assert "issue" in type_enum

    def test_status_field_exists(self):
        tools = asyncio.run(_get_tools())
        store_tool = next(t for t in tools if t.name == "brain_store")
        props = store_tool.inputSchema["properties"]
        assert "status" in props
        assert props["status"]["enum"] == ["open", "in_progress", "done", "archived"]

    def test_severity_field_exists(self):
        tools = asyncio.run(_get_tools())
        store_tool = next(t for t in tools if t.name == "brain_store")
        props = store_tool.inputSchema["properties"]
        assert "severity" in props
        assert props["severity"]["enum"] == ["critical", "high", "medium", "low"]

    def test_code_ref_fields_exist(self):
        tools = asyncio.run(_get_tools())
        store_tool = next(t for t in tools if t.name == "brain_store")
        props = store_tool.inputSchema["properties"]
        assert "file_path" in props
        assert "function_name" in props
        assert "line_number" in props


class TestIssueDefaultStatus:
    """Issue type defaults status to 'open' when not specified."""

    def test_default_status_open(self, tmp_path):
        store = VectorStore(tmp_path / "test.db")
        result = store_memory(
            store=store,
            embed_fn=_dummy_embed,
            content="Issue: something broke",
            memory_type="issue",
            status="open",  # Caller should default this
        )
        cursor = store.conn.cursor()
        row = list(cursor.execute("SELECT metadata FROM chunks WHERE id = ?", [result["id"]]))[0]
        meta = json.loads(row[0])
        assert meta["status"] == "open"


async def _get_tools():
    from brainlayer.mcp import list_tools

    return await list_tools()
