import sqlite3

import pytest

from brainlayer.drain import drain_once
from brainlayer.queue_io import enqueue_hook_chunk, enqueue_store, enqueue_watcher_chunk
from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore
from brainlayer.watcher_bridge import should_skip_chunk_content, should_skip_entry

JSONRPC_RECURSION_CONTENT = (
    'MCP BrainLayer Memory: Invalid JSON-RPC message: {"jsonrpc":"2.0","id":24,'
    '"result":{"content":[{"type":"text","text":"recursive output"}]}}'
)


def _make_entry(text: str) -> dict:
    return {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
        "timestamp": "2026-05-16T12:00:00Z",
    }


def test_watcher_preclassify_rejects_brain_search_output_box():
    entry = _make_entry('┌─ brain_search: "audit recursion" ─ 1 result\n│\n└─')

    assert should_skip_entry(entry) == "recursive_mcp_output"


def test_watcher_preclassify_rejects_other_brainlayer_mcp_output_boxes():
    entry = _make_entry("┌─ Entity search: Etan\n│ recursive entity output\n└─")

    assert should_skip_entry(entry) == "recursive_mcp_output"


def test_watcher_postchunk_rejects_jsonrpc_mcp_memory_output():
    assert should_skip_chunk_content(JSONRPC_RECURSION_CONTENT) == "recursive_mcp_output"


def test_direct_store_rejects_recursive_mcp_output(tmp_path):
    with VectorStore(tmp_path / "store-guard.db") as store:
        with pytest.raises(ValueError, match="recursive MCP output"):
            store_memory(
                store=store,
                embed_fn=None,
                content=JSONRPC_RECURSION_CONTENT,
                memory_type="note",
                project="brainlayer",
                tags=["correction:factual", "auto-detected"],
            )


def test_vector_upsert_rejects_recursive_mcp_output(tmp_path):
    with VectorStore(tmp_path / "upsert-guard.db") as store:
        with pytest.raises(ValueError, match="recursive MCP output"):
            store.upsert_chunks(
                [
                    {
                        "id": "rt-guarded-jsonrpc",
                        "content": JSONRPC_RECURSION_CONTENT,
                        "metadata": {},
                        "source_file": "session.jsonl",
                        "project": "brainlayer",
                        "content_type": "assistant_text",
                        "char_count": len(JSONRPC_RECURSION_CONTENT),
                    }
                ],
                [[0.1] * 1024],
            )


def test_update_chunk_rejects_recursive_mcp_output(tmp_path):
    with VectorStore(tmp_path / "update-guard.db") as store:
        store.upsert_chunks(
            [
                {
                    "id": "safe-update-target",
                    "content": "ordinary memory before attempted recursive update",
                    "metadata": {},
                    "source_file": "session.jsonl",
                    "project": "brainlayer",
                    "content_type": "assistant_text",
                    "char_count": 47,
                }
            ],
            [[0.2] * 1024],
        )

        with pytest.raises(ValueError, match="recursive MCP output"):
            store.update_chunk("safe-update-target", content=JSONRPC_RECURSION_CONTENT)

        assert store.get_chunk("safe-update-target")["content"] == "ordinary memory before attempted recursive update"


def test_drain_drops_recursive_mcp_output_events(tmp_path, monkeypatch):
    db_path = tmp_path / "drain-guard.db"
    queue_dir = tmp_path / "queue"
    VectorStore(db_path).close()
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    enqueue_store(
        chunk_id="queued-store-recursion",
        content=JSONRPC_RECURSION_CONTENT,
        project="brainlayer",
        tags=["correction:factual", "auto-detected"],
        queue_dir=queue_dir,
    )
    enqueue_watcher_chunk(
        chunk_id="queued-watcher-recursion",
        content='{"jsonrpc":"2.0","id":24,"result":"recursive watcher output"}',
        metadata={},
        source_file="session.jsonl",
        project="brainlayer",
        content_type="assistant_text",
        value_type="HIGH",
        created_at="2026-05-16T12:00:00Z",
        conversation_id="session",
        tags=["auto-detected"],
        queue_dir=queue_dir,
    )
    enqueue_hook_chunk(
        session_id="33abe108-session",
        content="MCP BrainLayer Memory: Invalid JSON-RPC message: recursive hook output",
        project="brainlayer",
        queue_dir=queue_dir,
    )

    drained = drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10, log_path=tmp_path / "drain.log")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, content FROM chunks").fetchall()

    assert drained == 3
    assert rows == []
    assert not list(queue_dir.glob("*.jsonl"))
