import sqlite3

import pytest

from brainlayer.drain import drain_once
from brainlayer.queue_io import enqueue_hook_chunk, enqueue_store, enqueue_watcher_chunk
from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore
from brainlayer.watcher_bridge import create_flush_callback, should_skip_chunk_content, should_skip_entry

JSONRPC_RECURSION_CONTENT = (
    'MCP BrainLayer Memory: Invalid JSON-RPC message: {"jsonrpc":"2.0","id":24,'
    '"result":{"content":[{"type":"text","text":"recursive output"}]}}'
)

RT_AGENT_A7_JUDGE_NOTES_CONTENT = (
    "Key observations:\n"
    '- **qid=20 "Etan TechGym title abstract sent WhatsApp Sagit content"**: '
    "Pair 49 contains Sagit context but says pre-draft.\n"
    '- **qid=22 "two-sentence hook opener"**: Both pairs are FM6 PreCompact pollution.\n'
    "Final JSONL fields: judge_agent_name, failure_modes_observed, judge_reasoning."
)

RT_AGENT_CONTEXT_ONLY_JUDGE_NOTES_CONTENT = (
    "Final judge pass summary:\n"
    "judge_agent_name=rt-eval-a7\n"
    "failure_modes_observed=[FM6, FM11]\n"
    "judge_reasoning=These are benchmark comparison notes, not durable user memory."
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


def test_watcher_preclassify_rejects_rt_agent_qid_judge_notes():
    entry = _make_entry(RT_AGENT_A7_JUDGE_NOTES_CONTENT)

    assert should_skip_entry(entry) == "recursive_mcp_output"


def test_watcher_postchunk_rejects_rt_agent_qid_judge_notes():
    assert should_skip_chunk_content(RT_AGENT_A7_JUDGE_NOTES_CONTENT) == "recursive_mcp_output"


def test_watcher_preclassify_rejects_rt_agent_context_only_judge_notes_from_subagent_source():
    entry = _make_entry(RT_AGENT_CONTEXT_ONLY_JUDGE_NOTES_CONTENT)
    entry["_source_file"] = (
        "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-orchestrator/"
        "session/subagents/agent-a7823570938b54ccd.jsonl"
    )

    assert should_skip_entry(entry) == "recursive_mcp_output"


def test_watcher_postchunk_rejects_rt_agent_context_only_judge_notes_from_source_context():
    source_file = (
        "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-orchestrator/"
        "session/subagents/agent-a7823570938b54ccd.jsonl"
    )

    assert (
        should_skip_chunk_content(
            RT_AGENT_CONTEXT_ONLY_JUDGE_NOTES_CONTENT,
            chunk_id="ordinary-watcher-id",
            source_file=source_file,
        )
        == "recursive_mcp_output"
    )


def test_non_arbitrated_watcher_drops_rt_agent_context_only_judge_notes(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_ARBITRATED", raising=False)
    db_path = tmp_path / "watcher-direct-rt-agent-guard.db"
    flush = create_flush_callback(db_path)
    entry = _make_entry(RT_AGENT_CONTEXT_ONLY_JUDGE_NOTES_CONTENT)
    entry["_source_file"] = (
        "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-orchestrator/"
        "session/subagents/agent-a7823570938b54ccd.jsonl"
    )

    flush([entry])

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, content FROM chunks").fetchall()

    assert rows == []


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


def test_direct_store_rejects_rt_agent_qid_judge_notes(tmp_path):
    with VectorStore(tmp_path / "store-rt-agent-guard.db") as store:
        with pytest.raises(ValueError, match="rt-agent judge notes"):
            store_memory(
                store=store,
                embed_fn=None,
                content=RT_AGENT_A7_JUDGE_NOTES_CONTENT,
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


def test_vector_upsert_rejects_rt_agent_a7_qid_judge_notes(tmp_path):
    with VectorStore(tmp_path / "upsert-rt-agent-guard.db") as store:
        with pytest.raises(ValueError, match="rt-agent judge notes"):
            store.upsert_chunks(
                [
                    {
                        "id": "rt-agent-a7-deadbeefcafefeed",
                        "content": RT_AGENT_A7_JUDGE_NOTES_CONTENT,
                        "metadata": {},
                        "source_file": (
                            "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-orchestrator/"
                            "session/subagents/agent-a7823570938b54ccd.jsonl"
                        ),
                        "project": "brainlayer",
                        "content_type": "assistant_text",
                        "char_count": len(RT_AGENT_A7_JUDGE_NOTES_CONTENT),
                    }
                ],
                [[0.1] * 1024],
            )


def test_vector_upsert_rejects_rt_agent_context_only_judge_notes(tmp_path):
    with VectorStore(tmp_path / "upsert-rt-agent-context-guard.db") as store:
        with pytest.raises(ValueError, match="rt-agent judge notes"):
            store.upsert_chunks(
                [
                    {
                        "id": "rt-agent-a7-contextonly",
                        "content": RT_AGENT_CONTEXT_ONLY_JUDGE_NOTES_CONTENT,
                        "metadata": {},
                        "source_file": (
                            "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-orchestrator/"
                            "session/subagents/agent-a7823570938b54ccd.jsonl"
                        ),
                        "project": "brainlayer",
                        "content_type": "assistant_text",
                        "char_count": len(RT_AGENT_CONTEXT_ONLY_JUDGE_NOTES_CONTENT),
                    }
                ],
                [[0.1] * 1024],
            )


def test_vector_upsert_allows_non_judge_rt_agent_chunk(tmp_path):
    with VectorStore(tmp_path / "upsert-rt-agent-allowed.db") as store:
        count = store.upsert_chunks(
            [
                {
                    "id": "rt-agent-a7-feedfacecafebeef",
                    "content": "Explore the landing page animation and summarize the CSS keyframe structure.",
                    "metadata": {},
                    "source_file": (
                        "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-voicelayer/"
                        "session/subagents/agent-a7c933d4439ab60b3.jsonl"
                    ),
                    "project": "brainlayer",
                    "content_type": "assistant_text",
                    "char_count": 75,
                }
            ],
            [[0.1] * 1024],
        )

        assert count == 1
        assert store.get_chunk("rt-agent-a7-feedfacecafebeef") is not None


def test_vector_upsert_skips_recursive_mcp_output_without_rolling_back_valid_siblings(tmp_path):
    with VectorStore(tmp_path / "upsert-mixed-guard.db") as store:
        count = store.upsert_chunks(
            [
                {
                    "id": "valid-sibling",
                    "content": "valid operational memory beside recursive output",
                    "metadata": {},
                    "source_file": "session.jsonl",
                    "project": "brainlayer",
                    "content_type": "assistant_text",
                    "char_count": 48,
                },
                {
                    "id": "recursive-sibling",
                    "content": JSONRPC_RECURSION_CONTENT,
                    "metadata": {},
                    "source_file": "session.jsonl",
                    "project": "brainlayer",
                    "content_type": "assistant_text",
                    "char_count": len(JSONRPC_RECURSION_CONTENT),
                },
            ],
            [[0.1] * 1024, [0.2] * 1024],
        )

        assert count == 1
        assert store.get_chunk("valid-sibling")["content"] == "valid operational memory beside recursive output"
        assert store.get_chunk("recursive-sibling") is None


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


def test_update_chunk_rejects_rt_agent_qid_judge_notes(tmp_path):
    with VectorStore(tmp_path / "update-rt-agent-guard.db") as store:
        store.upsert_chunks(
            [
                {
                    "id": "rt-agent-a7-update-target",
                    "content": "ordinary rt-agent memory before attempted judge-note update",
                    "metadata": {},
                    "source_file": (
                        "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-orchestrator/"
                        "session/subagents/agent-a7823570938b54ccd.jsonl"
                    ),
                    "project": "brainlayer",
                    "content_type": "assistant_text",
                    "char_count": 57,
                }
            ],
            [[0.2] * 1024],
        )

        with pytest.raises(ValueError, match="rt-agent judge notes"):
            store.update_chunk("rt-agent-a7-update-target", content=RT_AGENT_A7_JUDGE_NOTES_CONTENT)

        assert store.get_chunk("rt-agent-a7-update-target")["content"] == (
            "ordinary rt-agent memory before attempted judge-note update"
        )


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


def test_drain_drops_rt_agent_qid_judge_note_events(tmp_path, monkeypatch):
    db_path = tmp_path / "drain-rt-agent-guard.db"
    queue_dir = tmp_path / "queue"
    VectorStore(db_path).close()
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    enqueue_store(
        chunk_id="queued-store-rt-agent",
        content=RT_AGENT_A7_JUDGE_NOTES_CONTENT,
        project="brainlayer",
        tags=["correction:factual", "auto-detected"],
        queue_dir=queue_dir,
    )
    enqueue_watcher_chunk(
        chunk_id="rt-agent-a7-drainwatcher",
        content=RT_AGENT_A7_JUDGE_NOTES_CONTENT,
        metadata={},
        source_file=(
            "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-orchestrator/"
            "session/subagents/agent-a7823570938b54ccd.jsonl"
        ),
        project="brainlayer",
        content_type="assistant_text",
        value_type="HIGH",
        created_at="2026-05-16T12:00:00Z",
        conversation_id="session",
        tags=["correction:factual", "auto-detected"],
        queue_dir=queue_dir,
    )
    enqueue_hook_chunk(
        session_id="33abe108-session",
        content=RT_AGENT_A7_JUDGE_NOTES_CONTENT,
        project="brainlayer",
        queue_dir=queue_dir,
    )

    drained = drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10, log_path=tmp_path / "drain.log")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, content FROM chunks").fetchall()

    assert drained == 3
    assert rows == []
    assert not list(queue_dir.glob("*.jsonl"))


def test_drain_passes_rt_agent_context_to_judge_note_guard(tmp_path, monkeypatch):
    db_path = tmp_path / "drain-rt-agent-context-guard.db"
    queue_dir = tmp_path / "queue"
    VectorStore(db_path).close()
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    enqueue_store(
        chunk_id="rt-agent-a7-drainstore",
        content=RT_AGENT_CONTEXT_ONLY_JUDGE_NOTES_CONTENT,
        project="brainlayer",
        tags=["correction:factual", "auto-detected"],
        queue_dir=queue_dir,
    )
    enqueue_watcher_chunk(
        chunk_id="ordinary-drainwatcher-id",
        content=RT_AGENT_CONTEXT_ONLY_JUDGE_NOTES_CONTENT,
        metadata={},
        source_file=(
            "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-orchestrator/"
            "session/subagents/agent-a7823570938b54ccd.jsonl"
        ),
        project="brainlayer",
        content_type="assistant_text",
        value_type="HIGH",
        created_at="2026-05-16T12:00:00Z",
        conversation_id="session",
        tags=["correction:factual", "auto-detected"],
        queue_dir=queue_dir,
    )
    enqueue_hook_chunk(
        session_id="33abe108-session",
        chunk_id="rt-agent-a7-drainhook",
        content=RT_AGENT_CONTEXT_ONLY_JUDGE_NOTES_CONTENT,
        project="brainlayer",
        source_file=(
            "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-orchestrator/"
            "session/subagents/agent-a7823570938b54ccd.jsonl"
        ),
        queue_dir=queue_dir,
    )

    drained = drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10, log_path=tmp_path / "drain.log")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, content FROM chunks").fetchall()

    assert drained == 3
    assert rows == []
    assert not list(queue_dir.glob("*.jsonl"))
