"""WI-3 fallback replay attribution smoke tests.

These tests exercise the durable write paths directly. They intentionally do not
kill an MCP child process; transport-close is represented by a queued write file
that the replay/drain path must handle with its original attribution intact.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import apsw
import pytest

from brainlayer.chunk_origin import CHUNK_ORIGIN_AGENT_EXPLICIT, CHUNK_ORIGIN_USER_EXPLICIT


def _git_env() -> dict[str, str]:
    import os

    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git_init(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(("git", "init", "-q"), cwd=path, env=_git_env(), check=True)


def _write_markdown_fallback(repo: Path, *, chunk_origin: str, project: str = "systems") -> Path:
    path = repo / "docs.local" / "brain-store-fallback" / "pending.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "intended_brain_store: true\n"
        "memory_type: note\n"
        f"project: {project}\n"
        "importance: 8\n"
        "tags: [wi3, fallback]\n"
        "timestamp: 2026-06-14T06:00:00+00:00\n"
        "reason: transport_closed\n"
        "retry_attempted: true\n"
        f"chunk_origin: {chunk_origin}\n"
        "chunk_id:\n"
        "---\n"
        "WI3_MARKDOWN_REPLAY_TOKEN originating systems repo fallback body\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_db_busy_fallback_returns_deferred_receipt_and_durable_queue(tmp_path):
    """DB-busy is a real fallback condition: brain_store returns a loud queue receipt."""
    from brainlayer.mcp.store_handler import _store

    queue_dir = tmp_path / "queue"

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store", return_value=MagicMock()),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="systems"),
        patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("database is locked")),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        texts, structured = await _store(
            content="WI3_DB_BUSY_QUEUE_TOKEN systems fallback write",
            memory_type="note",
            project="systems",
            tags=["wi3", "db-busy"],
        )

    queue_files = list(queue_dir.glob("mcp-*.jsonl"))
    event = json.loads(queue_files[0].read_text(encoding="utf-8"))

    assert structured["status"] == "DEFERRED"
    assert structured["deferred"]["reason"] == "DB_BUSY"
    assert structured["deferred"]["action"] == "queued_for_drain"
    assert structured["chunk_id"] == event["chunk_id"]
    assert event["project"] == "systems"
    assert event["source"] == "mcp"
    assert any("queued" in text.text.lower() for text in texts)


def test_transport_closed_jsonl_replay_preserves_original_chunk_origin(tmp_path, monkeypatch):
    """A queued transport fallback must land with BEFORE chunk_origin preserved AFTER drain."""
    from brainlayer.drain import drain_once
    from brainlayer.queue_io import enqueue_store
    from brainlayer.vector_store import VectorStore

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    VectorStore(db_path).close()
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    queue_file = enqueue_store(
        chunk_id="manual-wi3-jsonl-origin",
        content="WI3_JSONL_REPLAY_TOKEN originating systems repo queued write",
        memory_type="note",
        project="systems",
        tags=["wi3", "transport-closed"],
        importance=7,
        chunk_origin=CHUNK_ORIGIN_USER_EXPLICIT,
        queue_dir=queue_dir,
    )
    before_event = json.loads(queue_file.read_text(encoding="utf-8"))

    drained = drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1, log_path=tmp_path / "drain.log")

    with VectorStore(db_path) as store:
        row = store.conn.cursor().execute(
            "SELECT project, source, source_file, chunk_origin FROM chunks WHERE id = ?",
            ("manual-wi3-jsonl-origin",),
        ).fetchone()

    assert before_event["project"] == "systems"
    assert before_event["chunk_origin"] == CHUNK_ORIGIN_USER_EXPLICIT
    assert drained == 1
    assert row == ("systems", "mcp", "brainlayer-queue", CHUNK_ORIGIN_USER_EXPLICIT)


def test_markdown_fallback_replay_preserves_origin_repo_and_chunk_origin(tmp_path):
    """Markdown fallback replay must not re-stamp originating repo attribution to unknown."""
    from brainlayer.fallback_replay import parse_fallback_file, replay_entry
    from brainlayer.store import store_memory
    from brainlayer.vector_store import VectorStore

    repo = tmp_path / "systems"
    _git_init(repo)
    fallback_path = _write_markdown_fallback(repo, chunk_origin=CHUNK_ORIGIN_AGENT_EXPLICIT)
    db_path = tmp_path / "brainlayer.db"

    entry = parse_fallback_file(fallback_path)
    with VectorStore(db_path) as store:
        result = replay_entry(
            entry,
            store_func=lambda **kwargs: store_memory(store=store, embed_fn=None, **kwargs),
            replayed_by="wi3-test",
        )
        row = store.conn.cursor().execute(
            "SELECT project, chunk_origin, metadata FROM chunks WHERE id = ?",
            (result.chunk_id,),
        ).fetchone()

    metadata = json.loads(row[2])
    assert result.error is None
    assert row[0] == "systems"
    assert row[1] == CHUNK_ORIGIN_AGENT_EXPLICIT
    assert metadata["origin_repo_path"] == str(repo.resolve())
    assert metadata["fallback_source_path"] == str(fallback_path)
    assert metadata["replayed_by"] == "wi3-test"


def test_legacy_pending_store_empty_chunk_id_lands_with_origin_and_is_queryable(tmp_path):
    """Legacy pending-stores lines with empty chunk_id still replay to a searchable row."""
    from brainlayer.mcp.store_handler import _flush_pending_stores
    from brainlayer.vector_store import VectorStore

    db_path = tmp_path / "brainlayer.db"
    pending_path = tmp_path / "pending-stores.jsonl"
    token = "WI3_EMPTY_CHUNK_ID_QUERYABLE_TOKEN"
    pending_path.write_text(
        json.dumps(
            {
                "chunk_id": "",
                "content": f"{token} originating systems repo legacy fallback",
                "memory_type": "note",
                "project": "systems",
                "tags": ["wi3", "empty-chunk-id"],
                "chunk_origin": CHUNK_ORIGIN_USER_EXPLICIT,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with VectorStore(db_path) as store:
        with patch("brainlayer.mcp.store_handler._get_pending_store_path", return_value=pending_path):
            flushed = _flush_pending_stores(store, None)
        row = store.conn.cursor().execute(
            "SELECT id, project, chunk_origin FROM chunks WHERE content LIKE ?",
            (f"%{token}%",),
        ).fetchone()
        results = store.search(query_text=token, n_results=5)

    assert flushed == 1
    assert row[0].startswith("manual-")
    assert row[1:] == ("systems", CHUNK_ORIGIN_USER_EXPLICIT)
    assert row[0] in results["ids"][0]


def test_metadata_only_queued_origin_is_preserved_for_old_fallback_events(tmp_path, monkeypatch):
    """Old queued files may only have metadata.chunk_origin; drain must still preserve it."""
    from brainlayer.drain import drain_once
    from brainlayer.vector_store import VectorStore

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    VectorStore(db_path).close()
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    (queue_dir / "mcp-metadata-origin.jsonl").write_text(
        json.dumps(
            {
                "kind": "store_memory",
                "chunk_id": "manual-wi3-metadata-origin",
                "content": "WI3_METADATA_ONLY_ORIGIN_TOKEN originating systems repo queued write",
                "memory_type": "note",
                "project": "systems",
                "metadata": {"chunk_origin": CHUNK_ORIGIN_USER_EXPLICIT},
                "source": "mcp",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    drained = drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1, log_path=tmp_path / "drain.log")

    with VectorStore(db_path) as store:
        row = store.conn.cursor().execute(
            "SELECT project, chunk_origin FROM chunks WHERE id = ?",
            ("manual-wi3-metadata-origin",),
        ).fetchone()

    assert drained == 1
    assert row == ("systems", CHUNK_ORIGIN_USER_EXPLICIT)


def test_deferred_mcp_row_stays_raw_etan_direct_with_high_enrichment_cap(tmp_path, monkeypatch):
    """Deferred MCP queue rows remain direct user-authored provenance after enrichment."""
    from brainlayer import enrichment_controller as controller
    from brainlayer.provenance_integration import resolve_entity_conflicts
    from brainlayer.vector_store import VectorStore

    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "100000")
    db_path = tmp_path / "brainlayer.db"
    chunk = {
        "id": "chunk-wi3-deferred-mcp-queue",
        "content": "WI3_PRIMARY_BACKEND: Groq",
        "metadata": {},
        "source_file": "brainlayer-queue",
        "project": "brainlayer",
        "content_type": "note",
        "value_type": "HIGH",
        "char_count": len("WI3_PRIMARY_BACKEND: Groq"),
        "source": "mcp",
        "sender": None,
        "created_at": "2026-06-14T00:00:00Z",
    }

    with VectorStore(db_path) as store:
        store.upsert_chunks([chunk], [[0.01] * 1024])
        controller._apply_enrichment(store, chunk, {"summary": "queued MCP store", "entities": []})
        store.upsert_entity("e-wi3-enrichment", "concept", "enrichment", canonical_name="enrichment")
        store.link_entity_chunk(
            "e-wi3-enrichment",
            chunk["id"],
            context="WI3_PRIMARY_BACKEND: Groq",
            mention_type="explicit",
        )

        row = store.conn.cursor().execute(
            "SELECT source, source_file, provenance_class FROM chunks WHERE id = ?",
            (chunk["id"],),
        ).fetchone()
        report = resolve_entity_conflicts(store, "enrichment", dry_run=True)
        results = store.search(query_text="WI3_PRIMARY_BACKEND", n_results=5, source_filter="mcp")

    authoritative = report.resolutions["WI3_PRIMARY_BACKEND"].authoritative
    assert row == ("mcp", "brainlayer-queue", "RAW-ETAN-DIRECT")
    assert authoritative is not None
    assert authoritative.id == chunk["id"]
    assert authoritative.provenance_class == "RAW-ETAN-DIRECT"
    assert authoritative.user_anchored is True
    assert results["ids"] == [[chunk["id"]]]
    assert results["metadatas"][0][0]["provenance_class"] == "RAW-ETAN-DIRECT"
