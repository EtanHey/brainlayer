"""Regression coverage for the published-intent queue fallback boundary.

These tests use only private scratch queue paths and synthetic store content.
"""

import asyncio
import json
import sqlite3

import pytest

from brainlayer import queue_io
from brainlayer.mcp import store_handler


def _use_private_queues(monkeypatch, unified_dir, legacy_path):
    monkeypatch.setattr(queue_io, "get_queue_dir", lambda: unified_dir)
    monkeypatch.setattr(store_handler, "_get_pending_store_path", lambda: legacy_path)


def test_post_publication_directory_sync_failure_does_not_create_legacy_copy(tmp_path, monkeypatch):
    unified_dir = tmp_path / "unified"
    unified_dir.mkdir()
    legacy_path = tmp_path / "pending-stores.jsonl"
    _use_private_queues(monkeypatch, unified_dir, legacy_path)

    real_sync_directory = queue_io._sync_directory

    def fail_after_jsonl_is_visible(directory):
        if directory == unified_dir and list(unified_dir.glob("*.jsonl")):
            raise OSError("synthetic post-publication directory sync failure")
        return real_sync_directory(directory)

    monkeypatch.setattr(queue_io, "_sync_directory", fail_after_jsonl_is_visible)

    with pytest.raises(OSError, match="synthetic post-publication directory sync failure"):
        store_handler._queue_store({"content": "synthetic published intent", "memory_type": "note"})

    published = list(unified_dir.glob("*.jsonl"))
    assert len(published) == 1
    event = json.loads(published[0].read_text(encoding="utf-8"))
    assert event["content"] == "synthetic published intent"
    assert event["kind"] == "store_memory"
    assert not legacy_path.exists()


def test_durable_unified_enqueue_returns_its_intent_without_legacy_write(tmp_path, monkeypatch):
    unified_dir = tmp_path / "unified"
    unified_dir.mkdir()
    legacy_path = tmp_path / "pending-stores.jsonl"
    _use_private_queues(monkeypatch, unified_dir, legacy_path)

    result = store_handler._queue_store({"content": "synthetic durable intent", "memory_type": "note"})

    assert result.parent == unified_dir
    assert result.is_file()
    event = json.loads(result.read_text(encoding="utf-8"))
    assert event["content"] == "synthetic durable intent"
    assert event["kind"] == "store_memory"
    assert event["source"] == "mcp"
    assert not legacy_path.exists()
    assert not list(unified_dir.glob("*.tmp"))


def test_prepublication_unified_failure_uses_one_legacy_fallback(tmp_path, monkeypatch):
    unified_dir = tmp_path / "unified"
    unified_dir.mkdir()
    legacy_path = tmp_path / "pending-stores.jsonl"
    _use_private_queues(monkeypatch, unified_dir, legacy_path)

    def fail_before_file_publication(path, text):
        raise OSError("synthetic pre-publication file sync failure")

    monkeypatch.setattr(queue_io, "_write_synced_queue_file", fail_before_file_publication)

    result = store_handler._queue_store({"content": "synthetic legacy intent", "memory_type": "note"})

    assert result == legacy_path
    assert not list(unified_dir.glob("*.jsonl"))
    assert not list(unified_dir.glob("*.tmp"))
    events = [json.loads(line) for line in legacy_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert events[0]["content"] == "synthetic legacy intent"


@pytest.mark.parametrize("message", ["directory sync failed", "directory busy after publication"])
@pytest.mark.parametrize("consumed", [False, True])
def test_store_does_not_acknowledge_or_retry_an_unconfirmed_publication(tmp_path, monkeypatch, message, consumed):
    unified_dir = tmp_path / "unified"
    unified_dir.mkdir()
    legacy_path = tmp_path / "pending-stores.jsonl"
    _use_private_queues(monkeypatch, unified_dir, legacy_path)
    monkeypatch.setattr(store_handler, "_interactive_queue_reason", lambda: "DB_BUSY")
    real_sync = queue_io._sync_directory
    attempts = []
    published_paths = []
    database = tmp_path / "private-drainer.db"

    def fail_published(directory):
        if directory == unified_dir and list(unified_dir.glob("*.jsonl")):
            attempts.append(directory)
            published_path = next(unified_dir.glob("*.jsonl"))
            published_paths.append(published_path)
            if consumed:
                # Simulate the drainer winning the race, with a real private
                # SQLite commit before the producer sees its failed barrier.
                event = json.loads(published_path.read_text())
                with sqlite3.connect(database) as connection:
                    connection.execute("CREATE TABLE committed_intents (chunk_id TEXT PRIMARY KEY)")
                    connection.execute("INSERT INTO committed_intents VALUES (?)", (event["chunk_id"],))
                published_path.unlink()
            raise OSError(message)
        return real_sync(directory)

    monkeypatch.setattr(queue_io, "_sync_directory", fail_published)
    with pytest.raises(OSError, match=message) as failure:
        asyncio.run(store_handler._store(content="Synthetic unconfirmed store intent", memory_type="note"))

    published = list(unified_dir.glob("*.jsonl"))
    assert len(published) == (0 if consumed else 1)
    assert failure.value.published_path == published_paths[0]
    if consumed:
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM committed_intents").fetchone()[0] == 1
    assert "do not retry" in str(failure.value)
    assert attempts == [unified_dir]
    assert not legacy_path.exists()
