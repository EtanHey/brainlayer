from __future__ import annotations

import sqlite3

import pytest

from brainlayer.vector_store import VectorStore
from brainlayer.watcher_bridge import create_flush_callback


def _control(source_file: str, role: str, digest: str = "a") -> dict:
    return {
        "type": "system",
        "sessionId": "native-session",
        "uuid": f"native-{role}",
        "parentUuid": None,
        "attributionAgent": role,
        "_provider": "claude",
        "_source_file": source_file,
        "_line_start_offset": 0,
        "_line_end_offset": 100,
        "_raw_record_hash": digest * 64,
    }


def _native_table_count(db_path) -> int:
    with sqlite3.connect(db_path) as conn:
        exists = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'native_record_links'").fetchone()[0]
        if not exists:
            return 0
        return conn.execute("SELECT COUNT(*) FROM native_record_links").fetchone()[0]


@pytest.mark.parametrize("arbitrated", [False, True])
def test_native_memory_reader_labels_are_rejected_at_direct_and_queued_flush_boundary(
    tmp_path, monkeypatch, arbitrated
):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    db_path = tmp_path / f"denied-{arbitrated}.db"
    queue_dir = tmp_path / f"queue-{arbitrated}"
    VectorStore(db_path).close()
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    source_file = str(tmp_path / "ordinary-provider-session.jsonl")
    entries = [
        _control(source_file, role, digest)
        for role, digest in zip(("brain-worker", "session-miner", "weave"), ("a", "b", "c"))
    ]

    result = create_flush_callback(db_path, arbitrated=arbitrated)(entries)

    assert result[source_file] == 100
    assert _native_table_count(db_path) == 0
    assert not list(queue_dir.glob("watcher-native-record-*.jsonl"))


@pytest.mark.parametrize("arbitrated", [False, True])
def test_ordinary_native_agent_is_positive_control_for_both_flush_modes(tmp_path, monkeypatch, arbitrated):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    db_path = tmp_path / f"ordinary-{arbitrated}.db"
    queue_dir = tmp_path / f"queue-{arbitrated}"
    VectorStore(db_path).close()
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))

    create_flush_callback(db_path, arbitrated=arbitrated)(
        [_control(str(tmp_path / "ordinary.jsonl"), "workflow-subagent")]
    )

    if arbitrated:
        assert len(list(queue_dir.glob("watcher-native-record-*.jsonl"))) == 1
    else:
        assert _native_table_count(db_path) == 1


def test_explicit_empty_denylist_override_preserves_class_rule_replacement(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_INGEST_DENYLIST", "")
    db_path = tmp_path / "override.db"
    VectorStore(db_path).close()

    create_flush_callback(db_path, arbitrated=False)([_control(str(tmp_path / "override.jsonl"), "brain-worker")])

    assert _native_table_count(db_path) == 1


def test_explicit_denylist_pattern_still_rejects_ordinary_native_record(tmp_path, monkeypatch):
    source_file = str(tmp_path / "pattern-denied.jsonl")
    monkeypatch.setenv("BRAINLAYER_INGEST_DENYLIST", source_file)
    db_path = tmp_path / "pattern.db"
    VectorStore(db_path).close()

    create_flush_callback(db_path, arbitrated=False)([_control(source_file, "workflow-subagent")])

    assert _native_table_count(db_path) == 0
