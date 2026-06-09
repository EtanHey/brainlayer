"""Fixture-only tests for provenance enrichment integration.

No live DB, no FTS triggers. These tests cover the mechanical column/payload
contract plus the v1 resolver consumer that reads kg_entity_chunks.
"""

from __future__ import annotations

import json
import sqlite3

import pytest


@pytest.fixture
def con(tmp_path):
    con = sqlite3.connect(tmp_path / "provenance_fixture.db")
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_type TEXT,
            sender TEXT,
            created_at TEXT,
            status TEXT DEFAULT 'active',
            superseded_by TEXT
        );
        CREATE TABLE kg_entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE kg_entity_chunks (
            entity_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            context TEXT,
            mention_type TEXT,
            PRIMARY KEY (entity_id, chunk_id)
        );
        """
    )
    con.execute("INSERT INTO kg_entities (id, name) VALUES ('e-control', 'controlLayer')")
    con.commit()
    return con


class Store:
    def __init__(self, conn):
        self.conn = conn
        self.update_calls = []

    def update_enrichment(self, **kwargs):
        self.update_calls.append(kwargs)


def test_ensure_provenance_class_column_adds_nullable_text(con):
    from brainlayer import enrichment_controller as controller

    assert "provenance_class" not in {row["name"] for row in con.execute("PRAGMA table_info(chunks)")}

    assert controller._ensure_provenance_class_column(Store(con)) is True

    cols = {row["name"]: row for row in con.execute("PRAGMA table_info(chunks)")}
    assert cols["provenance_class"]["type"] == "TEXT"
    assert cols["provenance_class"]["notnull"] == 0


def test_enrichment_update_payload_derives_provenance_class():
    from brainlayer import enrichment_controller as controller

    payload = controller._enrichment_update_payload(
        {
            "id": "chunk-1",
            "content": "nanoClaw is distinct from Hermes.",
            "content_type": "user_message",
            "sender": "user",
        },
        {"summary": "summary", "entities": []},
    )

    assert payload["provenance_class"] == "RAW-ETAN-DIRECT"


def test_direct_apply_enrichment_persists_provenance_class(con):
    from brainlayer import enrichment_controller as controller

    con.execute(
        "INSERT INTO chunks (id, content, content_type, sender, created_at) VALUES (?, ?, ?, ?, ?)",
        ("chunk-1", "Etan said: nanoClaw is distinct.", "assistant_text", "assistant", "2026-06-08T16:40:00Z"),
    )
    store = Store(con)

    controller._apply_enrichment(
        store,
        {
            "id": "chunk-1",
            "content": "Etan said: nanoClaw is distinct.",
            "content_type": "assistant_text",
            "sender": "assistant",
        },
        {"summary": "summary", "entities": []},
    )

    row = con.execute("SELECT provenance_class FROM chunks WHERE id = 'chunk-1'").fetchone()
    assert row["provenance_class"] == "AGENT-PARAPHRASE"


def test_queue_and_drain_preserve_provenance_class(tmp_path, con):
    from brainlayer.drain import _apply_enrichment
    from brainlayer.queue_io import enqueue_enrichment_updates

    con.execute(
        "INSERT INTO chunks (id, content, content_type, sender, created_at) VALUES (?, ?, ?, ?, ?)",
        ("chunk-1", "content", "assistant_text", "assistant", "2026-06-08T16:40:00Z"),
    )
    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    path = enqueue_enrichment_updates(
        [
            {
                "chunk_id": "chunk-1",
                "enrichment": {"summary": "summary"},
                "provenance_class": "AGENT-INFERENCE",
            }
        ],
        queue_dir=tmp_path,
    )
    event = json.loads(path.read_text(encoding="utf-8").strip())

    _apply_enrichment(con, event)

    row = con.execute("SELECT provenance_class FROM chunks WHERE id = 'chunk-1'").fetchone()
    assert event["provenance_class"] == "AGENT-INFERENCE"
    assert row["provenance_class"] == "AGENT-INFERENCE"


def test_resolve_entity_conflicts_defaults_to_dry_run(con):
    from brainlayer.provenance_integration import resolve_entity_conflicts

    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    con.executemany(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "c-direct",
                "DEFINITION: FILE_SCHEMA_AND_POLICIES",
                "user_message",
                "user",
                "2026-06-05T22:32:03Z",
                "RAW-ETAN-DIRECT",
            ),
            (
                "c-para",
                "DEFINITION: NEW_RUNNING_SERVICE",
                "assistant_text",
                "assistant",
                "2026-06-08T10:00:00Z",
                "AGENT-PARAPHRASE",
            ),
            (
                "c-infer",
                "ARBITRATION: CONTROLLAYER_DECIDES",
                "assistant_text",
                "assistant",
                "2026-06-05T21:47:27Z",
                "AGENT-INFERENCE",
            ),
        ],
    )
    con.executemany(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES ('e-control', ?)",
        [("c-direct",), ("c-para",), ("c-infer",)],
    )

    report = resolve_entity_conflicts(con, "controlLayer")

    assert report.dry_run is True
    assert report.resolutions["DEFINITION"].authoritative.id == "c-direct"
    assert report.resolutions["ARBITRATION"].disposition == "PENDING-USER-CONFIRM"
    assert report.superseded_count == 0
    assert report.pending_confirm_count == 0
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-para'").fetchone()) == (
        "active",
        None,
    )
    assert not con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='provenance_pending_user_confirm'"
    ).fetchone()


def test_resolve_entity_conflicts_write_path_uses_fixture_db(con):
    from brainlayer.provenance_integration import resolve_entity_conflicts

    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    con.executemany(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "c-direct",
                "DEFINITION: FILE_SCHEMA_AND_POLICIES",
                "user_message",
                "user",
                "2026-06-05T22:32:03Z",
                "RAW-ETAN-DIRECT",
            ),
            (
                "c-para",
                "DEFINITION: NEW_RUNNING_SERVICE",
                "assistant_text",
                "assistant",
                "2026-06-08T10:00:00Z",
                "AGENT-PARAPHRASE",
            ),
            (
                "c-infer",
                "ARBITRATION: CONTROLLAYER_DECIDES",
                "assistant_text",
                "assistant",
                "2026-06-05T21:47:27Z",
                "AGENT-INFERENCE",
            ),
        ],
    )
    con.executemany(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES ('e-control', ?)",
        [("c-direct",), ("c-para",), ("c-infer",)],
    )

    report = resolve_entity_conflicts(con, "controlLayer", dry_run=False)

    assert report.dry_run is False
    assert report.superseded_count == 1
    assert report.pending_confirm_count == 1
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-para'").fetchone()) == (
        "superseded",
        "c-direct",
    )
    pending = con.execute(
        """
        SELECT entity, attribute, chunk_id, value, provenance_class
        FROM provenance_pending_user_confirm
        """
    ).fetchone()
    assert tuple(pending) == (
        "controlLayer",
        "ARBITRATION",
        "c-infer",
        "CONTROLLAYER_DECIDES",
        "AGENT-INFERENCE",
    )


def test_two_unanchored_inferences_that_contradict_each_other_both_pending(con):
    from brainlayer.provenance_integration import resolve_entity_conflicts

    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    con.executemany(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "c-infer-a",
                "ARBITRATION: CONTROLLAYER_DECIDES",
                "assistant_text",
                "assistant",
                "2026-06-05T21:47:27Z",
                "AGENT-INFERENCE",
            ),
            (
                "c-infer-b",
                "ARBITRATION: VOICEBAR_DAEMON_ENFORCES",
                "assistant_text",
                "assistant",
                "2026-06-05T21:48:27Z",
                "AGENT-INFERENCE",
            ),
        ],
    )
    con.executemany(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES ('e-control', ?)",
        [("c-infer-a",), ("c-infer-b",)],
    )

    report = resolve_entity_conflicts(con, "controlLayer", dry_run=False)

    assert report.resolutions["ARBITRATION"].disposition == "PENDING-USER-CONFIRM"
    assert report.resolutions["ARBITRATION"].authoritative is None
    pending = con.execute("SELECT chunk_id FROM provenance_pending_user_confirm ORDER BY chunk_id").fetchall()
    assert [row["chunk_id"] for row in pending] == ["c-infer-a", "c-infer-b"]


def test_pending_confirm_enqueue_is_idempotent_for_same_inference(con):
    from brainlayer.provenance_integration import resolve_entity_conflicts

    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    con.execute(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "c-infer",
            "ARBITRATION: CONTROLLAYER_DECIDES",
            "assistant_text",
            "assistant",
            "2026-06-05T21:47:27Z",
            "AGENT-INFERENCE",
        ),
    )
    con.execute("INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES ('e-control', 'c-infer')")

    first = resolve_entity_conflicts(con, "controlLayer", dry_run=False)
    second = resolve_entity_conflicts(con, "controlLayer", dry_run=False)

    assert first.pending_confirm_count == 1
    assert second.pending_confirm_count == 0
    assert con.execute("SELECT COUNT(*) FROM provenance_pending_user_confirm").fetchone()[0] == 1


def test_confirm_pending_promotes_inference_and_resolves_attribute(con):
    from brainlayer.provenance_integration import confirm_pending, resolve_entity_conflicts

    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    con.executemany(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "c-para",
                "ARBITRATION: OLD_AGENT_FRAME",
                "assistant_text",
                "assistant",
                "2026-06-05T21:40:27Z",
                "AGENT-PARAPHRASE",
            ),
            (
                "c-infer",
                "ARBITRATION: CONFIRMED_SYSTEM_STATE",
                "assistant_text",
                "assistant",
                "2026-06-05T21:47:27Z",
                "AGENT-INFERENCE",
            ),
        ],
    )
    con.executemany(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES ('e-control', ?)",
        [("c-para",), ("c-infer",)],
    )
    resolve_entity_conflicts(con, "controlLayer", dry_run=False)

    report = confirm_pending(con, "c-infer")

    assert report.resolutions["ARBITRATION"].authoritative.id == "c-infer"
    assert con.execute("SELECT provenance_class FROM chunks WHERE id = 'c-infer'").fetchone()[0] == "RAW-ETAN-DIRECT"
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-para'").fetchone()) == (
        "superseded",
        "c-infer",
    )
    assert con.execute("SELECT COUNT(*) FROM provenance_pending_user_confirm").fetchone()[0] == 0
