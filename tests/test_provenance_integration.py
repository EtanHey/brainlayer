"""Fixture-only tests for provenance enrichment integration.

No live DB, no FTS triggers. These tests cover the mechanical column/payload
contract plus the v1 resolver consumer that reads kg_entity_chunks.
"""

from __future__ import annotations

import json
import logging
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


def test_get_chunk_readonly_hydrates_prev_assistant_for_endorsement_classification():
    from brainlayer import enrichment_controller as controller

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT,
            source_file TEXT,
            project TEXT,
            content_type TEXT,
            sender TEXT,
            value_type TEXT,
            tags TEXT,
            importance INTEGER,
            created_at TEXT,
            summary TEXT,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived_at TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                "assistant-prior",
                "controlLayer is an umbrella for the file schema and policies.",
                "assistant_text",
                "assistant",
                "2026-06-12T10:00:00Z",
            ),
            (
                "user-ack",
                "yes exactly, controlLayer is the file schema and policies",
                "user_message",
                "user",
                "2026-06-12T10:01:00Z",
            ),
        ],
    )

    class ReadonlyStore:
        def _read_cursor(self):
            return conn.cursor()

    chunk = controller._get_chunk_readonly(ReadonlyStore(), "user-ack")
    assert chunk is not None
    payload = controller._enrichment_update_payload(chunk, {"summary": "summary", "entities": []})

    assert chunk["prev_assistant_text"] == "controlLayer is an umbrella for the file schema and policies."
    assert payload["provenance_class"] == "ETAN-ENDORSEMENT"


def test_get_chunk_readonly_fallback_includes_prev_assistant_key():
    from brainlayer import enrichment_controller as controller

    class StoreWithoutReadCursor:
        def get_chunk(self, chunk_id):
            assert chunk_id == "user-ack"
            return {
                "id": "user-ack",
                "content": "yes exactly",
                "content_type": "user_message",
                "sender": "user",
            }

    chunk = controller._get_chunk_readonly(StoreWithoutReadCursor(), "user-ack")

    assert chunk is not None
    assert chunk["prev_assistant_text"] is None


def test_get_chunk_readonly_scopes_prev_assistant_to_same_conversation():
    from brainlayer import enrichment_controller as controller

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT,
            source_file TEXT,
            project TEXT,
            content_type TEXT,
            sender TEXT,
            value_type TEXT,
            tags TEXT,
            importance INTEGER,
            created_at TEXT,
            summary TEXT,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived_at TEXT,
            conversation_id TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO chunks (id, content, source_file, project, content_type, sender, created_at, conversation_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "other-assistant",
                "controlLayer is an umbrella for the file schema and policies.",
                "other-session.jsonl",
                "brainlayer",
                "assistant_text",
                "assistant",
                "2026-06-12T10:00:45Z",
                "other-session",
            ),
            (
                "same-assistant",
                "Hermes owns the fleet worker heartbeat.",
                "same-session.jsonl",
                "brainlayer",
                "assistant_text",
                "assistant",
                "2026-06-12T10:00:00Z",
                "same-session",
            ),
            (
                "user-ack",
                "yes exactly, controlLayer is the file schema and policies",
                "same-session.jsonl",
                "brainlayer",
                "user_message",
                "user",
                "2026-06-12T10:01:00Z",
                "same-session",
            ),
        ],
    )

    class ReadonlyStore:
        def _read_cursor(self):
            return conn.cursor()

    chunk = controller._get_chunk_readonly(ReadonlyStore(), "user-ack")
    assert chunk is not None
    payload = controller._enrichment_update_payload(chunk, {"summary": "summary", "entities": []})

    assert chunk["prev_assistant_text"] == "Hermes owns the fleet worker heartbeat."
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


def test_apply_enrichment_enqueues_mentioned_entities_for_provenance_sweep(con):
    from brainlayer import enrichment_controller as controller

    con.execute(
        "INSERT INTO chunks (id, content, content_type, sender, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            "chunk-1",
            "PRIMARY_BACKEND: Gemini",
            "assistant_text",
            "assistant",
            "2026-06-09T10:00:00Z",
        ),
    )

    controller._apply_enrichment(
        Store(con),
        {
            "id": "chunk-1",
            "content": "PRIMARY_BACKEND: Gemini",
            "content_type": "assistant_text",
            "sender": "assistant",
        },
        {"summary": "summary", "entities": [{"name": "enrichment"}, {"text": "orc"}]},
    )

    rows = con.execute("SELECT entity, chunk_id FROM provenance_resolve_queue ORDER BY entity").fetchall()
    assert [tuple(row) for row in rows] == [("enrichment", "chunk-1"), ("orc", "chunk-1")]


def test_apply_enrichment_auto_supersede_flag_defaults_off(con, monkeypatch):
    from brainlayer import enrichment_controller as controller

    monkeypatch.delenv("BRAINLAYER_AUTO_SUPERSEDE", raising=False)
    monkeypatch.setattr(
        controller,
        "auto_supersede",
        lambda *args, **kwargs: pytest.fail("auto_supersede should not run when flag is unset"),
        raising=False,
    )
    con.execute(
        "INSERT INTO chunks (id, content, content_type, sender, created_at) VALUES (?, ?, ?, ?, ?)",
        ("chunk-1", "PRIMARY_BACKEND: Gemini", "user_message", "user", "2026-06-09T10:00:00Z"),
    )

    controller._apply_enrichment(
        Store(con),
        {
            "id": "chunk-1",
            "content": "PRIMARY_BACKEND: Gemini",
            "content_type": "user_message",
            "sender": "user",
        },
        {"summary": "summary", "entities": [{"name": "provider router"}]},
    )

    assert con.execute("SELECT superseded_by FROM chunks WHERE id = 'chunk-1'").fetchone()["superseded_by"] is None


def test_apply_enrichment_auto_supersede_flag_on_runs_dry_run_only(con, monkeypatch, caplog):
    from brainlayer import enrichment_controller as controller

    monkeypatch.setenv("BRAINLAYER_AUTO_SUPERSEDE", "1")
    caplog.set_level(logging.INFO, logger="brainlayer.enrichment_controller")
    con.execute("INSERT INTO kg_entities (id, name) VALUES ('e-provider', 'provider router')")
    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    con.executemany(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "c-groq",
                "PRIMARY_BACKEND: Groq",
                "assistant_text",
                "assistant",
                "2026-03-26T00:00:00Z",
                "AGENT-PARAPHRASE",
            ),
            (
                "c-gemini",
                "PRIMARY_BACKEND: Gemini",
                "user_message",
                "user",
                "2026-06-09T10:00:00Z",
                None,
            ),
        ],
    )
    con.execute("INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES ('e-provider', 'c-groq')")

    controller._apply_enrichment(
        Store(con),
        {
            "id": "c-gemini",
            "content": "PRIMARY_BACKEND: Gemini",
            "content_type": "user_message",
            "sender": "user",
            "created_at": "2026-06-09T10:00:00Z",
        },
        {"summary": "summary", "entities": [{"name": "provider router"}]},
    )

    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-groq'").fetchone()) == (
        "active",
        None,
    )
    assert not con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='provenance_pending_user_confirm'"
    ).fetchone()
    assert "auto_supersede dry_run entity=provider router" in caplog.text
    assert "would_supersede=1" in caplog.text


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
                "entities": [{"name": "controlLayer"}],
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
    queued = con.execute("SELECT entity, chunk_id, reason FROM provenance_resolve_queue").fetchone()
    assert dict(queued) == {"entity": "controlLayer", "chunk_id": "chunk-1", "reason": "enrichment"}


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
    assert con.execute("SELECT provenance_class FROM chunks WHERE id = 'c-infer'").fetchone()[0] == "AGENT-INFERENCE"
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-para'").fetchone()) == (
        "superseded",
        "c-infer",
    )
    assert con.execute("SELECT COUNT(*) FROM provenance_pending_user_confirm").fetchone()[0] == 0


def test_confirm_pending_scopes_confirmation_to_single_claim(con):
    from brainlayer.provenance_integration import confirm_pending, list_pending_confirm, resolve_entity_conflicts

    con.execute("INSERT INTO kg_entities (id, name) VALUES ('e-enrichment', 'enrichment')")
    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    con.executemany(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "c-control-old",
                "ARBITRATION: OLD_AGENT_FRAME",
                "assistant_text",
                "assistant",
                "2026-06-05T21:40:27Z",
                "AGENT-PARAPHRASE",
            ),
            (
                "c-user-backend",
                "PRIMARY_BACKEND: Gemini",
                "user_message",
                "user",
                "2026-06-05T21:41:27Z",
                "RAW-ETAN-DIRECT",
            ),
            (
                "c-shared-infer",
                "inferred system state",
                "assistant_text",
                "assistant",
                "2026-06-05T21:47:27Z",
                "AGENT-INFERENCE",
            ),
        ],
    )
    con.executemany(
        """
        INSERT INTO kg_entity_chunks (entity_id, chunk_id, context)
        VALUES (?, ?, ?)
        """,
        [
            ("e-control", "c-control-old", json.dumps({"attribute": "ARBITRATION", "value": "OLD_AGENT_FRAME"})),
            (
                "e-control",
                "c-shared-infer",
                json.dumps({"attribute": "ARBITRATION", "value": "CONFIRMED_SYSTEM_STATE"}),
            ),
            ("e-enrichment", "c-user-backend", json.dumps({"attribute": "PRIMARY_BACKEND", "value": "Gemini"})),
            ("e-enrichment", "c-shared-infer", json.dumps({"attribute": "PRIMARY_BACKEND", "value": "Groq"})),
        ],
    )

    resolve_entity_conflicts(con, "controlLayer", dry_run=False)
    pending = list_pending_confirm(con)
    control_pending = next(item for item in pending if item["entity"] == "controlLayer")

    control_report = confirm_pending(con, control_pending["id"])
    enrichment_report = resolve_entity_conflicts(con, "enrichment", dry_run=False)

    assert control_report.resolutions["ARBITRATION"].authoritative.id == "c-shared-infer"
    assert (
        con.execute("SELECT provenance_class FROM chunks WHERE id = 'c-shared-infer'").fetchone()[0]
        == "AGENT-INFERENCE"
    )
    assert enrichment_report.resolutions["PRIMARY_BACKEND"].authoritative.id == "c-user-backend"
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-user-backend'").fetchone()) == (
        "active",
        None,
    )
    assert (
        con.execute(
            """
        SELECT COUNT(*)
        FROM provenance_confirmed_claims
        WHERE entity = 'controlLayer'
          AND attribute = 'ARBITRATION'
          AND chunk_id = 'c-shared-infer'
          AND provenance_class = 'RAW-ETAN-DIRECT'
        """
        ).fetchone()[0]
        == 1
    )


def test_confirm_pending_rejects_ambiguous_chunk_id(con):
    from brainlayer.provenance_integration import _ensure_pending_user_confirm_table, confirm_pending

    _ensure_pending_user_confirm_table(con)
    con.executemany(
        """
        INSERT INTO provenance_pending_user_confirm
        (id, entity, attribute, chunk_id, value, provenance_class, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "pending-arbitration",
                "controlLayer",
                "ARBITRATION",
                "c-shared-infer",
                "CONFIRMED_SYSTEM_STATE",
                "AGENT-INFERENCE",
                "needs-direct-confirmation",
                "2026-06-05T21:47:27Z",
            ),
            (
                "pending-backend",
                "enrichment",
                "PRIMARY_BACKEND",
                "c-shared-infer",
                "Groq",
                "AGENT-INFERENCE",
                "needs-direct-confirmation",
                "2026-06-05T21:48:27Z",
            ),
        ],
    )

    report = confirm_pending(con, "c-shared-infer")

    assert report.notes == ["Ambiguous pending provenance confirmation matched claim_id=c-shared-infer"]
    assert con.execute("SELECT COUNT(*) FROM provenance_pending_user_confirm").fetchone()[0] == 2


def test_reject_pending_rejects_ambiguous_chunk_id_without_archiving(con):
    from brainlayer.provenance_integration import _ensure_pending_user_confirm_table, reject_pending

    _ensure_pending_user_confirm_table(con)
    con.execute(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("c-shared-infer", "inferred system state", "assistant_text", "assistant", "2026-06-05T21:47:27Z"),
    )
    con.executemany(
        """
        INSERT INTO provenance_pending_user_confirm
        (id, entity, attribute, chunk_id, value, provenance_class, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "pending-arbitration",
                "controlLayer",
                "ARBITRATION",
                "c-shared-infer",
                "CONFIRMED_SYSTEM_STATE",
                "AGENT-INFERENCE",
                "needs-direct-confirmation",
                "2026-06-05T21:47:27Z",
            ),
            (
                "pending-backend",
                "enrichment",
                "PRIMARY_BACKEND",
                "c-shared-infer",
                "Groq",
                "AGENT-INFERENCE",
                "needs-direct-confirmation",
                "2026-06-05T21:48:27Z",
            ),
        ],
    )

    assert reject_pending(con, "c-shared-infer") is False
    assert con.execute("SELECT status FROM chunks WHERE id = 'c-shared-infer'").fetchone()[0] == "active"
    assert con.execute("SELECT COUNT(*) FROM provenance_pending_user_confirm").fetchone()[0] == 2


def test_provenance_sweep_drains_queue_and_supersedes_stale_lower_class(con):
    from brainlayer.provenance_integration import enqueue_provenance_resolution, sweep_provenance_queue

    con.execute("INSERT INTO kg_entities (id, name) VALUES ('e-enrichment', 'enrichment')")
    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    con.executemany(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "c-groq",
                "PRIMARY_BACKEND: Groq",
                "assistant_text",
                "assistant",
                "2026-03-26T00:00:00Z",
                "AGENT-PARAPHRASE",
            ),
            (
                "c-gemini",
                "PRIMARY_BACKEND: Gemini",
                "user_message",
                "user",
                "2026-06-09T00:00:00Z",
                "RAW-ETAN-DIRECT",
            ),
        ],
    )
    con.executemany(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES ('e-enrichment', ?)",
        [("c-groq",), ("c-gemini",)],
    )
    enqueue_provenance_resolution(con, "enrichment", chunk_id="c-gemini")

    result = sweep_provenance_queue(con)

    assert result.swept == 1
    assert result.superseded_count == 1
    assert con.execute("SELECT COUNT(*) FROM provenance_resolve_queue").fetchone()[0] == 0
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-groq'").fetchone()) == (
        "superseded",
        "c-gemini",
    )


def test_provenance_sweep_keeps_unresolved_entity_queued(con):
    from brainlayer.provenance_integration import enqueue_provenance_resolution, sweep_provenance_queue

    enqueue_provenance_resolution(con, "futureEntity", chunk_id="c-future")

    result = sweep_provenance_queue(con)

    assert result.swept == 1
    assert result.entities == ["futureEntity"]
    assert "No kg_entities row matched entity id/name/alias" in result.notes
    assert tuple(
        con.execute("SELECT entity, chunk_id FROM provenance_resolve_queue WHERE entity = 'futureEntity'").fetchone()
    ) == ("futureEntity", "c-future")


def test_provenance_sweep_bumps_unresolved_entity_retry_metadata(con):
    from brainlayer.provenance_integration import enqueue_provenance_resolution, sweep_provenance_queue

    enqueue_provenance_resolution(con, "futureEntity", chunk_id="c-future")
    before = con.execute(
        "SELECT attempts, updated_at FROM provenance_resolve_queue WHERE entity = 'futureEntity'"
    ).fetchone()

    sweep_provenance_queue(con)

    after = con.execute(
        "SELECT attempts, updated_at FROM provenance_resolve_queue WHERE entity = 'futureEntity'"
    ).fetchone()
    assert after["attempts"] == before["attempts"] + 1
    assert after["updated_at"] > before["updated_at"]


def test_provenance_sweep_does_not_auto_supersede_personal_data(con):
    from brainlayer.provenance_integration import enqueue_provenance_resolution, sweep_provenance_queue

    con.execute("INSERT INTO kg_entities (id, name) VALUES ('e-etan', 'Etan')")
    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    con.executemany(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "c-health-old",
                "HEALTH_STATUS: recovering",
                "journal",
                "user",
                "2026-06-01T00:00:00Z",
                "RAW-ETAN-DIRECT",
            ),
            (
                "c-health-new",
                "HEALTH_STATUS: recovered",
                "journal",
                "user",
                "2026-06-09T00:00:00Z",
                "RAW-ETAN-DIRECT",
            ),
        ],
    )
    con.executemany(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES ('e-etan', ?)",
        [("c-health-old",), ("c-health-new",)],
    )
    enqueue_provenance_resolution(con, "Etan", chunk_id="c-health-new")

    result = sweep_provenance_queue(con)

    assert result.swept == 1
    assert result.superseded_count == 0
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-health-old'").fetchone()) == (
        "active",
        None,
    )


def test_provenance_sweep_does_not_auto_supersede_unstructured_tag_mentions(con):
    from brainlayer.provenance_integration import enqueue_provenance_resolution, sweep_provenance_queue

    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    con.executemany(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "c-tag-old",
                "Legacy BrainLayer enrichment note",
                "assistant_text",
                "assistant",
                "2026-06-01T00:00:00Z",
                "AGENT-PARAPHRASE",
            ),
            (
                "c-tag-new",
                "Current BrainLayer enrichment note",
                "user_message",
                "user",
                "2026-06-09T00:00:00Z",
                "RAW-ETAN-DIRECT",
            ),
        ],
    )
    con.executemany(
        """
        INSERT INTO kg_entity_chunks (entity_id, chunk_id, mention_type)
        VALUES ('e-control', ?, 'tag')
        """,
        [("c-tag-old",), ("c-tag-new",)],
    )
    enqueue_provenance_resolution(con, "controlLayer", chunk_id="c-tag-new")

    result = sweep_provenance_queue(con)

    assert result.swept == 1
    assert result.superseded_count == 0
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-tag-old'").fetchone()) == (
        "active",
        None,
    )


def test_pending_reject_archives_inference_and_removes_queue_row(con):
    from brainlayer.provenance_integration import list_pending_confirm, reject_pending, resolve_entity_conflicts

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
    resolve_entity_conflicts(con, "controlLayer", dry_run=False)

    pending = list_pending_confirm(con)
    assert len(pending) == 1
    assert pending[0]["chunk_id"] == "c-infer"

    assert reject_pending(con, "c-infer") is True

    assert list_pending_confirm(con) == []
    assert con.execute("SELECT status FROM chunks WHERE id = 'c-infer'").fetchone()[0] == "archived"


def test_entity_authority_annotations_show_authoritative_and_superseded_values(con):
    from brainlayer.provenance_integration import (
        enqueue_provenance_resolution,
        get_entity_provenance_annotations,
        sweep_provenance_queue,
    )

    con.execute("INSERT INTO kg_entities (id, name) VALUES ('e-enrichment', 'enrichment')")
    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    con.executemany(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "c-groq",
                "PRIMARY_BACKEND: Groq",
                "assistant_text",
                "assistant",
                "2026-03-26T00:00:00Z",
                "AGENT-PARAPHRASE",
            ),
            (
                "c-gemini",
                "PRIMARY_BACKEND: Gemini",
                "user_message",
                "user",
                "2026-06-09T00:00:00Z",
                "RAW-ETAN-DIRECT",
            ),
        ],
    )
    con.executemany(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES ('e-enrichment', ?)",
        [("c-groq",), ("c-gemini",)],
    )
    enqueue_provenance_resolution(con, "enrichment", chunk_id="c-gemini")
    sweep_provenance_queue(con)

    annotations = get_entity_provenance_annotations(con, "enrichment")

    assert annotations["PRIMARY_BACKEND"]["authoritative"]["value"] == "Gemini"
    assert annotations["PRIMARY_BACKEND"]["authoritative"]["provenance_class"] == "RAW-ETAN-DIRECT"
    assert annotations["PRIMARY_BACKEND"]["superseded"][0]["value"] == "Groq"


def test_generic_note_factual_chunk_can_be_superseded(con):
    from brainlayer.provenance_integration import enqueue_provenance_resolution, sweep_provenance_queue

    con.execute("INSERT INTO kg_entities (id, name) VALUES ('e-enrichment', 'enrichment')")
    con.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    con.executemany(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "c-note-old",
                "PRIMARY_BACKEND: Groq",
                "note",
                "user",
                "2026-03-26T00:00:00Z",
                "AGENT-PARAPHRASE",
            ),
            (
                "c-direct-new",
                "PRIMARY_BACKEND: Gemini",
                "user_message",
                "user",
                "2026-06-09T00:00:00Z",
                "RAW-ETAN-DIRECT",
            ),
        ],
    )
    con.executemany(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES ('e-enrichment', ?)",
        [("c-note-old",), ("c-direct-new",)],
    )

    enqueue_provenance_resolution(con, "enrichment", chunk_id="c-direct-new")
    sweep_provenance_queue(con)

    row = con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-note-old'").fetchone()
    assert tuple(row) == (
        "superseded",
        "c-direct-new",
    )
