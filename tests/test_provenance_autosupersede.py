"""Fixture-only tests for ingest-time provenance autosupersession.

No live BrainLayer DB. The module under test is intentionally standalone:
it gathers same-entity chunks, detects same-attribute contradictions, delegates
authority to the provenance class gate, and only writes reversible supersedes
when dry_run=False.
"""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def con(tmp_path):
    conn = sqlite3.connect(tmp_path / "autosupersede_fixture.db")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_type TEXT,
            sender TEXT,
            created_at TEXT,
            provenance_class TEXT,
            status TEXT DEFAULT 'active',
            superseded_by TEXT
        );
        CREATE TABLE kg_entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE kg_entity_aliases (
            alias TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY (alias, entity_id)
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
    conn.commit()
    return conn


def _entity(conn, entity_id, name, *aliases):
    conn.execute("INSERT INTO kg_entities (id, name) VALUES (?, ?)", (entity_id, name))
    for alias in aliases:
        conn.execute("INSERT INTO kg_entity_aliases (alias, entity_id) VALUES (?, ?)", (alias, entity_id))


def _chunk(
    conn,
    chunk_id,
    entity_id,
    content,
    *,
    content_type="assistant_text",
    sender="assistant",
    created_at="2026-06-01T00:00:00Z",
    provenance_class="AGENT-INFERENCE",
    link=True,
):
    conn.execute(
        """
        INSERT INTO chunks (id, content, content_type, sender, created_at, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (chunk_id, content, content_type, sender, created_at, provenance_class),
    )
    if link:
        conn.execute("INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES (?, ?)", (entity_id, chunk_id))


def _new_chunk(chunk_id, entity, content, *, provenance_class="RAW-ETAN-DIRECT", created_at="2026-06-09T00:00:00Z"):
    return {
        "id": chunk_id,
        "entity": entity,
        "content": content,
        "content_type": "user_message",
        "sender": "user",
        "created_at": created_at,
        "provenance_class": provenance_class,
    }


def test_gather_same_entity_uses_normalized_aliases(con):
    from brainlayer.provenance_autosupersede import gather_same_entity

    _entity(con, "e-nano", "nano claw")
    _chunk(con, "c-old", "e-nano", "IDENTITY: HERMES_ADJACENT")

    rows = gather_same_entity(con, "nanoClaw")

    assert [row["id"] for row in rows] == ["c-old"]


def test_detect_contradiction_requires_same_attribute_and_different_value():
    from brainlayer.provenance_autosupersede import detect_contradiction

    is_contradiction, attribute = detect_contradiction(
        _new_chunk("c-new", "nanoClaw", "IDENTITY: DISTINCT"),
        {"content": "IDENTITY: HERMES_ADJACENT", "context": None, "mention_type": None, "id": "c-old"},
    )

    assert is_contradiction is True
    assert attribute == "IDENTITY"

    same_value, _ = detect_contradiction(
        _new_chunk("c-new", "nanoClaw", "IDENTITY: DISTINCT"),
        {"content": "IDENTITY: DISTINCT", "context": None, "mention_type": None, "id": "c-agree"},
    )
    different_attribute, _ = detect_contradiction(
        _new_chunk("c-new", "nanoClaw", "IDENTITY: DISTINCT"),
        {"content": "STATUS: ACTIVE", "context": None, "mention_type": None, "id": "c-status"},
    )
    assert same_value is False
    assert different_attribute is False


def test_nanoclaw_supersedes_stale_direct_and_flags_agent_inference(con):
    from brainlayer.provenance_autosupersede import auto_supersede

    _entity(con, "e-nano", "nano claw")
    _chunk(
        con,
        "c-old-direct",
        "e-nano",
        "IDENTITY: HERMES_ADJACENT",
        content_type="user_message",
        sender="user",
        created_at="2026-06-08T14:31:46Z",
        provenance_class="RAW-ETAN-DIRECT",
    )
    _chunk(
        con,
        "c-inference",
        "e-nano",
        "IDENTITY: GROWS_TOWARD_HERMES",
        created_at="2026-06-08T15:50:38Z",
        provenance_class="AGENT-INFERENCE",
    )
    _chunk(
        con,
        "c-new-direct",
        "e-nano",
        "IDENTITY: DISTINCT",
        content_type="user_message",
        sender="user",
        created_at="2026-06-08T18:50:44Z",
        provenance_class="RAW-ETAN-DIRECT",
        link=False,
    )

    report = auto_supersede(con, _new_chunk("c-new-direct", "nanoClaw", "IDENTITY: DISTINCT"), dry_run=False)

    assert report.superseded_count == 1
    assert report.pending_confirm_count == 1
    assert report.would_supersede_count == 1
    assert (report.entity, report.contradiction_count) == ("nano claw", 2)
    assert {decision.action for decision in report.decisions} == {"SUPERSEDE", "PENDING-USER-CONFIRM"}
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-old-direct'").fetchone()) == (
        "superseded",
        "c-new-direct",
    )
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-inference'").fetchone()) == (
        "active",
        None,
    )


def test_groq_to_gemini_operational_evidence_can_supersede_stale_status(con):
    from brainlayer.provenance_autosupersede import auto_supersede

    _entity(con, "e-provider", "provider router")
    _chunk(
        con,
        "c-groq",
        "e-provider",
        "PRIMARY_BACKEND: Groq",
        created_at="2026-03-26T00:00:00Z",
        provenance_class="AGENT-PARAPHRASE",
    )
    _chunk(
        con,
        "c-gemini",
        "e-provider",
        "PRIMARY_BACKEND: Gemini",
        created_at="2026-06-09T00:00:00Z",
        provenance_class="OPERATIONAL-EVIDENCE",
        link=False,
    )

    report = auto_supersede(
        con,
        _new_chunk(
            "c-gemini",
            "provider router",
            "PRIMARY_BACKEND: Gemini",
            provenance_class="OPERATIONAL-EVIDENCE",
        ),
        dry_run=False,
        enable_operational_evidence=True,
    )

    assert report.superseded_count == 1
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-groq'").fetchone()) == (
        "superseded",
        "c-gemini",
    )


def test_db_path_canonical_brainlayer_supersedes_old_zikaron(con):
    from brainlayer.provenance_autosupersede import auto_supersede

    _entity(con, "e-db", "BrainLayer database")
    _chunk(
        con,
        "c-zikaron",
        "e-db",
        "DB_PATH: ~/.local/share/zikaron/zikaron.db",
        created_at="2026-01-01T00:00:00Z",
        provenance_class="AGENT-PARAPHRASE",
    )
    _chunk(
        con,
        "c-brainlayer",
        "e-db",
        "DB_PATH: ~/.local/share/brainlayer/brainlayer.db",
        content_type="user_message",
        sender="user",
        created_at="2026-06-09T00:00:00Z",
        provenance_class="RAW-ETAN-DIRECT",
        link=False,
    )

    report = auto_supersede(
        con,
        _new_chunk("c-brainlayer", "BrainLayer database", "DB_PATH: ~/.local/share/brainlayer/brainlayer.db"),
        dry_run=False,
    )

    assert report.superseded_count == 1
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-zikaron'").fetchone()) == (
        "superseded",
        "c-brainlayer",
    )


def test_mcp_tool_count_same_class_uses_recency_and_supersedes_prior_counts(con):
    from brainlayer.provenance_autosupersede import auto_supersede

    _entity(con, "e-mcp", "MCP tool count")
    for chunk_id, value, created_at in [
        ("c-14", "14", "2026-06-01T00:00:00Z"),
        ("c-3", "3", "2026-06-02T00:00:00Z"),
        ("c-12", "12", "2026-06-03T00:00:00Z"),
    ]:
        _chunk(
            con,
            chunk_id,
            "e-mcp",
            f"MCP_TOOL_COUNT: {value}",
            content_type="user_message",
            sender="user",
            created_at=created_at,
            provenance_class="RAW-ETAN-DIRECT",
        )
    _chunk(
        con,
        "c-13",
        "e-mcp",
        "MCP_TOOL_COUNT: 13",
        content_type="user_message",
        sender="user",
        created_at="2026-06-09T00:00:00Z",
        provenance_class="RAW-ETAN-DIRECT",
        link=False,
    )

    report = auto_supersede(con, _new_chunk("c-13", "MCP tool count", "MCP_TOOL_COUNT: 13"), dry_run=False)

    assert report.superseded_count == 3
    superseded = con.execute(
        """
        SELECT id, status, superseded_by
        FROM chunks
        WHERE id IN ('c-14', 'c-3', 'c-12')
        ORDER BY id
        """
    ).fetchall()
    assert [tuple(row) for row in superseded] == [
        ("c-12", "superseded", "c-13"),
        ("c-14", "superseded", "c-13"),
        ("c-3", "superseded", "c-13"),
    ]


def test_personal_entity_is_skipped_and_never_auto_superseded(con):
    from brainlayer.provenance_autosupersede import auto_supersede

    _entity(con, "e-journal", "journal")
    _chunk(
        con,
        "c-health-old",
        "e-journal",
        "HEALTH_STATUS: recovering",
        content_type="journal",
        sender="user",
        created_at="2026-06-01T00:00:00Z",
        provenance_class="RAW-ETAN-DIRECT",
    )
    _chunk(
        con,
        "c-health-new",
        "e-journal",
        "HEALTH_STATUS: recovered",
        content_type="journal",
        sender="user",
        created_at="2026-06-09T00:00:00Z",
        provenance_class="RAW-ETAN-DIRECT",
        link=False,
    )

    report = auto_supersede(
        con,
        _new_chunk("c-health-new", "journal", "HEALTH_STATUS: recovered"),
        dry_run=False,
    )

    assert report.skipped_reason == "skipped: personal"
    assert report.superseded_count == 0
    assert report.decisions[0].action == "SKIP"
    assert tuple(con.execute("SELECT status, superseded_by FROM chunks WHERE id = 'c-health-old'").fetchone()) == (
        "active",
        None,
    )
