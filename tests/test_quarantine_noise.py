"""Tests for the noise-quarantine script."""

from __future__ import annotations

import json
import sqlite3
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.quarantine_noise import is_f_infra_root_content, is_precompact_candidate, run


def test_infra_root_normalization():
    """Root diagnostics are recognized when they appear at the content head."""
    assert is_f_infra_root_content("BrainLayer MCP not connected right now.")
    assert is_f_infra_root_content("🚨 **BrainLayer MCP NOT CONNECTED** I'll continue.")
    assert is_f_infra_root_content("⚠️ BrainLayer MCP not connected — can't call tools.")
    assert is_f_infra_root_content("Assistant: BrainLayer MCP is down.")
    assert not is_f_infra_root_content("Some background; BrainLayer MCP not connected")


def test_precompact_detection():
    """PreCompact chunks are recognized by content and checkpoint tags."""
    assert is_precompact_candidate("# PreCompact Checkpoint\nsession now.", None, None)
    assert is_precompact_candidate("normal text", "precompact_checkpoint", '["tag:one"]')
    assert is_precompact_candidate("normal text", None, '["precompact", "session"]')
    assert is_precompact_candidate("normal text", None, '["pre-compact", "session-checkpoint"]')
    assert is_precompact_candidate("normal text", None, None, "brainbar-c89abcd")
    assert not is_precompact_candidate("normal text", None, '["precompact-hook"]')
    assert not is_precompact_candidate("normal text", None, '["session-checkpoint"]')
    assert not is_precompact_candidate("normal text", None, None)


def _make_db(db_path: Path, rows: list[dict[str, object]]) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            tags TEXT,
            chunk_origin TEXT,
            project TEXT,
            source TEXT,
            source_file TEXT,
            created_at TEXT,
            importance REAL,
            archived INTEGER,
            status TEXT,
            archived_at TEXT
        )
        """
    )
    for row in rows:
        conn.execute(
            """
            INSERT INTO chunks (
                id, content, tags, chunk_origin, project, source, source_file,
                created_at, importance, archived, status, archived_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["content"],
                row["tags"],
                row["chunk_origin"],
                row["project"],
                row["source"],
                row["source_file"],
                row["created_at"],
                row["importance"],
                row["archived"],
                row["status"],
                row["archived_at"],
            ),
        )
    conn.commit()
    conn.close()


def test_dry_run_does_not_mutate_db(tmp_path: Path):
    """Dry-run mode reports candidates without writing fields."""
    db_path = tmp_path / "dry.db"
    _make_db(
        db_path,
        [
            {
                "id": "rt-root",
                "content": "BrainLayer MCP not connected right now.",
                "tags": None,
                "chunk_origin": None,
                "project": "brainlayer",
                "source": "realtime",
                "source_file": "a.jsonl",
                "created_at": "2026-05-17T00:00:00Z",
                "importance": 8.0,
                "archived": 0,
                "status": "active",
                "archived_at": None,
            },
            {
                "id": "pre-a",
                "content": "# PreCompact Checkpoint\\nsession",
                "tags": '["precompact"]',
                "chunk_origin": "precompact_checkpoint",
                "project": "brainlayer",
                "source": "mcp",
                "source_file": "b.jsonl",
                "created_at": "2026-05-17T00:01:00Z",
                "importance": 9.0,
                "archived": 0,
                "status": "active",
                "archived_at": None,
            },
        ],
    )

    out = StringIO()
    with redirect_stdout(out):
        code = run(["--db", str(db_path), "--json"])
    assert code == 0
    payload = json.loads(out.getvalue())
    assert payload["mode"] == "dry_run"
    assert payload["applied"] == 0
    assert payload["summary"]["total"] == 2
    assert set(row["id"] for row in payload["candidates"]) == {"rt-root", "pre-a"}

    with sqlite3.connect(db_path) as check:
        rows = {
            row[0]: row[1:] for row in check.execute("SELECT id, importance, archived, status, archived_at FROM chunks")
        }
    assert rows["rt-root"] == (8.0, 0, "active", None)
    assert rows["pre-a"] == (9.0, 0, "active", None)


def test_apply_updates_only_quarantine_fields_and_tag(tmp_path: Path):
    """Apply mode sets archive lifecycle fields and keeps existing tags intact."""
    db_path = tmp_path / "apply.db"
    _make_db(
        db_path,
        [
            {
                "id": "pre-valid",
                "content": "normal [PreCompact checkpoint]\nnote",
                "tags": '["keep-me"]',
                "chunk_origin": "precompact_checkpoint",
                "project": "brainlayer",
                "source": "mcp",
                "source_file": "b.jsonl",
                "created_at": "2026-05-17T00:00:00Z",
                "importance": 5.0,
                "archived": 0,
                "status": "active",
                "archived_at": None,
            },
            {
                "id": "rt-root",
                "content": "BrainLayer MCP not connected right now.",
                "tags": "not-json",
                "chunk_origin": None,
                "project": "brainlayer",
                "source": "realtime",
                "source_file": "a.jsonl",
                "created_at": "2026-05-17T01:00:00Z",
                "importance": 7.0,
                "archived": 1,
                "status": "archived",
                "archived_at": "2026-01-01T00:00:00Z",
            },
        ],
    )

    out = StringIO()
    with redirect_stdout(out):
        code = run(["--db", str(db_path), "--apply"])
    assert code == 0
    assert "Tag updates applied" in out.getvalue()

    with sqlite3.connect(db_path) as check:
        row_a = check.execute(
            "SELECT importance, archived, status, archived_at, tags FROM chunks WHERE id = 'pre-valid'"
        ).fetchone()
        row_b = check.execute(
            "SELECT importance, archived, status, archived_at, tags FROM chunks WHERE id = 'rt-root'"
        ).fetchone()

    assert row_a[0] == 0
    assert row_a[1] == 1
    assert row_a[2] == "archived"
    assert row_a[4] is not None
    assert "quarantined/noise" in row_a[4]
    assert "keep-me" in row_a[4]

    assert row_b[0] == 0
    assert row_b[1] == 1
    assert row_b[2] == "archived"
    assert row_b[4] == "not-json"
