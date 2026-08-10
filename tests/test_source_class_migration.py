import json
import sqlite3
from pathlib import Path

import pytest

import brainlayer.source_class_migration as migration

GIT_SHA = "0123456789abcdef0123456789abcdef01234567"


def _legacy_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, source_file TEXT, source TEXT, provenance_class TEXT)")
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?)",
            [
                ("cli", "/Users/test/.codex/sessions/a.jsonl", "codex", None),
                ("desktop", "/Users/test/.codex/sessions/app.jsonl", "codex", "t3-app-session"),
                (
                    "subagent",
                    "/Users/test/.claude/projects/-Users-test-Gits-domica/s/subagents/a.jsonl",
                    "claude_code",
                    None,
                ),
                (
                    "fleet",
                    "/Users/test/.claude/projects/-Users-test-Gits-brainlayer/s/subagents/a.jsonl",
                    "claude_code",
                    None,
                ),
                (
                    "brain-worker",
                    "/Users/test/.claude/projects/-Users-test-Gits-brainlayer/s/subagents/brain-worker/a.jsonl",
                    "claude_code",
                    None,
                ),
                ("ambiguous", "/tmp/unattributed.jsonl", "manual", None),
            ],
        )
        conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(content, chunk_id UNINDEXED)")
        conn.executemany(
            "INSERT INTO chunks_fts(content, chunk_id) VALUES (?, ?)",
            [("visibility token", row[0]) for row in conn.execute("SELECT id FROM chunks")],
        )
        conn.execute("CREATE TABLE chunk_fts_rowids (chunk_id TEXT PRIMARY KEY, rowid INTEGER NOT NULL)")
        conn.executemany(
            "INSERT INTO chunk_fts_rowids(chunk_id, rowid) VALUES (?, ?)",
            [(row[0], position) for position, row in enumerate(conn.execute("SELECT id FROM chunks"), start=1)],
        )


def test_migration_backfills_only_unambiguous_rows_and_writes_sha_ledgers(tmp_path: Path) -> None:
    db_path = tmp_path / "copy.db"
    _legacy_db(db_path)

    receipt = migration.migrate_source_class(db_path, git_sha=GIT_SHA, actor="pytest")

    assert receipt["row_count_before"] == receipt["row_count_after"] == 6
    assert receipt["distribution"] == {
        "NULL": 1,
        "brain-worker": 1,
        "cli-agent": 1,
        "desktop": 1,
        "fleet-coordination": 1,
        "subagent": 1,
    }
    assert receipt["fts_rows_removed"] == {"chunk_fts_rowids": 1, "chunks_fts": 1}
    with sqlite3.connect(db_path) as conn:
        assert dict(conn.execute("SELECT id, source_class FROM chunks")) == {
            "ambiguous": None,
            "brain-worker": "brain-worker",
            "cli": "cli-agent",
            "desktop": "desktop",
            "fleet": "fleet-coordination",
            "subagent": "subagent",
        }
        schema_details = json.loads(
            conn.execute(
                "SELECT details FROM schema_migrations WHERE name = ?",
                (migration.MIGRATION_NAME,),
            ).fetchone()[0]
        )
        event = conn.execute(
            "SELECT commit_hash, actor, status FROM migration_events WHERE id = ?",
            (migration.MIGRATION_EVENT_ID,),
        ).fetchone()
        assert schema_details["git_sha"] == GIT_SHA
        assert event == (GIT_SHA, "pytest", "success")
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'brain-worker'").fetchone()[0] == 0

    second = migration.migrate_source_class(db_path, git_sha=GIT_SHA, actor="pytest")
    assert second["already_applied"] is True
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM migration_events WHERE id = ?",
                (migration.MIGRATION_EVENT_ID,),
            ).fetchone()[0]
            == 1
        )


def test_migration_rejects_non_commit_sha(tmp_path: Path) -> None:
    db_path = tmp_path / "copy.db"
    _legacy_db(db_path)

    with pytest.raises(ValueError, match="40-character"):
        migration.migrate_source_class(db_path, git_sha="HEAD", actor="pytest")


def test_migration_rerun_rejects_a_different_code_sha(tmp_path: Path) -> None:
    db_path = tmp_path / "copy.db"
    _legacy_db(db_path)
    migration.migrate_source_class(db_path, git_sha=GIT_SHA, actor="pytest")

    with pytest.raises(RuntimeError, match="ledger SHA mismatch"):
        migration.migrate_source_class(
            db_path,
            git_sha="fedcba9876543210fedcba9876543210fedcba98",
            actor="pytest",
        )


def test_migration_refuses_canonical_database_even_when_explicit(monkeypatch, tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.db"
    _legacy_db(canonical)
    monkeypatch.setattr(migration, "get_db_path", lambda: canonical)

    with pytest.raises(ValueError, match="canonical"):
        migration.migrate_source_class(canonical, git_sha=GIT_SHA, actor="pytest")


def test_migration_allows_canonical_database_only_with_supervised_gate(monkeypatch, tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.db"
    _legacy_db(canonical)
    monkeypatch.setattr(migration, "get_db_path", lambda: canonical)
    monkeypatch.setenv("BRAINLAYER_OFFLINE_MIGRATOR_GATED_SWAP", "1")

    receipt = migration.migrate_source_class(canonical, git_sha=GIT_SHA, actor="pytest")

    assert receipt["row_count_before"] == receipt["row_count_after"] == 6
    assert receipt["git_sha"] == GIT_SHA


def test_idempotent_malformed_receipt_closes_connection(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "copy.db"
    _legacy_db(db_path)
    migration.migrate_source_class(db_path, git_sha=GIT_SHA, actor="pytest")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE schema_migrations SET details = 'not-json' WHERE name = ?",
            (migration.MIGRATION_NAME,),
        )

    real_connect = sqlite3.connect
    tracked: list[object] = []

    class TrackedConnection:
        def __init__(self, *args, **kwargs) -> None:
            self.connection = real_connect(*args, **kwargs)
            self.closed = False

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def close(self) -> None:
            self.closed = True
            self.connection.close()

    def connect(*args, **kwargs):
        connection = TrackedConnection(*args, **kwargs)
        tracked.append(connection)
        return connection

    monkeypatch.setattr(migration.sqlite3, "connect", connect)

    with pytest.raises(json.JSONDecodeError):
        migration.migrate_source_class(db_path, git_sha=GIT_SHA, actor="pytest")

    assert len(tracked) == 1
    assert tracked[0].closed is True
