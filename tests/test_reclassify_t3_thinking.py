"""Reversible migration against an isolated SQLite store."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reclassify_t3_thinking.py"


@pytest.fixture
def migration():
    spec = importlib.util.spec_from_file_location("reclassify", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "copy.db"
    with sqlite3.connect(path) as c:
        c.execute(
            "CREATE TABLE chunks(id TEXT PRIMARY KEY, source TEXT, sender TEXT, "
            "source_class TEXT, metadata TEXT, content TEXT)"
        )
        for i, source, sender, sc, message in [
            (1, "t3", "assistant", "cli-agent", "msg%3Athinking%3A1"),
            (2, "t3", "assistant", None, "msg:thinking:0"),
            (3, "other", "assistant", "cli-agent", "msg:thinking:0"),
            (4, "t3", "user", "cli-agent", "msg:thinking:0"),
            (5, "t3", "assistant", "cli-agent", "thinking-policy"),
            (6, "t3", "assistant", "agent-thinking", "msg:thinking:0"),
        ]:
            metadata = (
                '{ "message_id": '
                + json.dumps("message:provider:claudeAgent:native-item:" + message)
                + ', "thinking_source_class": null, "custom": 1 }'
            )
            c.execute(
                "INSERT INTO chunks VALUES(?,?,?,?,?,?)",
                (str(i), source, sender, sc, metadata, "thinking mentioned in ordinary text"),
            )
    return path


def snapshot(db):
    with sqlite3.connect(db) as c:
        return c.execute("SELECT * FROM chunks ORDER BY id").fetchall()


def test_dry_apply_idempotent_undo_exact(migration, db):
    original = snapshot(db)
    assert migration.main(["--db", str(db)]) == 0
    assert snapshot(db) == original
    assert migration.main(["--db", str(db), "--apply", "--run-id", "test"]) == 0
    applied = snapshot(db)
    assert [r[3] for r in applied[:2]] == ["agent-thinking"] * 2
    assert applied[2:] == original[2:]
    assert json.loads(applied[0][4])["thinking_source_class"] == "cli-agent"
    assert json.loads(applied[1][4])["thinking_source_class"] is None
    assert migration.main(["--db", str(db), "--apply"]) == 0
    assert migration.main(["--db", str(db), "--apply", "--run-id", "test"]) == 0
    assert snapshot(db) == applied
    assert migration.main(["--db", str(db), "--undo", "--run-id", "test"]) == 0
    assert snapshot(db) == original
    assert migration.main(["--db", str(db), "--undo", "--run-id", "test"]) == 0
    assert snapshot(db) == original


def test_live_path_refused_before_open(migration):
    with pytest.raises(ValueError, match="canonical"):
        migration.main(["--db", str(Path.home() / ".local/share/brainlayer/brainlayer.db")])


def test_undo_refuses_later_edits(migration, db):
    migration.main(["--db", str(db), "--apply", "--run-id", "test"])
    with sqlite3.connect(db) as c:
        c.execute("UPDATE chunks SET metadata='{}' WHERE id='1'")
    before = snapshot(db)
    with pytest.raises(ValueError, match="changed since"):
        migration.main(["--db", str(db), "--undo", "--run-id", "test"])
    assert snapshot(db) == before


def test_live_hardlink_refused(migration, db, monkeypatch, tmp_path):
    monkeypatch.setattr(migration, "CANONICAL_DB", db)
    alias = tmp_path / "alias.db"
    alias.hardlink_to(db)
    with pytest.raises(ValueError, match="canonical"):
        migration.main(["--db", str(alias), "--apply"])


def test_busy_transaction_rolls_back_then_retries(migration, db, monkeypatch):
    monkeypatch.setattr(migration.time, "sleep", lambda _: None)
    attempts = []
    with sqlite3.connect(db, isolation_level=None) as c:

        def operation():
            c.execute("UPDATE chunks SET source_class='changed' WHERE id='1'")
            attempts.append(1)
            if len(attempts) == 1:
                exc = sqlite3.OperationalError("busy")
                exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
                raise exc

        migration.transaction(c, operation)
    assert len(attempts) == 2
    assert snapshot(db)[0][3] == "changed"


def test_partial_batches_undo_by_run(migration, db, monkeypatch):
    monkeypatch.setattr(migration, "BATCH_SIZE", 1)
    original = snapshot(db)
    transaction = migration.transaction
    batches = []

    def interrupted(c, operation):
        if batches:
            raise RuntimeError("interrupted")
        transaction(c, operation)
        batches.append(1)

    monkeypatch.setattr(migration, "transaction", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        migration.main(["--db", str(db), "--apply", "--run-id", "partial"])
    assert snapshot(db) != original
    monkeypatch.setattr(migration, "transaction", transaction)
    migration.main(["--db", str(db), "--undo", "--run-id", "partial"])
    assert snapshot(db) == original
