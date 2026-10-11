import hashlib
import json
import sqlite3
import struct
from contextlib import closing
from pathlib import Path

import pytest
import sqlite_vec

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
                ("claude-desktop", "/desktop.jsonl", "claude_desktop", None),
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
        conn.execute("CREATE TABLE chunk_vectors (chunk_id TEXT PRIMARY KEY, embedding BLOB)")
        conn.execute("CREATE TABLE chunk_vectors_binary (chunk_id TEXT PRIMARY KEY, embedding BLOB)")
        for table in ("chunk_vectors", "chunk_vectors_binary"):
            conn.executemany(
                f"INSERT INTO {table}(chunk_id, embedding) VALUES (?, X'00')",
                [(row[0],) for row in conn.execute("SELECT id FROM chunks")],
            )


def test_migration_backfills_only_unambiguous_rows_and_writes_sha_ledgers(tmp_path: Path) -> None:
    db_path = tmp_path / "copy.db"
    _legacy_db(db_path)

    receipt = migration.migrate_source_class(db_path, git_sha=GIT_SHA, actor="pytest")

    assert receipt["row_count_before"] == receipt["row_count_after"] == 7
    assert receipt["distribution"] == {
        "NULL": 1,
        "brain-worker": 1,
        "cli-agent": 2,
        "desktop": 1,
        "fleet-coordination": 1,
        "subagent": 1,
    }
    assert receipt["fts_rows_removed"] == {"chunk_fts_rowids": 1, "chunks_fts": 1}
    assert receipt["vector_rows_removed"] == {"chunk_vectors": 1, "chunk_vectors_binary": 1}
    with sqlite3.connect(db_path) as conn:
        assert dict(conn.execute("SELECT id, source_class FROM chunks")) == {
            "ambiguous": None,
            "brain-worker": "brain-worker",
            "cli": "cli-agent",
            "claude-desktop": "desktop",
            "desktop": "cli-agent",
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
        assert conn.execute("SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = 'brain-worker'").fetchone()[0] == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM chunk_vectors_binary WHERE chunk_id = 'brain-worker'").fetchone()[0] == 0
        )

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


def test_migration_uses_subagent_attribution_and_not_chunk_role_mentions(tmp_path: Path) -> None:
    db_path = tmp_path / "copy.db"
    ordinary = tmp_path / ".claude" / "projects" / "-Users-test-Gits-brainlayer" / "s" / "subagents" / "a.jsonl"
    worker = ordinary.with_name("b.jsonl")
    ordinary.parent.mkdir(parents=True)
    ordinary.write_text(json.dumps({"attributionAgent": "general-purpose"}) + "\n")
    worker.write_text(json.dumps({"attributionAgent": "brain_worker"}) + "\n")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT, source_file TEXT, source TEXT, provenance_class TEXT)"
        )
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, 'claude_code', NULL)",
            [
                ("ordinary", "Discuss the brain-worker and /weave rules.", str(ordinary)),
                ("worker", "Neutral task text.", str(worker)),
            ],
        )

    migration.migrate_source_class(db_path, git_sha=GIT_SHA, actor="pytest")

    with sqlite3.connect(db_path) as conn:
        assert dict(conn.execute("SELECT id, source_class FROM chunks")) == {
            "ordinary": "fleet-coordination",
            "worker": "brain-worker",
        }


def test_migration_loads_vec0_and_purges_brain_worker_candidates(tmp_path: Path) -> None:
    db_path = tmp_path / "vec0-copy.db"
    _legacy_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE chunk_vectors")
        conn.execute("DROP TABLE chunk_vectors_binary")
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())
        conn.enable_load_extension(False)
        conn.execute("CREATE VIRTUAL TABLE chunk_vectors USING vec0(chunk_id TEXT PRIMARY KEY, embedding FLOAT[1024])")
        conn.execute(
            "CREATE VIRTUAL TABLE chunk_vectors_binary USING vec0(chunk_id TEXT PRIMARY KEY, embedding BIT[1024])"
        )
        float_embedding = struct.pack("<1024f", *([0.0] * 1024))
        for chunk_id in ("cli", "brain-worker"):
            conn.execute("INSERT INTO chunk_vectors(chunk_id, embedding) VALUES (?, ?)", (chunk_id, float_embedding))
            conn.execute(
                "INSERT INTO chunk_vectors_binary(chunk_id, embedding) VALUES (?, vec_quantize_binary(?))",
                (chunk_id, float_embedding),
            )

    receipt = migration.migrate_source_class(db_path, git_sha=GIT_SHA, actor="pytest")

    assert receipt["vector_rows_removed"] == {"chunk_vectors": 1, "chunk_vectors_binary": 1}
    with sqlite3.connect(db_path) as conn:
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())
        assert conn.execute("SELECT chunk_id FROM chunk_vectors").fetchall() == [("cli",)]
        assert conn.execute("SELECT chunk_id FROM chunk_vectors_binary").fetchall() == [("cli",)]


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

    assert receipt["row_count_before"] == receipt["row_count_after"] == 7
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


def _t3_copy(path):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, source_file, provenance_class, source_class, source)")
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
            [
                ("thread", "/t3.jsonl", "t3-thread", "desktop", "realtime_watcher"),
                ("app", "/codex.jsonl", "t3-app-session", "desktop", "codex"),
                ("null", "/t3.jsonl", "t3-thread", None, "t3"),
                ("already", "/t3.jsonl", "t3-thread", "cli-agent", "t3"),
                ("desktop", "/desktop.jsonl", None, "desktop", "claude_desktop"),
                ("chatgpt", "/Users/test/.codex/sessions/app.jsonl", "t3-thread", "desktop", "chatgpt"),
                ("gemini", "/gemini.jsonl", "t3-thread", "desktop", "gemini_desktop"),
                ("codex", "/codex.jsonl", "codex-session", "cli-agent", "codex"),
                ("worker", "/worker.jsonl", "t3-thread", "brain-worker", "t3"),
                ("epistemic", "/t3.jsonl", "AGENT-INFERENCE", "desktop", "t3"),
                ("codex-path", "/Users/test/.codex/sessions/x.jsonl", "RAW-ETAN-DIRECT", "desktop", "realtime_watcher"),
            ],
        )


def test_t3_reclassification_dry_run_apply_and_idempotence(tmp_path):
    path = tmp_path / "copy.db"
    _t3_copy(path)
    before = path.read_bytes()
    dry = migration.reclassify_t3_source_class(path, git_sha=GIT_SHA, actor="pytest")
    assert path.read_bytes() == before
    assert dry["candidate_rows"] == 4 and dry["updated_rows"] == 0
    assert dry["before"] == dry["after"]
    applied = migration.reclassify_t3_source_class(path, git_sha=GIT_SHA, actor="pytest", apply=True)
    assert applied["updated_rows"] == 4
    assert applied["after"]["AGENT-INFERENCE/cli-agent"] == 1
    assert applied["after"]["RAW-ETAN-DIRECT/cli-agent"] == 1
    with sqlite3.connect(path) as conn:
        assert dict(conn.execute("SELECT id, source_class FROM chunks")) == {
            "thread": "cli-agent",
            "app": "cli-agent",
            "null": None,
            "already": "cli-agent",
            "desktop": "desktop",
            "chatgpt": "desktop",
            "gemini": "desktop",
            "codex": "cli-agent",
            "worker": "brain-worker",
            "epistemic": "cli-agent",
            "codex-path": "cli-agent",
        }
    repeat = migration.reclassify_t3_source_class(path, git_sha=GIT_SHA, actor="pytest", apply=True)
    assert repeat["candidate_rows"] == repeat["updated_rows"] == repeat["batches"] == 0


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_t3_reclassification_canonical_gate_and_readonly_dry_run(tmp_path, monkeypatch, alias):
    canonical = tmp_path / "canonical.db"
    _t3_copy(canonical)
    monkeypatch.setattr(migration, "get_db_path", lambda: canonical)
    monkeypatch.delenv("BRAINLAYER_OFFLINE_MIGRATOR_GATED_SWAP", raising=False)
    target = canonical
    if alias != "direct":
        target = tmp_path / "alias.db"
        if alias == "symlink":
            target.symlink_to(canonical)
        else:
            target.hardlink_to(canonical)
    with pytest.raises(ValueError, match="canonical"):
        migration.reclassify_t3_source_class(target, git_sha=GIT_SHA, actor="pytest", apply=True)
    real_connect = sqlite3.connect
    opened = []

    def connect(database, **kwargs):
        opened.append(database)
        return real_connect(database, **kwargs)

    monkeypatch.setattr(migration.sqlite3, "connect", connect)
    before = canonical.read_bytes()
    dry = migration.reclassify_t3_source_class(target, git_sha=GIT_SHA, actor="pytest")
    assert dry["candidate_rows"] == 4
    assert canonical.read_bytes() == before
    assert opened == [f"{canonical.as_uri()}?mode=ro"]
    monkeypatch.setenv("BRAINLAYER_OFFLINE_MIGRATOR_GATED_SWAP", "1")
    applied = migration.reclassify_t3_source_class(target, git_sha=GIT_SHA, actor="pytest", apply=True)
    assert applied["updated_rows"] == 4
    assert opened[-1] == f"{canonical.as_uri()}?mode=rw"


def test_t3_reclassification_checkpoints_before_every_three_batches_and_after(tmp_path, monkeypatch):
    path = tmp_path / "copy.db"
    _t3_copy(path)
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, 't3')",
            [(f"bulk-{i}", "/t3.jsonl", "AGENT-PARAPHRASE", "desktop") for i in range(15000)],
        )
    checkpoints = []
    real_checkpoint = migration._checkpoint_full

    def checkpoint(conn):
        checkpoints.append(conn.execute("SELECT COUNT(*) FROM chunks WHERE source_class = 'cli-agent'").fetchone()[0])
        real_checkpoint(conn)

    monkeypatch.setattr(migration, "_checkpoint_full", checkpoint)
    receipt = migration.reclassify_t3_source_class(path, git_sha=GIT_SHA, actor="pytest", apply=True)
    assert receipt["updated_rows"] == 15004 and receipt["batches"] == 4
    assert checkpoints == [2, 15002, 15006]
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM migration_events").fetchone()[0] == 4


def test_t3_reclassification_ledger_audits_exact_ids_and_supports_rollback(tmp_path):
    path = tmp_path / "copy.db"
    _t3_copy(path)
    applied = migration.reclassify_t3_source_class(path, git_sha=GIT_SHA, actor="pytest", apply=True)
    with sqlite3.connect(path) as conn:
        event_id, commit_hash, actor, status, details = conn.execute(
            "SELECT id, commit_hash, actor, status, details FROM migration_events"
        ).fetchone()
        details = json.loads(details)
        changed_ids = ["app", "codex-path", "epistemic", "thread"]
        expected_hash = hashlib.sha256(json.dumps(changed_ids, separators=(",", ":")).encode()).hexdigest()
        assert (commit_hash, actor, status) == (GIT_SHA, "pytest", "success")
        assert applied["ledger_event_ids"] == [event_id]
        assert applied["changed_ids_sha256"] == details["changed_ids_sha256"] == expected_hash
        assert details["changed_ids"] == changed_ids
        assert details["changed_count"] == applied["updated_rows"] == 4
        assert details["prior_class"] == "desktop"
        conn.executemany(
            "UPDATE chunks SET source_class = ? WHERE id = ? AND source_class = 'cli-agent'",
            [(details["prior_class"], chunk_id) for chunk_id in details["changed_ids"]],
        )
    dry = migration.reclassify_t3_source_class(path, git_sha=GIT_SHA, actor="pytest")
    assert dry["candidate_rows"] == 4


def test_t3_reclassification_ledger_failure_rolls_back_batch(tmp_path, monkeypatch):
    path = tmp_path / "copy.db"
    _t3_copy(path)
    ensure_ledgers = migration._ensure_ledgers

    def failing_ledger(conn):
        ensure_ledgers(conn)
        conn.execute("""CREATE TRIGGER reject_event BEFORE INSERT ON migration_events
                        BEGIN SELECT RAISE(ABORT, 'ledger unavailable'); END""")

    monkeypatch.setattr(migration, "_ensure_ledgers", failing_ledger)
    with pytest.raises(sqlite3.IntegrityError, match="ledger unavailable"):
        migration.reclassify_t3_source_class(path, git_sha=GIT_SHA, actor="pytest", apply=True)
    dry = migration.reclassify_t3_source_class(path, git_sha=GIT_SHA, actor="pytest")
    assert dry["candidate_rows"] == 4
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM migration_events").fetchone()[0] == 0


@pytest.mark.parametrize("batch_size", [0, 4999, 10001])
def test_t3_reclassification_enforces_bounded_batches(tmp_path, batch_size):
    path = tmp_path / "copy.db"
    _t3_copy(path)
    with pytest.raises(ValueError, match="5,000.*10,000"):
        migration.reclassify_t3_source_class(path, git_sha=GIT_SHA, actor="pytest", batch_size=batch_size)
