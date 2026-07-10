from __future__ import annotations

import json
import threading
from pathlib import Path

import apsw


def _configure_paths(monkeypatch, tmp_path: Path, *, enabled: bool = True) -> tuple[Path, Path]:
    log_path = tmp_path / "logs" / "writer-telemetry.jsonl"
    heartbeat_dir = tmp_path / "heartbeats"
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY", "1" if enabled else "0")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(log_path))
    monkeypatch.setenv("BRAINLAYER_WRITER_HEARTBEAT_DIR", str(heartbeat_dir))
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_MAX_BYTES", "1048576")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_BACKUPS", "2")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_HEARTBEAT_INTERVAL_MS", "10")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_FTS_SAMPLE_TTL_SECONDS", "0")
    return log_path, heartbeat_dir


def _events(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_sql_fingerprint_is_stable_and_does_not_expose_literals():
    from brainlayer.writer_telemetry import fingerprint_sql

    first = fingerprint_sql(" INSERT  INTO chunks(content) VALUES ('private-one') ")
    second = fingerprint_sql("INSERT INTO chunks(content)\nVALUES ('private-two')")

    assert first.digest == second.digest
    assert first.normalized == "INSERT INTO chunks(content) VALUES (?)"
    assert "private" not in first.normalized


def test_append_event_rotates_at_configured_size(tmp_path, monkeypatch):
    from brainlayer.writer_telemetry import append_event

    log_path, _heartbeat_dir = _configure_paths(monkeypatch, tmp_path)
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_MAX_BYTES", "180")

    assert append_event({"event": "one", "payload": "x" * 120}) is True
    assert append_event({"event": "two", "payload": "y" * 120}) is True

    assert log_path.exists()
    assert log_path.with_name("writer-telemetry.jsonl.1").exists()
    assert json.loads(log_path.read_text(encoding="utf-8"))["event"] == "two"


def test_append_event_is_fail_open_when_sink_is_unwritable(tmp_path, monkeypatch):
    from brainlayer.writer_telemetry import append_event

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block", encoding="utf-8")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY", "1")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(blocker / "writer-telemetry.jsonl"))

    assert append_event({"event": "must-not-break-writer"}) is False


def test_writer_span_emits_start_end_metrics_and_clears_heartbeat(tmp_path, monkeypatch):
    from brainlayer.writer_telemetry import writer_span

    log_path, heartbeat_dir = _configure_paths(monkeypatch, tmp_path)
    db_path = tmp_path / "brainlayer.db"
    conn = apsw.Connection(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(content)")

    with writer_span(
        conn,
        db_path=db_path,
        producer="index",
        lane="batch",
        operation="upsert_chunks",
        rows_planned=1,
    ) as span:
        heartbeat_files = list(heartbeat_dir.glob("writer-txn-*.json"))
        assert len(heartbeat_files) == 1
        active = json.loads(heartbeat_files[0].read_text(encoding="utf-8"))
        assert active["active_transactions"][0]["producer"] == "index"
        assert active["active_transactions"][0]["txn_started_monotonic"] > 0

        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO chunks VALUES (?, ?)", ("one", "hello"))
        conn.execute("INSERT INTO chunks_fts(content) VALUES (?)", ("hello",))
        conn.execute("COMMIT")
        span.commit(rows_touched=1)

    assert list(heartbeat_dir.glob("writer-txn-*.json")) == []
    events = _events(log_path)
    assert [event["event"] for event in events] == ["txn_started", "txn_finished"]
    finished = events[-1]
    assert finished["producer"] == "index"
    assert finished["lane"] == "batch"
    assert finished["operation"] == "upsert_chunks"
    assert finished["outcome"] == "commit"
    assert finished["duration_ms"] >= 0
    assert finished["rows_planned"] == 1
    assert finished["rows_touched"] >= 1
    assert finished["executor_pid"] > 0
    assert finished["wal_bytes_before"] >= 0
    assert finished["wal_bytes_after"] >= 0
    assert finished["wal_frames_before"] >= 0
    assert finished["wal_frames_after"] >= 0
    assert "chunks_fts" in finished["fts_segments"]
    assert any(statement["normalized_sql"].startswith("INSERT INTO chunks") for statement in finished["statements"])
    assert all(statement["max_duration_ms"] >= 0 for statement in finished["statements"])


def test_finishing_span_wins_race_with_stale_heartbeat_flush(tmp_path, monkeypatch):
    import brainlayer.writer_telemetry as writer_telemetry

    _log_path, heartbeat_dir = _configure_paths(monkeypatch, tmp_path)
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_HEARTBEAT_INTERVAL_MS", "1000")
    db_path = tmp_path / "brainlayer.db"
    conn = apsw.Connection(str(db_path))
    conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY)")
    span = writer_telemetry.start_writer_span(
        conn,
        db_path=db_path,
        producer="index",
        lane="batch",
        operation="upsert_chunks",
    )
    heartbeat_path = next(heartbeat_dir.glob("writer-txn-*.json"))
    stale_flush_entered = threading.Event()
    release_stale_flush = threading.Event()
    finish_returned = threading.Event()
    original_atomic_write = writer_telemetry._atomic_write_json

    def delayed_atomic_write(path, payload):
        stale_flush_entered.set()
        assert release_stale_flush.wait(3.0)
        return original_atomic_write(path, payload)

    monkeypatch.setattr(writer_telemetry, "_atomic_write_json", delayed_atomic_write)
    stale_flush = threading.Thread(target=writer_telemetry._flush_heartbeat, args=(heartbeat_path,))
    stale_flush.start()
    assert stale_flush_entered.wait(1.0)

    def finish_span() -> None:
        span.finish("commit")
        finish_returned.set()

    finisher = threading.Thread(target=finish_span)
    finisher.start()
    finish_returned.wait(0.1)
    release_stale_flush.set()
    stale_flush.join(timeout=3.0)
    finisher.join(timeout=3.0)

    assert not stale_flush.is_alive()
    assert not finisher.is_alive()
    assert not heartbeat_path.exists()
    conn.close()


def test_writer_span_records_rollback_without_masking_writer_result(tmp_path, monkeypatch):
    from brainlayer.writer_telemetry import writer_span

    log_path, heartbeat_dir = _configure_paths(monkeypatch, tmp_path)
    db_path = tmp_path / "brainlayer.db"
    conn = apsw.Connection(str(db_path))
    conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY)")

    with writer_span(
        conn,
        db_path=db_path,
        producer="mcp",
        lane="interactive",
        operation="store_memory",
    ) as span:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO chunks VALUES ('rolled-back')")
        conn.execute("ROLLBACK")
        span.rollback(error="synthetic")

    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (0,)
    assert list(heartbeat_dir.glob("writer-txn-*.json")) == []
    finished = _events(log_path)[-1]
    assert finished["outcome"] == "rollback"
    assert finished["error"] == "synthetic"


def test_writer_span_is_noop_when_disabled(tmp_path, monkeypatch):
    from brainlayer.writer_telemetry import writer_span

    log_path, heartbeat_dir = _configure_paths(monkeypatch, tmp_path, enabled=False)
    db_path = tmp_path / "brainlayer.db"
    conn = apsw.Connection(str(db_path))
    conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY)")

    with writer_span(
        conn,
        db_path=db_path,
        producer="index",
        lane="batch",
        operation="upsert_chunks",
    ) as span:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO chunks VALUES ('one')")
        conn.execute("COMMIT")
        span.commit(rows_touched=1)

    assert not log_path.exists()
    assert not heartbeat_dir.exists()


def test_writer_span_preserves_existing_exec_trace(tmp_path, monkeypatch):
    from brainlayer.writer_telemetry import writer_span

    _configure_paths(monkeypatch, tmp_path)
    db_path = tmp_path / "brainlayer.db"
    conn = apsw.Connection(str(db_path))
    conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY)")
    traced: list[str] = []

    def trace(_cursor, statement, _bindings):
        traced.append(str(statement))
        return True

    conn.setexectrace(trace)
    with writer_span(
        conn,
        db_path=db_path,
        producer="index",
        lane="batch",
        operation="upsert_chunks",
    ) as span:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO chunks VALUES ('one')")
        conn.execute("COMMIT")
        span.commit(rows_touched=1)

    assert any("INSERT INTO chunks" in statement for statement in traced)
    assert conn.getexectrace() is trace


def test_fts_sampling_skips_missing_shadow_tables_without_sqlite_errors(tmp_path, monkeypatch, caplog):
    import brainlayer.vector_store  # noqa: F401 - installs the production APSW log callback
    from brainlayer.writer_telemetry import writer_span

    _configure_paths(monkeypatch, tmp_path)
    db_path = tmp_path / "empty.db"
    conn = apsw.Connection(str(db_path))

    with writer_span(
        conn,
        db_path=db_path,
        producer="vector_store",
        lane="maintenance",
        operation="init",
    ) as span:
        conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY)")
        span.complete()

    assert "no such table" not in caplog.text.lower()
