from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import apsw

from brainlayer.writer_telemetry import fingerprint_sql, writer_span


def _wait_for_current_statement(heartbeat_dir: Path, fingerprint: str, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    last_payload: dict = {}
    while time.monotonic() < deadline:
        files = list(heartbeat_dir.glob("writer-txn-*.json"))
        if files:
            try:
                last_payload = json.loads(files[0].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.01)
                continue
            active = last_payload.get("active_transactions") or []
            if active and active[0].get("current_statement_fingerprint") == fingerprint:
                return active[0]
        time.sleep(0.01)
    raise AssertionError(f"slow statement fingerprint never became visible; last={last_payload}")


def test_slow_fts_write_exposes_open_heartbeat_and_completed_statement_span(tmp_path, monkeypatch):
    log_path = tmp_path / "writer-telemetry.jsonl"
    heartbeat_dir = tmp_path / "heartbeats"
    db_path = tmp_path / "fts-gate.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY", "1")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(log_path))
    monkeypatch.setenv("BRAINLAYER_WRITER_HEARTBEAT_DIR", str(heartbeat_dir))
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_HEARTBEAT_INTERVAL_MS", "10")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_FTS_SAMPLE_TTL_SECONDS", "0")
    setup = apsw.Connection(str(db_path))
    setup.execute("PRAGMA journal_mode=WAL")
    setup.execute("CREATE TABLE source_docs(content TEXT NOT NULL)")
    setup.executemany("INSERT INTO source_docs VALUES (?)", [(f"document {index}",) for index in range(100)])
    setup.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(content)")
    setup.close()

    entered_slow_function = threading.Event()
    release_slow_function = threading.Event()
    worker_errors: list[BaseException] = []
    sql = "INSERT INTO chunks_fts(content) SELECT slow_fts(content) FROM source_docs"
    expected_fingerprint = fingerprint_sql(sql).digest

    def run_slow_fts_write() -> None:
        conn = apsw.Connection(str(db_path))

        def slow_fts(value: str) -> str:
            entered_slow_function.set()
            if not release_slow_function.wait(5.0):
                raise TimeoutError("gate did not release slow FTS function")
            return value

        conn.create_scalar_function("slow_fts", slow_fts, 1)
        try:
            with writer_span(
                conn,
                db_path=db_path,
                producer="gate",
                lane="maintenance",
                operation="slow_fts_write",
                rows_planned=100,
            ) as span:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(sql)
                conn.execute("COMMIT")
                span.commit(rows_touched=100)
        except BaseException as exc:
            worker_errors.append(exc)
        finally:
            conn.close()

    worker = threading.Thread(target=run_slow_fts_write, name="slow-fts-gate")
    worker.start()
    assert entered_slow_function.wait(3.0), "slow FTS statement did not start"

    active = _wait_for_current_statement(heartbeat_dir, expected_fingerprint)
    assert active["txn_started_at"]
    assert active["txn_started_monotonic"] > 0
    assert active["current_statement"].startswith("INSERT INTO chunks_fts")
    assert active["current_statement_started_at"]
    assert active["current_statement_open_ms"] >= 0

    release_slow_function.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert worker_errors == []
    assert list(heartbeat_dir.glob("writer-txn-*.json")) == []

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    started = [event for event in events if event["event"] == "txn_started"]
    finished = [event for event in events if event["event"] == "txn_finished"]
    assert len(started) == 1
    assert len(finished) == 1
    assert finished[0]["outcome"] == "commit"
    assert finished[0]["duration_ms"] > 0
    statement = next(item for item in finished[0]["statements"] if item["fingerprint"] == expected_fingerprint)
    assert statement["started_at"]
    assert statement["max_duration_ms"] > 0
    assert statement["vm_steps"] > 0
