from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import apsw
import pytest
import sqlite_vec

from brainlayer.drain import _precompute_event_embeddings, drain_once
from brainlayer.native_session_header_events import (
    MAX_EVENT_BYTES,
    NativeSessionHeaderEventError,
    build_native_session_header_event,
)
from brainlayer.queue_io import enqueue_watcher_native_session_header
from brainlayer.raw_source_archive import RawSourceArchive, create_raw_source_archive
from brainlayer.vector_store import VectorStore

SESSION_A = "01a08d1e-42f6-7093-83a4-1c1b160f4911"
SESSION_B = "01a08d1e-4373-7631-b908-1a465cb013d3"


def _header(session_id: str, **extra: object) -> bytes:
    payload = {"id": session_id, **extra}
    return (json.dumps({"type": "session_meta", "payload": payload}, separators=(",", ":")) + "\n").encode()


def _coordinates(raw: bytes, start: int = 0) -> dict[str, object]:
    return {
        "record_start_offset": start,
        "record_end_offset": start + len(raw),
        "raw_record_hash": hashlib.sha256(raw).hexdigest(),
    }


def _intent(
    archive: RawSourceArchive,
    source: Path,
    root: dict,
    occurrence: dict,
    raw: bytes,
    start: int = 0,
) -> dict[str, object]:
    return {
        "archive": archive,
        "provider": "codex",
        "source_file": str(source),
        "watcher_generation": 7,
        "root_receipt": root,
        "occurrence_receipt": occurrence,
        **_coordinates(raw, start),
    }


def _enqueue(queue: Path, intent: dict[str, object]) -> Path:
    values = dict(intent)
    archive = values.pop("archive")
    return enqueue_watcher_native_session_header(raw_archive=archive, queue_dir=queue, **values)


def _rows(db: Path, table: str) -> list[tuple]:
    with sqlite3.connect(db) as conn:
        return conn.execute(f"SELECT * FROM {table}").fetchall()


def _new_db(path: Path) -> None:
    VectorStore(path).close()


def _vector_count(path: Path) -> int:
    conn = apsw.Connection(str(path))
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0])
    finally:
        conn.close()


def test_real_queue_drain_catalogs_header_without_chunk_or_embedding(tmp_path: Path) -> None:
    db, queue, log = tmp_path / "db.sqlite", tmp_path / "queue", tmp_path / "drain.log"
    _new_db(db)
    archive = create_raw_source_archive(db, {})
    source = tmp_path / "session.jsonl"
    raw = _header(SESSION_A, base_instructions="must never enter queue", body={"secret": "absent"})
    source.write_bytes(raw)
    receipt = archive.capture(source, "codex")
    queue_path = _enqueue(queue, _intent(archive, source, receipt, receipt, raw))
    envelope = queue_path.read_text(encoding="utf-8")
    assert "base_instructions" not in envelope and "secret" not in envelope

    def no_embedding(_text: str) -> list[float]:
        raise AssertionError("native session header must not request an embedding")

    assert drain_once(db_path=db, queue_dir=queue, log_path=log, embed_fn=no_embedding) == 1
    assert not queue_path.exists()
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (0,)
        row = conn.execute(
            "SELECT disposition,session_id,header_context_json FROM native_session_header_occurrences"
        ).fetchone()
    assert _vector_count(db) == 0
    assert row[0:2] == ("canonical", SESSION_A)
    assert json.loads(row[2]) == {"/payload/id": SESSION_A}


def test_repeat_conflict_invalid_and_commit_before_unlink_replay_are_idempotent(tmp_path: Path) -> None:
    db, queue = tmp_path / "db.sqlite", tmp_path / "queue"
    _new_db(db)
    archive = create_raw_source_archive(db, {})
    source = tmp_path / "session.jsonl"
    first = _header(SESSION_A)
    source.write_bytes(first)
    root = archive.capture(source, "codex")
    canonical_path = _enqueue(queue, _intent(archive, source, root, root, first))
    saved = canonical_path.read_bytes()
    assert drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "log") == 1

    # Simulate a committed transaction whose queue unlink was not durable.
    queue.mkdir(exist_ok=True)
    (queue / "replay-with-unrelated-name.jsonl").write_bytes(saved)
    assert drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "log") == 1
    assert len(_rows(db, "native_session_header_occurrences")) == 1

    with source.open("ab") as output:
        output.write(first)
    repeated_receipt = archive.capture(source, "codex")
    _enqueue(queue, _intent(archive, source, root, repeated_receipt, first, len(first)))
    assert drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "log") == 1

    conflict = _header(SESSION_B)
    conflict_start = source.stat().st_size
    with source.open("ab") as output:
        output.write(conflict)
    conflict_receipt = archive.capture(source, "codex")
    _enqueue(queue, _intent(archive, source, root, conflict_receipt, conflict, conflict_start))
    assert drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "log") == 1

    invalid = b'{"type":"session_meta","payload":{"id":"bad"}}\n'
    invalid_start = source.stat().st_size
    with source.open("ab") as output:
        output.write(invalid)
    invalid_receipt = archive.capture(source, "codex")
    _enqueue(queue, _intent(archive, source, root, invalid_receipt, invalid, invalid_start))
    assert drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "log") == 1

    with sqlite3.connect(db) as conn:
        dispositions = [
            row[0]
            for row in conn.execute(
                "SELECT disposition FROM native_session_header_occurrences ORDER BY record_start_offset"
            )
        ]
        status = conn.execute("SELECT status FROM native_source_generations").fetchone()[0]
    assert dispositions == ["canonical", "repeat", "conflict", "invalid"]
    assert status == "invalid"


@pytest.mark.parametrize("failure", ["missing", "corrupt", "wrong_archive"])
def test_source_receipt_failure_retains_queue_without_ack(tmp_path: Path, failure: str) -> None:
    db, queue, log = tmp_path / "db.sqlite", tmp_path / "queue", tmp_path / "drain.log"
    _new_db(db)
    archive = RawSourceArchive(tmp_path / "archive")
    source = tmp_path / "session.jsonl"
    raw = _header(SESSION_A)
    source.write_bytes(raw)
    receipt = archive.capture(source, "codex")
    queued = _enqueue(queue, _intent(archive, source, receipt, receipt, raw))
    drain_archive = archive
    if failure == "missing":
        Path(receipt["manifest_path"]).unlink()
    elif failure == "corrupt":
        manifest_path = Path(receipt["manifest_path"])
        manifest_path.chmod(0o600)
        manifest_path.write_text("{}", encoding="utf-8")
    else:
        drain_archive = RawSourceArchive(tmp_path / "other-archive")

    assert drain_once(db_path=db, queue_dir=queue, log_path=log, raw_archive=drain_archive) == 0
    assert queued.exists()
    assert "drain failed" in log.read_text(encoding="utf-8")
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='native_source_generations'"
        ).fetchone() in {(0,), (1,)}
        if conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='native_source_generations'"
        ).fetchone() == (1,):
            assert conn.execute("SELECT COUNT(*) FROM native_source_generations").fetchone() == (0,)


def test_strict_malformed_and_oversized_payloads_refuse_before_enqueue(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    archive = RawSourceArchive(tmp_path / "archive")
    source = tmp_path / "session.jsonl"
    raw = _header(SESSION_A)
    source.write_bytes(raw)
    receipt = archive.capture(source, "codex")
    valid = _intent(archive, source, receipt, receipt, raw)

    for key, value in (("provider", "claude"), ("watcher_generation", True), ("source_file", "relative")):
        malformed = dict(valid)
        malformed[key] = value
        with pytest.raises(NativeSessionHeaderEventError):
            _enqueue(queue, malformed)
    extra = dict(valid)
    extra["metadata"] = {"header_body": "x"}
    with pytest.raises(NativeSessionHeaderEventError):
        _enqueue(queue, extra)

    oversized_receipt = json.loads(json.dumps(receipt))
    oversized_receipt["source_stat"]["oversized"] = "x" * (MAX_EVENT_BYTES + 1)
    oversized = dict(valid)
    oversized["root_receipt"] = oversized_receipt
    with pytest.raises(NativeSessionHeaderEventError):
        _enqueue(queue, oversized)
    assert not list(queue.glob("*.jsonl")) if queue.exists() else True


def test_payload_kind_cannot_be_overwritten_by_caller(tmp_path: Path) -> None:
    archive = RawSourceArchive(tmp_path / "archive")
    source = tmp_path / "session.jsonl"
    raw = _header(SESSION_A)
    source.write_bytes(raw)
    receipt = archive.capture(source, "codex")
    kwargs = _intent(archive, source, receipt, receipt, raw)
    kwargs["kind"] = "store_memory"
    with pytest.raises(NativeSessionHeaderEventError):
        _enqueue(tmp_path / "queue", kwargs)

    db = tmp_path / "db.sqlite"
    _new_db(db)
    queued = _enqueue(tmp_path / "queue", _intent(archive, source, receipt, receipt, raw))
    event = json.loads(queued.read_text(encoding="utf-8"))
    event["kind"] = "unknown"
    queued.write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert (
        drain_once(
            db_path=db,
            queue_dir=tmp_path / "queue",
            log_path=tmp_path / "drain.log",
            raw_archive=archive,
        )
        == 0
    )
    assert queued.exists()


def test_transaction_rolls_back_all_headers_and_retains_batch(tmp_path: Path) -> None:
    db, queue, log = tmp_path / "db.sqlite", tmp_path / "queue", tmp_path / "drain.log"
    _new_db(db)
    archive = RawSourceArchive(tmp_path / "archive")
    source = tmp_path / "session.jsonl"
    first = _header(SESSION_A)
    second = _header(SESSION_B)
    source.write_bytes(first + second)
    receipt = archive.capture(source, "codex")
    from brainlayer.queue_io import enqueue_watcher_native_session_headers

    intents = []
    for raw, start in ((first, 0), (second, len(first))):
        item = _intent(archive, source, receipt, receipt, raw, start)
        item.pop("archive")
        intents.append(item)
    queued = enqueue_watcher_native_session_headers(intents, raw_archive=archive, queue_dir=queue)
    rows = [json.loads(line) for line in queued.read_text(encoding="utf-8").splitlines()]
    rows[1]["raw_record_hash"] = "0" * 64
    queued.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert drain_once(db_path=db, queue_dir=queue, log_path=log, raw_archive=archive) == 0
    assert queued.exists()
    with sqlite3.connect(db) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "native_source_generations" in tables:
            assert conn.execute("SELECT COUNT(*) FROM native_source_generations").fetchone() == (0,)
            assert conn.execute("SELECT COUNT(*) FROM native_session_header_occurrences").fetchone() == (0,)


def test_in_memory_db_without_explicit_archive_refuses_and_precompute_skips_header(tmp_path: Path) -> None:
    archive = RawSourceArchive(tmp_path / "archive")
    source = tmp_path / "session.jsonl"
    raw = _header(SESSION_A)
    source.write_bytes(raw)
    receipt = archive.capture(source, "codex")
    event = build_native_session_header_event(**_intent(archive, source, receipt, receipt, raw))
    called = False

    def embed(_text: str) -> list[float]:
        nonlocal called
        called = True
        return [0.0]

    assert _precompute_event_embeddings([event], embed) == {}
    assert called is False

    queued = _enqueue(tmp_path / "queue", _intent(archive, source, receipt, receipt, raw))
    assert (
        drain_once(
            db_path=Path(":memory:"),
            queue_dir=tmp_path / "queue",
            log_path=tmp_path / "drain.log",
        )
        == 0
    )
    assert queued.exists()
