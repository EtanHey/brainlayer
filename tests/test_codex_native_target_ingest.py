import hashlib
import json
from pathlib import Path

import pytest

from brainlayer.ingest.codex import ingest_codex_session, parse_codex_session
from brainlayer.vector_store import VectorStore


def _write(path: Path, entries: list[dict]) -> list[bytes]:
    raw = [json.dumps(entry, separators=(",", ":")).encode() + b"\n" for entry in entries]
    path.write_bytes(b"".join(raw))
    return raw


def _meta(session_id: object = "session-1") -> dict:
    return {"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-01-01T00:00:00Z"}}


def _call(target: str = "thread-1") -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "mcp__codex_app__send_message_to_thread",
            "arguments": json.dumps({"threadId": target}),
            "call_id": "call-1",
        },
    }


def _assistant() -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "A sufficiently long response that is indexable while the nearby function call remains a distinct physical native record.",
                }
            ],
        },
    }


def _native_rows(db_path: Path):
    store = VectorStore(db_path)
    try:
        return list(
            store.conn.execute(
                "SELECT source_file,record_start_offset,record_end_offset,raw_record_hash,declared_targets_json FROM native_record_links"
            )
        )
    finally:
        store.close()


def test_call_only_ingest_catalogs_target_without_creating_content_chunk(tmp_path):
    source = tmp_path / "call-only.jsonl"
    raw = _write(source, [_meta(), _call()])
    db_path = tmp_path / "catalog.sqlite"
    assert ingest_codex_session(source, db_path=db_path) == 0
    rows = _native_rows(db_path)
    assert len(rows) == 1
    assert rows[0][0:3] == (str(source), len(raw[0]), len(raw[0]) + len(raw[1]))
    assert rows[0][3] == hashlib.sha256(raw[1]).hexdigest()
    assert json.loads(rows[0][4])[0]["target"] == "thread-1"


def test_parser_stays_pure_without_sink_and_sink_uses_exact_call_record(tmp_path):
    source = tmp_path / "parser.jsonl"
    raw = _write(source, [_meta(), _call(), _assistant()])
    baseline = list(parse_codex_session(source))
    sink: list[dict] = []
    assert list(parse_codex_session(source, native_record_sink=sink.append)) == baseline
    assert len(sink) == 1
    assert sink[0]["record_start_offset"] == len(raw[0])
    assert sink[0]["raw_record_hash"] == hashlib.sha256(raw[1]).hexdigest()


def test_mixed_call_and_content_keep_distinct_physical_records(tmp_path, monkeypatch):
    from brainlayer import index_new
    from brainlayer.embeddings import EmbeddedChunk

    source = tmp_path / "mixed.jsonl"
    raw = _write(source, [_meta(), _call(), _assistant()])
    db_path = tmp_path / "mixed.sqlite"
    monkeypatch.setattr(
        index_new,
        "embed_chunks",
        lambda chunks, on_progress=None: [EmbeddedChunk(chunk, [0.1] * 1024) for chunk in chunks],
    )
    assert ingest_codex_session(source, db_path=db_path) == 1
    store = VectorStore(db_path)
    try:
        target_offset = store.conn.execute("SELECT record_start_offset FROM native_record_links").fetchone()[0]
        content_offset = store.conn.execute("SELECT record_start_offset FROM chunk_source_records").fetchone()[0]
    finally:
        store.close()
    assert target_offset == len(raw[0])
    assert content_offset == len(raw[0]) + len(raw[1])
    assert target_offset != content_offset


def test_missing_session_dry_run_retry_and_admission_controls(tmp_path, monkeypatch, caplog):
    missing = tmp_path / "missing.jsonl"
    _write(missing, [_call()])
    db_path = tmp_path / "missing.sqlite"
    assert ingest_codex_session(missing, db_path=db_path) == 0
    assert "missing explicit session_meta.payload.id" in caplog.text
    store = VectorStore(db_path)
    try:
        assert store.conn.execute("SELECT name FROM sqlite_master WHERE name='native_record_links'").fetchone() is None
    finally:
        store.close()
    source = tmp_path / "retry.jsonl"
    _write(source, [_meta(), _call()])
    assert ingest_codex_session(source, db_path=db_path, dry_run=True) == 0
    assert ingest_codex_session(source, db_path=db_path) == 0
    assert ingest_codex_session(source, db_path=db_path) == 0
    assert len(_native_rows(db_path)) == 1
    monkeypatch.setattr("brainlayer.native_record_links.native_record_entry_is_admitted", lambda _: False)
    skipped = tmp_path / "skipped.jsonl"
    _write(skipped, [_meta("session-2"), _call("thread-2")])
    assert ingest_codex_session(skipped, db_path=db_path) == 0
    assert len(_native_rows(db_path)) == 1


def test_catalog_failure_rolls_back_and_precedes_content_indexing(tmp_path, monkeypatch):
    source = tmp_path / "failure.jsonl"
    _write(source, [_meta(), _call(), _assistant()])
    db_path = tmp_path / "failure.sqlite"
    monkeypatch.setattr(
        "brainlayer.native_record_links.catalog_native_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("catalog failure")),
    )
    with pytest.raises(RuntimeError, match="catalog failure"):
        ingest_codex_session(source, db_path=db_path)
    store = VectorStore(db_path)
    try:
        assert store.conn.execute("SELECT name FROM sqlite_master WHERE name='native_record_links'").fetchone() is None
        assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (0,)
    finally:
        store.close()


def test_raw_attribution_exclusion_and_oversized_call_id_are_bounded(tmp_path):
    excluded = tmp_path / "excluded.jsonl"
    entry = _call()
    entry["attributionAgent"] = "brain-worker"
    _write(excluded, [_meta(), entry])
    excluded_db = tmp_path / "excluded.sqlite"
    assert ingest_codex_session(excluded, db_path=excluded_db) == 0
    store = VectorStore(excluded_db)
    try:
        assert store.conn.execute("SELECT name FROM sqlite_master WHERE name='native_record_links'").fetchone() is None
    finally:
        store.close()

    oversized = tmp_path / "oversized.jsonl"
    entry = _call()
    entry["payload"]["call_id"] = "x" * (40 * 1024)
    _write(oversized, [_meta(), entry])
    db_path = tmp_path / "oversized.sqlite"
    assert ingest_codex_session(oversized, db_path=db_path) == 0
    store = VectorStore(db_path)
    try:
        headers, diagnostics = store.conn.execute(
            "SELECT declared_headers_json,target_diagnostics_json FROM native_record_links"
        ).fetchone()
    finally:
        store.close()
    assert "/payload/call_id" not in json.loads(headers)
    assert "call ID" in json.loads(diagnostics)[0]


def test_more_than_one_batch_flushes_before_late_failure_without_indexing(tmp_path, monkeypatch):
    import brainlayer.native_record_links as links

    source = tmp_path / "late-failure.jsonl"
    _write(source, [_meta(), *[_call(f"thread-{i}") for i in range(101)], _assistant()])
    db_path = tmp_path / "late-failure.sqlite"
    original = links.catalog_native_record
    calls = 0

    def fail_on_101(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 101:
            raise RuntimeError("late catalog failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(links, "catalog_native_record", fail_on_101)
    with pytest.raises(RuntimeError, match="late catalog failure"):
        ingest_codex_session(source, db_path=db_path)
    store = VectorStore(db_path)
    try:
        assert store.conn.execute("SELECT COUNT(*) FROM native_record_links").fetchone() == (100,)
        assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (0,)
    finally:
        store.close()
