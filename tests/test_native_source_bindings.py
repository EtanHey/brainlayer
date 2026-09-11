from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from brainlayer.native_session_headers import catalog_native_session_header
from brainlayer.native_source_bindings import (
    NativeSourceBindingConflict,
    NativeSourceBindingError,
    bind_native_source_record,
    lookup_native_source_binding,
    lookup_native_source_context,
    native_response_record_role,
)
from brainlayer.raw_source_archive import RawSourceArchive
from brainlayer.source_records import ensure_chunk_source_records, record_watcher_source

SESSION_A = "01a08d1e-42f6-7093-83a4-1c1b160f4911"
SESSION_B = "01a08d1e-4373-7631-b908-1a465cb013d3"


def _header(session_id: str = SESSION_A, *, context: dict | None = None) -> bytes:
    payload = {"id": session_id}
    if context:
        payload.update(context)
    return (json.dumps({"type": "session_meta", "payload": payload}, separators=(",", ":")) + "\n").encode()


def _assistant(*, role: str = "assistant") -> bytes:
    return (
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "output_text", "text": "A bounded answer."}],
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO chunks VALUES ('chunk-1')")
    ensure_chunk_source_records(conn)
    return conn


def _archive(tmp_path: Path, raw: bytes) -> tuple[sqlite3.Connection, RawSourceArchive, Path, dict]:
    source = tmp_path / "codex.jsonl"
    source.write_bytes(raw)
    archive = RawSourceArchive(tmp_path / "archive", chunk_bytes=32, max_source_bytes=1024 * 1024)
    root = archive.capture(source, "codex")
    return _db(), archive, source, root


def _source_row(conn: sqlite3.Connection, source: Path, raw: bytes, start: int, end: int) -> str:
    digest = hashlib.sha256(raw).hexdigest()
    record_watcher_source(
        conn,
        canonical_chunk_id="chunk-1",
        source_file=str(source),
        metadata={
            "_line_start_offset": start,
            "_line_end_offset": end,
            "_raw_record_hash": digest,
            "record_context": {"role": "assistant", "declared": True},
        },
    )
    return conn.execute("SELECT record_key FROM chunk_source_records").fetchone()[0]


def _bind(conn, archive, source, root, current, raw, start):
    return bind_native_source_record(
        conn,
        archive,
        canonical_chunk_id="chunk-1",
        source_file=str(source),
        watcher_generation=7,
        root_receipt=root,
        occurrence_receipt=current,
        record_start_offset=start,
        record_end_offset=start + len(raw),
        raw_record_hash=hashlib.sha256(raw).hexdigest(),
    )


def _catalog_header(conn, archive, source, root, current, raw, start=0):
    return catalog_native_session_header(
        conn,
        archive,
        provider="codex",
        source_file=str(source),
        watcher_generation=7,
        root_receipt=root,
        occurrence_receipt=current,
        record_start_offset=start,
        record_end_offset=start + len(raw),
        raw_record_hash=hashlib.sha256(raw).hexdigest(),
    )


def test_assistant_after_header_binds_to_canonical_context(tmp_path: Path):
    header = _header(context={"agent_nickname": "Ada"})
    assistant = _assistant()
    conn, archive, source, root = _archive(tmp_path, header + assistant)
    key = _source_row(conn, source, assistant, len(header), len(header) + len(assistant))
    current = archive.capture(source, "codex")
    _catalog_header(conn, archive, source, root, current, header)

    result = _bind(conn, archive, source, root, current, assistant, len(header))
    assert result.record_role == "assistant"
    assert lookup_native_source_context(conn, record_key=key) == {
        "/payload/agent_nickname": "Ada",
        "/payload/id": SESSION_A,
    }


def test_assistant_before_header_stays_unknown_even_after_late_header(tmp_path: Path):
    assistant = _assistant()
    header = _header()
    conn, archive, source, root = _archive(tmp_path, assistant + header)
    key = _source_row(conn, source, assistant, 0, len(assistant))
    current = archive.capture(source, "codex")
    _bind(conn, archive, source, root, current, assistant, 0)
    assert lookup_native_source_context(conn, record_key=key) is None
    _catalog_header(conn, archive, source, root, current, header, len(assistant))
    assert lookup_native_source_context(conn, record_key=key) is None


def test_queue_order_reversal_binds_first_then_header_catalog_enables_lookup(tmp_path: Path):
    header = _header(context={"agent_path": "/root/a"})
    assistant = _assistant()
    conn, archive, source, root = _archive(tmp_path, header + assistant)
    key = _source_row(conn, source, assistant, len(header), len(header) + len(assistant))
    current = archive.capture(source, "codex")
    _bind(conn, archive, source, root, current, assistant, len(header))
    assert lookup_native_source_context(conn, record_key=key) is None
    _catalog_header(conn, archive, source, root, current, header)
    assert lookup_native_source_context(conn, record_key=key) == {
        "/payload/agent_path": "/root/a",
        "/payload/id": SESSION_A,
    }


def test_late_conflict_and_invalid_header_taint_existing_lookup(tmp_path: Path):
    header = _header()
    assistant = _assistant()
    conn, archive, source, root = _archive(tmp_path, header + assistant)
    key = _source_row(conn, source, assistant, len(header), len(header) + len(assistant))
    current = archive.capture(source, "codex")
    _bind(conn, archive, source, root, current, assistant, len(header))
    _catalog_header(conn, archive, source, root, current, header)
    assert lookup_native_source_context(conn, record_key=key) is not None

    conflict = _header(SESSION_B)
    source.write_bytes(header + assistant + conflict)
    latest = archive.capture(source, "codex")
    _catalog_header(conn, archive, source, root, latest, conflict, len(header) + len(assistant))
    assert lookup_native_source_context(conn, record_key=key) is None


def test_late_invalid_header_taints_existing_lookup(tmp_path: Path):
    header = _header()
    assistant = _assistant()
    conn, archive, source, root = _archive(tmp_path, header + assistant)
    key = _source_row(conn, source, assistant, len(header), len(header) + len(assistant))
    current = archive.capture(source, "codex")
    _bind(conn, archive, source, root, current, assistant, len(header))
    _catalog_header(conn, archive, source, root, current, header)
    assert lookup_native_source_context(conn, record_key=key) is not None

    invalid = b"not-json\n"
    source.write_bytes(header + assistant + invalid)
    latest = archive.capture(source, "codex")
    _catalog_header(conn, archive, source, root, latest, invalid, len(header) + len(assistant))
    assert lookup_native_source_context(conn, record_key=key) is None


@pytest.mark.parametrize(
    "raw",
    [
        b'{"type":"response_item","payload":{"type":"message","role":"assistant","role":"user"}}\n',
        b'{"type":"response_item","payload":{"type":"message","role":"assistant","x":NaN}}\n',
        b'{"type":"response_item","payload":{"type":"function_call","role":"assistant"}}\n',
    ],
)
def test_role_is_read_strictly_from_raw_bytes_and_bad_shapes_refuse(tmp_path: Path, raw: bytes):
    conn, archive, source, root = _archive(tmp_path, raw)
    _source_row(conn, source, raw, 0, len(raw))
    current = archive.capture(source, "codex")
    before = conn.execute("SELECT * FROM chunk_source_records").fetchone()
    with pytest.raises(NativeSourceBindingError):
        _bind(conn, archive, source, root, current, raw, 0)
    assert conn.execute("SELECT * FROM chunk_source_records").fetchone() == before


def test_declared_context_cannot_make_generic_record_assistant(tmp_path: Path):
    raw = b'{"type":"response_item","payload":{"type":"function_call"}}\n'
    conn, archive, source, root = _archive(tmp_path, raw)
    key = _source_row(conn, source, raw, 0, len(raw))
    current = archive.capture(source, "codex")
    with pytest.raises(NativeSourceBindingError):
        _bind(conn, archive, source, root, current, raw, 0)
    assert lookup_native_source_binding(conn, record_key=key) is None


def test_conflicting_rebind_rolls_back_and_multiple_rows_stay_scoped(tmp_path: Path):
    header = _header()
    assistant = _assistant()
    conn, archive, source, root = _archive(tmp_path, header + assistant + assistant)
    first_key = _source_row(conn, source, assistant, len(header), len(header) + len(assistant))
    second_raw_start = len(header) + len(assistant)
    record_watcher_source(
        conn,
        canonical_chunk_id="chunk-1",
        source_file=str(source),
        metadata={
            "_line_start_offset": second_raw_start,
            "_line_end_offset": second_raw_start + len(assistant),
            "_raw_record_hash": hashlib.sha256(assistant).hexdigest(),
            "record_context": {"role": "forged"},
        },
    )
    second_key = conn.execute(
        "SELECT record_key FROM chunk_source_records WHERE record_key<>?", (first_key,)
    ).fetchone()[0]
    current = archive.capture(source, "codex")
    _bind(conn, archive, source, root, current, assistant, len(header))
    bound_columns = conn.execute(
        "SELECT native_generation_key,native_provider,native_watcher_generation,"
        "native_record_role,native_occurrence_key,native_record_manifest_sha256 "
        "FROM chunk_source_records WHERE record_key=?",
        (first_key,),
    ).fetchone()
    with pytest.raises(NativeSourceBindingConflict):
        bind_native_source_record(
            conn,
            archive,
            canonical_chunk_id="chunk-1",
            source_file=str(source),
            watcher_generation=8,
            root_receipt=root,
            occurrence_receipt=current,
            record_start_offset=len(header),
            record_end_offset=len(header) + len(assistant),
            raw_record_hash=hashlib.sha256(assistant).hexdigest(),
            record_key=first_key,
        )
    assert lookup_native_source_binding(conn, record_key=first_key) is None
    assert conn.execute(
        "SELECT native_generation_key,native_provider,native_watcher_generation,"
        "native_record_role,native_occurrence_key,native_record_manifest_sha256 "
        "FROM chunk_source_records WHERE record_key=?",
        (first_key,),
    ).fetchone() == bound_columns
    assert conn.execute(
        "SELECT native_generation_key,native_provider,native_watcher_generation,"
        "native_record_role,native_occurrence_key,native_record_manifest_sha256 "
        "FROM chunk_source_records WHERE record_key=?",
        (second_key,),
    ).fetchone() == (None, None, None, None, None, None)
    _catalog_header(conn, archive, source, root, current, header)
    assert lookup_native_source_binding(conn, record_key=first_key) is not None
    assert lookup_native_source_binding(conn, record_key=second_key) is None


def test_migration_preserves_legacy_rows_and_generic_recorder_leaves_binding_null(tmp_path: Path):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO chunks VALUES ('chunk-1')")
    conn.execute(
        "CREATE TABLE chunk_source_records (record_key TEXT PRIMARY KEY, canonical_chunk_id TEXT NOT NULL, "
        "source_file TEXT NOT NULL, record_start_offset INTEGER NOT NULL, record_end_offset INTEGER NOT NULL, "
        "raw_record_hash TEXT NOT NULL, session_id TEXT, observed_at TEXT NOT NULL, "
        "record_context_json TEXT NOT NULL DEFAULT '[]')"
    )
    conn.execute(
        "INSERT INTO chunk_source_records VALUES (?,?,?,?,?,?,?,?,?)",
        ("legacy", "chunk-1", "legacy.jsonl", 0, 2, "a" * 64, "s", "now", '[{"declared":true}]'),
    )
    ensure_chunk_source_records(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(chunk_source_records)")}
    assert {
        "native_generation_key",
        "native_provider",
        "native_watcher_generation",
        "native_record_role",
        "native_occurrence_key",
    } <= columns
    assert (
        conn.execute("SELECT record_context_json FROM chunk_source_records WHERE record_key='legacy'").fetchone()[0]
        == '[{"declared":true}]'
    )
    record_watcher_source(
        conn,
        canonical_chunk_id="chunk-1",
        source_file="legacy.jsonl",
        metadata={
            "_line_start_offset": 0,
            "_line_end_offset": 2,
            "_raw_record_hash": "a" * 64,
            "generation_key": "forged",
            "role": "assistant",
        },
    )
    assert (
        conn.execute(
            "SELECT native_generation_key,native_record_role "
            "FROM chunk_source_records WHERE record_key='legacy'"
        ).fetchone()
        == (None, None)
    )


def test_public_role_parser_has_explicit_cap():
    with pytest.raises(NativeSourceBindingError, match="byte cap"):
        native_response_record_role(b" " * (256 * 1024 + 1))
