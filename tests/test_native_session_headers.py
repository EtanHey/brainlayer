from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from brainlayer.native_session_headers import (
    NativeHeaderCatalogError,
    NativeSourceGenerationConflict,
    catalog_native_session_header,
    ensure_native_session_headers_schema,
    native_source_generation_status,
)
from brainlayer.raw_source_archive import RawSourceArchive

SESSION_A = "01a08d1e-42f6-7093-83a4-1c1b160f4911"
SESSION_B = "01a08d1e-4373-7631-b908-1a465cb013d3"


def archive(tmp_path: Path) -> RawSourceArchive:
    return RawSourceArchive(tmp_path / "archive", chunk_bytes=32, max_source_bytes=1024 * 1024)


def header(session_id: str, **extra: object) -> bytes:
    payload = {"id": session_id, **extra}
    return (json.dumps({"type": "session_meta", "payload": payload}, separators=(",", ":")) + "\n").encode()


def catalog(
    conn: sqlite3.Connection,
    store: RawSourceArchive,
    source: Path,
    root: dict,
    current: dict,
    raw: bytes,
    start: int = 0,
    *,
    generation: int = 7,
):
    return catalog_native_session_header(
        conn,
        store,
        provider="codex",
        source_file=str(source),
        watcher_generation=generation,
        root_receipt=root,
        occurrence_receipt=current,
        record_start_offset=start,
        record_end_offset=start + len(raw),
        raw_record_hash=hashlib.sha256(raw).hexdigest(),
    )


def test_prefix_append_keeps_generation_deduplicates_occurrence_and_late_conflict_taints_old(
    tmp_path: Path,
) -> None:
    source = tmp_path / "session.jsonl"
    first = header(SESSION_A, agent_path="/root/a", agent_nickname="Ada")
    source.write_bytes(first)
    store = archive(tmp_path)
    root = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")

    canonical = catalog(conn, store, source, root, root, first)
    assert canonical.disposition == "canonical"
    assert canonical.generation_status == "active"
    with source.open("ab") as handle:
        handle.write(b'{"type":"response_item","payload":{"role":"assistant"}}\n')
    appended = store.capture(source, "codex")
    repeated = catalog(conn, store, source, root, appended, first)
    assert repeated.generation_key == canonical.generation_key
    assert repeated.occurrence_key == canonical.occurrence_key
    assert repeated.inserted is False
    assert conn.execute("SELECT COUNT(*) FROM native_session_header_occurrences").fetchone() == (1,)

    conflicting = header(SESSION_B, agent_path="/root/b", agent_nickname="Bea")
    conflict_start = source.stat().st_size
    with source.open("ab") as handle:
        handle.write(conflicting)
    latest = store.capture(source, "codex")
    conflict = catalog(conn, store, source, root, latest, conflicting, conflict_start)
    assert conflict.disposition == "conflict"
    assert native_source_generation_status(conn, canonical.generation_key)["status"] == "conflicted"
    assert conn.execute("SELECT COUNT(*) FROM native_session_header_occurrences").fetchone() == (2,)


def test_minimal_header_is_known_session_without_invented_emitter_and_body_is_not_retained(
    tmp_path: Path,
) -> None:
    source = tmp_path / "minimal.jsonl"
    raw = header(SESSION_A, base_instructions="do not retain", other={"body": "secret"})
    source.write_bytes(raw)
    store = archive(tmp_path)
    receipt = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")

    result = catalog(conn, store, source, receipt, receipt, raw)
    stored = conn.execute("SELECT session_id,header_context_json FROM native_session_header_occurrences").fetchone()
    assert result.session_id == SESSION_A
    assert json.loads(stored[1]) == {"/payload/id": SESSION_A}
    assert "base_instructions" not in stored[1] and "secret" not in stored[1]


@pytest.mark.parametrize(
    "raw,error",
    [
        (
            b'{"type":"session_meta","payload":{"id":"' + SESSION_A.encode() + b'","source":{"subagent":"bad"}}}\n',
            "invalid_subagent",
        ),
        (
            b'{"type":"session_meta","payload":{"id":"'
            + SESSION_A.encode()
            + b'","id":"'
            + SESSION_B.encode()
            + b'"}}\n',
            "duplicate JSON key",
        ),
        (
            b'{"type":"session_meta","payload":{"id":"' + SESSION_A.encode() + b'","depth":NaN}}\n',
            "nonfinite",
        ),
    ],
)
def test_invalid_metadata_is_retained_as_diagnostic_unknown(
    tmp_path: Path,
    raw: bytes,
    error: str,
) -> None:
    source = tmp_path / f"invalid-{hashlib.sha256(raw).hexdigest()}.jsonl"
    source.write_bytes(raw)
    store = archive(tmp_path)
    receipt = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")

    result = catalog(conn, store, source, receipt, receipt, raw)
    row = conn.execute(
        "SELECT disposition,header_context_json,validation_error FROM native_session_header_occurrences"
    ).fetchone()
    assert result.generation_status == "invalid"
    assert row[0] == "invalid" and row[1] is None and error in row[2]


def test_nonprefix_rewrite_cannot_extend_existing_generation(tmp_path: Path) -> None:
    source = tmp_path / "rewrite.jsonl"
    first = header(SESSION_A)
    source.write_bytes(first)
    store = archive(tmp_path)
    root = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")
    catalog(conn, store, source, root, root, first)
    rewritten = header(SESSION_B)
    source.write_bytes(rewritten)
    divergent = store.capture(source, "codex")

    with pytest.raises(NativeSourceGenerationConflict, match="does not extend"):
        catalog(conn, store, source, root, divergent, rewritten)

    assert conn.execute("SELECT COUNT(*) FROM native_source_generations").fetchone() == (1,)
    assert conn.execute("SELECT COUNT(*) FROM native_session_header_occurrences").fetchone() == (1,)


@pytest.mark.parametrize("mutation", ["hash", "midline"])
def test_physical_hash_and_lf_bounds_refuse_without_catalog_rows(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / f"{mutation}.jsonl"
    raw = header(SESSION_A)
    source.write_bytes(raw)
    store = archive(tmp_path)
    receipt = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")
    kwargs = {"start": 1} if mutation == "midline" else {}
    digest = "0" * 64 if mutation == "hash" else hashlib.sha256(raw[1:]).hexdigest()

    with pytest.raises(NativeHeaderCatalogError):
        catalog_native_session_header(
            conn,
            store,
            provider="codex",
            source_file=str(source),
            watcher_generation=0,
            root_receipt=receipt,
            occurrence_receipt=receipt,
            record_start_offset=kwargs.get("start", 0),
            record_end_offset=len(raw),
            raw_record_hash=digest,
        )

    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='native_source_generations'"
    ).fetchone() == (1,)
    assert conn.execute("SELECT COUNT(*) FROM native_source_generations").fetchone() == (0,)


def test_occurrence_and_generation_status_change_share_one_transaction(tmp_path: Path) -> None:
    source = tmp_path / "atomic.jsonl"
    raw = header(SESSION_A)
    source.write_bytes(raw)
    store = archive(tmp_path)
    receipt = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")
    ensure_native_session_headers_schema(conn)
    conn.execute(
        "CREATE TRIGGER stop_header BEFORE INSERT ON native_session_header_occurrences "
        "BEGIN SELECT RAISE(ABORT,'stop'); END"
    )

    with pytest.raises(sqlite3.IntegrityError, match="stop"):
        catalog(conn, store, source, receipt, receipt, raw)

    assert conn.execute("SELECT COUNT(*) FROM native_source_generations").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM native_session_header_occurrences").fetchone() == (0,)
