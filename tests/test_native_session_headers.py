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


def legacy_receipt(store: RawSourceArchive, source: Path, raw: bytes) -> dict:
    chunks = []
    for start in range(0, len(raw), store.chunk_bytes):
        data = raw[start : start + store.chunk_bytes]
        digest = hashlib.sha256(data).hexdigest()
        store._publish_chunk(digest, data)
        chunks.append({"sha256": digest, "size": len(data)})
    info = source.stat()
    source_stat = {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": len(raw),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }
    document = {
        "format_version": 1,
        "provider": "codex",
        "source_path": str(source),
        "source_stat": source_stat,
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "chunks": chunks,
    }
    encoded = store._canonical_manifest(document)
    digest = hashlib.sha256(encoded).hexdigest()
    path = store.manifests / f"{digest}.json"
    store._publish_bytes(path, encoded, prefix=".legacy-")
    return {
        "manifest_sha256": digest,
        "manifest_path": str(path),
        "source_path": str(source),
        "provider": "codex",
        "sha256": document["sha256"],
        "size": len(raw),
        "source_stat": source_stat,
    }


def test_prefix_append_keeps_generation_deduplicates_occurrence_and_late_conflict_taints_old(
    tmp_path: Path,
) -> None:
    source = tmp_path / "session.jsonl"
    first = header(SESSION_A, agent_path="/root/a", agent_nickname="Ada")
    source.write_bytes(first)
    store = archive(tmp_path)
    root = store.capture(source, "codex")
    assert root["format_version"] == 2
    conn = sqlite3.connect(":memory:")

    canonical = catalog(conn, store, source, root, root, first)
    assert canonical.disposition == "canonical"
    assert canonical.generation_status == "active"
    with source.open("ab") as handle:
        handle.write(b'{"type":"response_item","payload":{"role":"assistant"}}\n')
    appended = store.capture(source, "codex")
    assert appended["format_version"] == 2
    assert store.is_prefix(root, appended) is True
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


def test_v1_root_and_current_v2_append_bridge_catalogs_one_generation(tmp_path: Path) -> None:
    source = tmp_path / "mixed-format.jsonl"
    raw = header(SESSION_A)
    source.write_bytes(raw)
    store = archive(tmp_path)
    root = legacy_receipt(store, source, raw)
    with source.open("ab") as output:
        output.write(b'{"type":"response_item"}\n')
    current = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")

    result = catalog(conn, store, source, root, current, raw)

    assert "format_version" not in root
    assert current["format_version"] == 2
    assert store.is_prefix(root, current) is True
    assert result.disposition == "canonical"
    stored = native_source_generation_status(conn, result.generation_key)
    assert stored["root_manifest_sha256"] == root["manifest_sha256"]
    assert stored["current_manifest_sha256"] == current["manifest_sha256"]


@pytest.mark.parametrize("claimed", [True, 2.0, 1, 3, None])
def test_catalog_rejects_nonexact_v2_format_version(tmp_path: Path, claimed: object) -> None:
    source = tmp_path / "format.jsonl"
    raw = header(SESSION_A)
    source.write_bytes(raw)
    store = archive(tmp_path)
    receipt = store.capture(source, "codex")
    receipt["format_version"] = claimed

    with pytest.raises(NativeHeaderCatalogError, match="format version"):
        catalog(sqlite3.connect(":memory:"), store, source, receipt, receipt, raw)


@pytest.mark.parametrize("kind", ["extra", "missing"])
def test_catalog_rejects_unknown_or_missing_v2_receipt_keys(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "shape.jsonl"
    raw = header(SESSION_A)
    source.write_bytes(raw)
    store = archive(tmp_path)
    receipt = store.capture(source, "codex")
    if kind == "extra":
        receipt["unknown"] = "value"
    else:
        receipt.pop("size")

    with pytest.raises(NativeHeaderCatalogError, match="invalid shape"):
        catalog(sqlite3.connect(":memory:"), store, source, receipt, receipt, raw)


def test_catalog_rejects_v1_receipt_labeled_as_v2_by_archive_validation(tmp_path: Path) -> None:
    source = tmp_path / "spoof.jsonl"
    raw = header(SESSION_A)
    source.write_bytes(raw)
    store = archive(tmp_path)
    receipt = legacy_receipt(store, source, raw)
    receipt["format_version"] = 2

    with pytest.raises(NativeHeaderCatalogError, match="verification failed"):
        catalog(sqlite3.connect(":memory:"), store, source, receipt, receipt, raw)


def test_catalog_rejects_v2_receipt_with_format_label_removed(tmp_path: Path) -> None:
    source = tmp_path / "stripped.jsonl"
    raw = header(SESSION_A)
    source.write_bytes(raw)
    store = archive(tmp_path)
    receipt = store.capture(source, "codex")
    receipt.pop("format_version")

    with pytest.raises(NativeHeaderCatalogError, match="verification failed"):
        catalog(sqlite3.connect(":memory:"), store, source, receipt, receipt, raw)


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


def test_malformed_preexisting_schema_refuses_before_catalog_data_write(tmp_path: Path) -> None:
    raw = header(SESSION_A)
    source = tmp_path / "malformed.jsonl"
    store = archive(tmp_path)
    source.write_bytes(raw)
    receipt = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE native_source_generations (generation_key TEXT PRIMARY KEY)")
    conn.commit()

    with pytest.raises(NativeHeaderCatalogError, match="schema is incomplete"):
        catalog(conn, store, source, receipt, receipt, raw)

    assert conn.execute("SELECT COUNT(*) FROM native_source_generations").fetchone() == (0,)
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='native_session_header_occurrences'"
    ).fetchone() == (0,)


def test_wrong_preexisting_index_refuses_before_catalog_data_write(tmp_path: Path) -> None:
    raw = header(SESSION_A)
    source = tmp_path / "wrong-index.jsonl"
    source.write_bytes(raw)
    store = archive(tmp_path)
    receipt = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")
    ensure_native_session_headers_schema(conn)
    conn.execute("DROP INDEX idx_native_source_generations_source")
    conn.execute("CREATE INDEX idx_native_source_generations_source ON native_source_generations(status)")
    conn.commit()

    with pytest.raises(NativeHeaderCatalogError, match="index columns mismatch"):
        catalog(conn, store, source, receipt, receipt, raw)

    assert conn.execute("SELECT COUNT(*) FROM native_source_generations").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM native_session_header_occurrences").fetchone() == (0,)


@pytest.mark.parametrize("predicate", ["'CANONICAL'", "'can onical'"])
def test_schema_requires_canonical_unique_partial_index_predicate(tmp_path: Path, predicate: str) -> None:
    raw = header(SESSION_A)
    source = tmp_path / "canonical-index.jsonl"
    source.write_bytes(raw)
    store = archive(tmp_path)
    receipt = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")
    ensure_native_session_headers_schema(conn)
    conn.execute("DROP INDEX idx_native_session_headers_canonical")
    conn.execute(
        "CREATE UNIQUE INDEX idx_native_session_headers_canonical "
        f"ON native_session_header_occurrences(generation_key) WHERE disposition = {predicate}"
    )
    conn.commit()

    with pytest.raises(NativeHeaderCatalogError, match="index predicate mismatch"):
        catalog(conn, store, source, receipt, receipt, raw)


def test_schema_preserves_check_constraint_literal_bytes(tmp_path: Path) -> None:
    raw = header(SESSION_A)
    source = tmp_path / "check-literal.jsonl"
    source.write_bytes(raw)
    store = archive(tmp_path)
    receipt = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")
    ensure_native_session_headers_schema(conn)
    generation_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='native_source_generations'"
    ).fetchone()[0]
    occurrence_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='native_session_header_occurrences'"
    ).fetchone()[0]
    for index in (
        "idx_native_source_generations_source",
        "idx_native_session_headers_generation",
        "idx_native_session_headers_canonical",
    ):
        conn.execute(f"DROP INDEX {index}")
    conn.execute("DROP TABLE native_session_header_occurrences")
    conn.execute("DROP TABLE native_source_generations")
    altered_generation_sql = generation_sql.replace(
        "status IN ('active','conflicted','invalid')",
        "status IN ('ACTIVE','conflicted','invalid')",
    )
    assert altered_generation_sql != generation_sql
    conn.execute(altered_generation_sql)
    conn.execute(occurrence_sql)
    conn.execute(
        "CREATE INDEX idx_native_source_generations_source "
        "ON native_source_generations(provider,source_file,watcher_generation,status,generation_key)"
    )
    conn.execute(
        "CREATE INDEX idx_native_session_headers_generation "
        "ON native_session_header_occurrences(generation_key,disposition,record_start_offset,occurrence_key)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX idx_native_session_headers_canonical "
        "ON native_session_header_occurrences(generation_key) WHERE disposition = 'canonical'"
    )

    with pytest.raises(NativeHeaderCatalogError, match="generation constraints mismatch"):
        catalog(conn, store, source, receipt, receipt, raw)

    assert conn.execute("SELECT COUNT(*) FROM native_source_generations").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM native_session_header_occurrences").fetchone() == (0,)


def test_older_prefix_replay_does_not_downgrade_current_receipt(tmp_path: Path) -> None:
    source = tmp_path / "older-prefix.jsonl"
    raw = header(SESSION_A)
    source.write_bytes(raw)
    store = archive(tmp_path)
    root = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")
    first = catalog(conn, store, source, root, root, raw)
    with source.open("ab") as output:
        output.write(b'{"type":"response_item"}\n')
    latest = store.capture(source, "codex")

    catalog(conn, store, source, root, latest, raw)
    current_after_latest = native_source_generation_status(conn, first.generation_key)["current_manifest_sha256"]
    catalog(conn, store, source, root, root, raw)
    current_after_old = native_source_generation_status(conn, first.generation_key)["current_manifest_sha256"]

    assert current_after_latest == latest["manifest_sha256"]
    assert current_after_old == current_after_latest


def test_late_invalid_header_taints_generation_and_preserves_canonical(tmp_path: Path) -> None:
    source = tmp_path / "late-invalid.jsonl"
    raw = header(SESSION_A)
    source.write_bytes(raw)
    store = archive(tmp_path)
    root = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")
    first = catalog(conn, store, source, root, root, raw)
    invalid = b"not-json\n"
    start = len(raw)
    with source.open("ab") as output:
        output.write(invalid)
    latest = store.capture(source, "codex")

    second = catalog(conn, store, source, root, latest, invalid, start)
    status = native_source_generation_status(conn, first.generation_key)
    assert second.disposition == "invalid"
    assert status["status"] == "invalid"
    assert status["canonical_occurrence_key"] == first.occurrence_key
    assert conn.execute("SELECT COUNT(*) FROM native_session_header_occurrences").fetchone() == (2,)


def test_distinct_offset_same_context_is_repeat_and_active(tmp_path: Path) -> None:
    source = tmp_path / "repeat.jsonl"
    raw = header(SESSION_A)
    source.write_bytes(raw)
    store = archive(tmp_path)
    root = store.capture(source, "codex")
    conn = sqlite3.connect(":memory:")
    first = catalog(conn, store, source, root, root, raw)
    start = len(raw)
    with source.open("ab") as output:
        output.write(raw)
    latest = store.capture(source, "codex")

    second = catalog(conn, store, source, root, latest, raw, start)
    status = native_source_generation_status(conn, first.generation_key)
    assert second.disposition == "repeat"
    assert second.inserted is True
    assert status["status"] == "active"
    assert conn.execute("SELECT COUNT(*) FROM native_session_header_occurrences").fetchone() == (2,)
