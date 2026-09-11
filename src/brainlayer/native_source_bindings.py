"""Verified bindings from native response records to provider session headers.

This module is deliberately a producer-side boundary.  The ordinary watcher
source recorder keeps declared context only; it never writes these columns.
Only bytes read from an injected :class:`RawSourceArchive` can create a typed
assistant binding.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .native_session_headers import (
    NativeHeaderCatalogError,
    reserve_native_source_generation,
)
from .raw_source_archive import RawArchiveError, RawSourceArchive
from .source_records import ensure_chunk_source_records

MAX_NATIVE_RECORD_BYTES = 256 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")


class NativeSourceBindingError(NativeHeaderCatalogError):
    """A physical native source record could not be safely bound."""


class NativeSourceBindingConflict(NativeSourceBindingError):
    """An existing source row is already bound to a different identity."""


@dataclass(frozen=True)
class NativeSourceBindingResult:
    record_key: str
    canonical_chunk_id: str
    generation_key: str
    provider: str
    watcher_generation: int
    record_role: str
    occurrence_key: str | None
    record_manifest_sha256: str


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@contextmanager
def _savepoint(conn: Any) -> Iterator[None]:
    conn.execute("SAVEPOINT native_source_binding")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK TO native_source_binding")
        conn.execute("RELEASE native_source_binding")
        raise
    conn.execute("RELEASE native_source_binding")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def native_response_record_role(raw: bytes) -> str:
    """Return ``assistant`` only for a strict native Codex message record.

    The role is parsed from the archived bytes.  No normalized metadata is
    accepted as an alternative authority.
    """
    if not isinstance(raw, bytes) or len(raw) > MAX_NATIVE_RECORD_BYTES:
        raise NativeSourceBindingError("native record exceeds byte cap")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"nonfinite JSON value: {token}")),
        )
    except (UnicodeDecodeError, ValueError, TypeError, OverflowError) as error:
        raise NativeSourceBindingError(f"native response record is not strict JSON: {error}") from error
    if not isinstance(value, dict) or value.get("type") != "response_item":
        raise NativeSourceBindingError("native record is not a response_item")
    payload = value.get("payload")
    if not isinstance(payload, dict) or payload.get("type") != "message":
        raise NativeSourceBindingError("native response_item is not a message")
    if payload.get("role") != "assistant":
        raise NativeSourceBindingError("native response_item role is not assistant")
    return "assistant"


def _record_key(canonical_chunk_id: str, source_file: str, start: int, end: int, digest: str) -> str:
    return hashlib.sha256(
        json.dumps([canonical_chunk_id, source_file, start, end, digest], separators=(",", ":")).encode()
    ).hexdigest()


def _read_verified_record(
    archive: RawSourceArchive,
    receipt: dict[str, Any],
    *,
    start: int,
    end: int,
    digest: str,
) -> bytes:
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end <= start
        or end - start > MAX_NATIVE_RECORD_BYTES
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(receipt.get("size")) is not int
        or end > receipt["size"]
    ):
        raise NativeSourceBindingError("invalid native record bounds or digest")
    try:
        with archive.open_snapshot(receipt) as snapshot:
            if start:
                snapshot.seek(start - 1)
                if snapshot.read(1) != b"\n":
                    raise NativeSourceBindingError("native record does not begin at an LF boundary")
            snapshot.seek(start)
            raw = snapshot.read(end - start)
    except RawArchiveError as error:
        raise NativeSourceBindingError("native record receipt verification failed") from error
    if not raw.endswith(b"\n") or hashlib.sha256(raw).hexdigest() != digest:
        raise NativeSourceBindingError("native record bytes do not match LF bounds and digest")
    return raw


def _source_row(conn: Any, record_key: str) -> tuple[Any, ...] | None:
    return conn.execute(
        "SELECT record_key,canonical_chunk_id,source_file,record_start_offset,record_end_offset,"
        "raw_record_hash,native_generation_key,native_provider,native_watcher_generation,"
        "native_record_role,native_occurrence_key,native_record_manifest_sha256 "
        "FROM chunk_source_records WHERE record_key=?",
        (record_key,),
    ).fetchone()


def _canonical_header_occurrence(conn: Any, generation_key: str, assistant_start: int) -> str | None:
    row = conn.execute(
        "SELECT o.occurrence_key "
        "FROM native_session_header_occurrences AS o "
        "JOIN native_source_generations AS g ON g.generation_key=o.generation_key "
        "WHERE o.generation_key=? AND o.disposition='canonical' AND g.status='active' "
        "AND o.record_end_offset<=? ORDER BY o.record_end_offset, o.occurrence_key LIMIT 1",
        (generation_key, assistant_start),
    ).fetchone()
    return str(row[0]) if row is not None else None


def bind_native_source_record(
    conn: Any,
    archive: RawSourceArchive,
    *,
    canonical_chunk_id: str,
    source_file: str,
    provider: str = "codex",
    watcher_generation: int,
    root_receipt: dict[str, Any],
    occurrence_receipt: dict[str, Any],
    record_start_offset: int,
    record_end_offset: int,
    raw_record_hash: str,
    record_key: str | None = None,
) -> NativeSourceBindingResult:
    """Attach one existing source row to a verified assistant occurrence.

    The caller remains responsible for the surrounding transaction.  This
    function uses a savepoint and never commits.  A missing or malformed
    record, a receipt mismatch, or a competing binding leaves the source row
    untouched.
    """
    if not isinstance(canonical_chunk_id, str) or not canonical_chunk_id:
        raise NativeSourceBindingError("canonical_chunk_id must be a nonempty string")
    if not isinstance(source_file, str) or source_file != str(Path(source_file).absolute()):
        raise NativeSourceBindingError("source_file must be absolute")
    if provider != "codex":
        raise NativeSourceBindingError("native source binding provider must be codex")
    expected_key = _record_key(
        canonical_chunk_id, source_file, record_start_offset, record_end_offset, raw_record_hash
    )
    if record_key is None:
        record_key = expected_key
    if record_key != expected_key:
        raise NativeSourceBindingError("record_key does not match source-record identity")

    ensure_chunk_source_records(conn)
    with _savepoint(conn):
        row = _source_row(conn, record_key)
        if row is None:
            raise NativeSourceBindingError("source record does not exist")
        if tuple(row[:6]) != (
            record_key,
            canonical_chunk_id,
            source_file,
            record_start_offset,
            record_end_offset,
            raw_record_hash,
        ):
            raise NativeSourceBindingError("source record physical identity differs")

        generation_key = reserve_native_source_generation(
            conn,
            archive,
            provider=provider,
            source_file=source_file,
            watcher_generation=watcher_generation,
            root_receipt=root_receipt,
            current_receipt=occurrence_receipt,
        )
        raw = _read_verified_record(
            archive,
            occurrence_receipt,
            start=record_start_offset,
            end=record_end_offset,
            digest=raw_record_hash,
        )
        role = native_response_record_role(raw)
        occurrence_key = _canonical_header_occurrence(conn, generation_key, record_start_offset)
        manifest_sha256 = occurrence_receipt.get("manifest_sha256")
        if not isinstance(manifest_sha256, str) or _SHA256.fullmatch(manifest_sha256) is None:
            raise NativeSourceBindingError("occurrence receipt has no verified manifest hash")

        existing = row[6:]
        if any(value is not None for value in existing):
            stable = (existing[0], existing[1], existing[2], existing[3])
            desired = (generation_key, provider, watcher_generation, role)
            if stable != desired or existing[5] is None or (
                existing[4] is not None and existing[4] != occurrence_key
            ):
                raise NativeSourceBindingConflict("source record is already bound to a different identity")
            return NativeSourceBindingResult(
                record_key,
                canonical_chunk_id,
                generation_key,
                provider,
                watcher_generation,
                role,
                existing[4],
                manifest_sha256,
            )

        conn.execute(
            "UPDATE chunk_source_records SET native_generation_key=?,native_provider=?,"
            "native_watcher_generation=?,native_record_role=?,native_occurrence_key=?,"
            "native_record_manifest_sha256=? WHERE record_key=?",
            (
                generation_key,
                provider,
                watcher_generation,
                role,
                occurrence_key,
                manifest_sha256,
                record_key,
            ),
        )
        return NativeSourceBindingResult(
            record_key,
            canonical_chunk_id,
            generation_key,
            "codex",
            watcher_generation,
            role,
            occurrence_key,
            manifest_sha256,
        )


def lookup_native_source_binding(
    conn: Any,
    *,
    record_key: str,
    canonical_chunk_id: str | None = None,
) -> dict[str, Any] | None:
    """Return live, provider-recorded header evidence for one bound row.

    A row remains unbound when the header is absent, after a late conflict or
    invalid occurrence, when the canonical header is after the assistant, or
    when the source-row binding no longer matches the generation identity.
    """
    try:
        row = _source_row(conn, record_key)
    except Exception:
        return None
    if row is None or (canonical_chunk_id is not None and row[1] != canonical_chunk_id):
        return None
    if (
        row[6] is None
        or row[7] != "codex"
        or row[8] is None
        or row[9] != "assistant"
        or not isinstance(row[11], str)
        or _SHA256.fullmatch(row[11]) is None
    ):
        return None
    generation = conn.execute(
        "SELECT provider,source_file,watcher_generation,status,canonical_occurrence_key "
        "FROM native_source_generations WHERE generation_key=?",
        (row[6],),
    ).fetchone()
    if generation is None or generation[0] != row[7] or generation[1] != row[2] or generation[2] != row[8]:
        return None
    if generation[3] != "active":
        return None
    occurrence = conn.execute(
        "SELECT occurrence_key,record_start_offset,record_end_offset,raw_record_hash,"
        "session_id,header_context_json,disposition FROM native_session_header_occurrences "
        "WHERE occurrence_key=? AND generation_key=?",
        (generation[4], row[6]),
    ).fetchone()
    if occurrence is None or occurrence[6] != "canonical" or occurrence[2] > row[3]:
        return None
    if row[10] is not None and row[10] != occurrence[0]:
        return None
    if not isinstance(occurrence[5], str):
        return None
    try:
        context = json.loads(occurrence[5], parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(context, dict):
        return None
    return {
        "generation_key": row[6],
        "occurrence_key": occurrence[0],
        "provider": row[7],
        "source_file": row[2],
        "watcher_generation": row[8],
        "record_key": row[0],
        "canonical_chunk_id": row[1],
        "record_role": row[9],
        "session_id": occurrence[4],
        "header_context": context,
    }


def lookup_native_source_context(
    conn: Any,
    *,
    record_key: str,
    canonical_chunk_id: str | None = None,
) -> dict[str, Any] | None:
    """Return only provider-recorded header context, or ``None``."""
    binding = lookup_native_source_binding(conn, record_key=record_key, canonical_chunk_id=canonical_chunk_id)
    return None if binding is None else binding["header_context"]


# Names suitable for the next producer/Swift adapter without making the
# generic watcher recorder a trusted path.
record_verified_native_source_binding = bind_native_source_record
lookup_verified_native_source_context = lookup_native_source_context
