"""Independent physical catalog for provider session headers.

Stored fields are provider-recorded metadata. They do not authenticate a sender,
human, or original author.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .raw_source_archive import RawSourceArchive

MAX_HEADER_RECORD_BYTES = 256 * 1024
MAX_CONTEXT_BYTES = 32 * 1024
MAX_RECEIPT_BYTES = 64 * 1024
MAX_SCALAR_BYTES = 4 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECEIPT_KEYS = {
    "manifest_sha256",
    "manifest_path",
    "source_path",
    "provider",
    "sha256",
    "size",
    "source_stat",
}
_PAYLOAD_STRINGS = ("id", "session_id", "parent_thread_id", "agent_path", "agent_nickname", "agent_role")
_SPAWN_STRINGS = ("parent_thread_id", "agent_path", "agent_nickname", "agent_role")


class NativeHeaderCatalogError(ValueError):
    pass


class NativeSourceGenerationConflict(NativeHeaderCatalogError):
    pass


@dataclass(frozen=True)
class NativeHeaderCatalogResult:
    generation_key: str
    occurrence_key: str
    disposition: str
    generation_status: str
    session_id: str | None
    inserted: bool


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@contextmanager
def _savepoint(conn: Any) -> Iterator[None]:
    conn.execute("SAVEPOINT native_header_catalog")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK TO native_header_catalog")
        conn.execute("RELEASE native_header_catalog")
        raise
    conn.execute("RELEASE native_header_catalog")


def ensure_native_session_headers_schema(conn: Any) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS native_source_generations (
        generation_key TEXT PRIMARY KEY, provider TEXT NOT NULL, source_file TEXT NOT NULL,
        watcher_generation INTEGER NOT NULL, root_manifest_sha256 TEXT NOT NULL,
        root_receipt_json TEXT NOT NULL, current_manifest_sha256 TEXT NOT NULL,
        current_receipt_json TEXT NOT NULL, status TEXT NOT NULL,
        canonical_occurrence_key TEXT, first_seen_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        UNIQUE(provider, source_file, watcher_generation, root_manifest_sha256),
        CHECK(watcher_generation >= 0), CHECK(status IN ('active','conflicted','invalid')))"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS native_session_header_occurrences (
        occurrence_key TEXT PRIMARY KEY, generation_key TEXT NOT NULL,
        disposition TEXT NOT NULL, manifest_sha256 TEXT NOT NULL,
        record_start_offset INTEGER NOT NULL, record_end_offset INTEGER NOT NULL,
        raw_record_hash TEXT NOT NULL, session_id TEXT, header_context_json TEXT,
        header_context_sha256 TEXT, validation_error TEXT, observed_at TEXT NOT NULL,
        UNIQUE(generation_key, record_start_offset, record_end_offset, raw_record_hash),
        FOREIGN KEY(generation_key) REFERENCES native_source_generations(generation_key),
        CHECK(disposition IN ('canonical','repeat','conflict','invalid')))"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_native_source_generations_source "
        "ON native_source_generations(provider,source_file,watcher_generation,status,generation_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_native_session_headers_generation "
        "ON native_session_header_occurrences(generation_key,disposition,record_start_offset,occurrence_key)"
    )


def _verified_receipt(
    archive: RawSourceArchive,
    receipt: Any,
    provider: str,
    source_file: str,
) -> str:
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise NativeHeaderCatalogError("raw archive receipt has invalid shape")
    manifest_sha = receipt.get("manifest_sha256")
    expected_manifest = archive.manifests / f"{manifest_sha}.json"
    if (
        not isinstance(manifest_sha, str)
        or _SHA256.fullmatch(manifest_sha) is None
        or receipt.get("provider") != provider
        or receipt.get("source_path") != source_file
        or receipt.get("manifest_path") != str(expected_manifest)
    ):
        raise NativeHeaderCatalogError("raw archive receipt identity mismatch")
    try:
        raw = _canonical(receipt)
    except (TypeError, ValueError) as error:
        raise NativeHeaderCatalogError("raw archive receipt is not canonical JSON data") from error
    if len(raw.encode()) > MAX_RECEIPT_BYTES:
        raise NativeHeaderCatalogError("raw archive receipt exceeds byte cap")
    with archive.open_snapshot(receipt):
        pass
    return raw


def _generation_key(provider: str, source_file: str, watcher_generation: int, root_manifest: str) -> str:
    return hashlib.sha256(
        _canonical(["native-source-generation-v1", provider, source_file, watcher_generation, root_manifest]).encode()
    ).hexdigest()


def _reserve(
    conn: Any,
    archive: RawSourceArchive,
    *,
    provider: str,
    source_file: str,
    watcher_generation: int,
    root_receipt: dict[str, Any],
    current_receipt: dict[str, Any],
) -> str:
    if provider != "codex":
        raise NativeHeaderCatalogError("native session header provider must be codex")
    if source_file != str(Path(source_file).absolute()):
        raise NativeHeaderCatalogError("source_file must be absolute")
    if type(watcher_generation) is not int or watcher_generation < 0:
        raise NativeHeaderCatalogError("watcher_generation must be a nonnegative integer")
    root_json = _verified_receipt(archive, root_receipt, provider, source_file)
    current_json = _verified_receipt(archive, current_receipt, provider, source_file)
    if not archive.is_prefix(root_receipt, current_receipt):
        raise NativeSourceGenerationConflict("current receipt does not extend generation root")
    key = _generation_key(provider, source_file, watcher_generation, root_receipt["manifest_sha256"])
    row = conn.execute(
        "SELECT provider,source_file,watcher_generation,root_manifest_sha256,root_receipt_json,"
        "current_receipt_json FROM native_source_generations WHERE generation_key=?",
        (key,),
    ).fetchone()
    now = datetime.now(timezone.utc).isoformat()
    if row is None:
        conn.execute(
            "INSERT INTO native_source_generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                provider,
                source_file,
                watcher_generation,
                root_receipt["manifest_sha256"],
                root_json,
                current_receipt["manifest_sha256"],
                current_json,
                "active",
                None,
                now,
                now,
            ),
        )
        return key
    if tuple(row[:5]) != (provider, source_file, watcher_generation, root_receipt["manifest_sha256"], root_json):
        raise NativeSourceGenerationConflict("stored generation root differs")
    stored_current = json.loads(row[5])
    if current_json == row[5] or archive.is_prefix(current_receipt, stored_current):
        return key
    if not archive.is_prefix(stored_current, current_receipt):
        raise NativeSourceGenerationConflict("current receipt diverges from stored generation")
    conn.execute(
        "UPDATE native_source_generations SET current_manifest_sha256=?,current_receipt_json=?,updated_at=? "
        "WHERE generation_key=?",
        (current_receipt["manifest_sha256"], current_json, now, key),
    )
    return key


def reserve_native_source_generation(
    conn: Any,
    archive: RawSourceArchive,
    *,
    provider: str,
    source_file: str,
    watcher_generation: int,
    root_receipt: dict[str, Any],
    current_receipt: dict[str, Any],
) -> str:
    ensure_native_session_headers_schema(conn)
    with _savepoint(conn):
        return _reserve(
            conn,
            archive,
            provider=provider,
            source_file=source_file,
            watcher_generation=watcher_generation,
            root_receipt=root_receipt,
            current_receipt=current_receipt,
        )


def _record_bytes(
    archive: RawSourceArchive,
    receipt: dict[str, Any],
    start: int,
    end: int,
    digest: str,
) -> bytes:
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end <= start
        or end - start > MAX_HEADER_RECORD_BYTES
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(receipt.get("size")) is not int
        or end > receipt["size"]
    ):
        raise NativeHeaderCatalogError("invalid physical header bounds or digest")
    with archive.open_snapshot(receipt) as source:
        if start:
            source.seek(start - 1)
            if source.read(1) != b"\n":
                raise NativeHeaderCatalogError("header does not begin at an LF record boundary")
        source.seek(start)
        raw = source.read(end - start)
    if not raw.endswith(b"\n") or hashlib.sha256(raw).hexdigest() != digest:
        raise NativeHeaderCatalogError("header bytes do not match LF bounds and digest")
    return raw


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _context(raw: bytes) -> tuple[dict[str, Any] | None, str | None, str | None]:
    try:
        line = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite {value}")),
        )
        if not isinstance(line, dict) or line.get("type") != "session_meta":
            raise ValueError("not_session_meta")
        payload = line.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("invalid_payload")
        context: dict[str, Any] = {}

        def add_string(pointer: str, container: dict[str, Any], field: str) -> None:
            if field not in container or container[field] is None:
                return
            value = container[field]
            if not isinstance(value, str) or not value or "\x00" in value or len(value.encode()) > MAX_SCALAR_BYTES:
                raise ValueError(f"invalid_{field}")
            context[pointer] = value

        for field in _PAYLOAD_STRINGS:
            add_string(f"/payload/{field}", payload, field)
        session_id = context.get("/payload/id")
        if not isinstance(session_id, str) or str(uuid.UUID(session_id)) != session_id.lower():
            raise ValueError("invalid_session_id")
        source = payload.get("source")
        if source is not None:
            if not isinstance(source, dict):
                raise ValueError("invalid_source")
            subagent = source.get("subagent")
            if subagent is not None:
                if not isinstance(subagent, dict):
                    raise ValueError("invalid_subagent")
                spawn = subagent.get("thread_spawn")
                if spawn is not None:
                    if not isinstance(spawn, dict):
                        raise ValueError("invalid_thread_spawn")
                    for field in _SPAWN_STRINGS:
                        add_string(f"/payload/source/subagent/thread_spawn/{field}", spawn, field)
                    if "depth" in spawn:
                        depth = spawn["depth"]
                        if type(depth) is not int or not 0 <= depth <= 128:
                            raise ValueError("invalid_depth")
                        context["/payload/source/subagent/thread_spawn/depth"] = depth
        encoded = _canonical(context)
        if len(encoded.encode()) > MAX_CONTEXT_BYTES:
            raise ValueError("context_byte_cap")
        return context, None, session_id
    except (UnicodeDecodeError, ValueError, TypeError, OverflowError) as error:
        return None, str(error)[:128] or type(error).__name__, None


def catalog_native_session_header(
    conn: Any,
    archive: RawSourceArchive,
    *,
    provider: str,
    source_file: str,
    watcher_generation: int,
    root_receipt: dict[str, Any],
    occurrence_receipt: dict[str, Any],
    record_start_offset: int,
    record_end_offset: int,
    raw_record_hash: str,
) -> NativeHeaderCatalogResult:
    ensure_native_session_headers_schema(conn)
    _verified_receipt(archive, occurrence_receipt, provider, source_file)
    raw = _record_bytes(archive, occurrence_receipt, record_start_offset, record_end_offset, raw_record_hash)
    context, error, session_id = _context(raw)
    context_json = _canonical(context) if context is not None else None
    context_sha = hashlib.sha256(context_json.encode()).hexdigest() if context_json is not None else None
    with _savepoint(conn):
        generation_key = _reserve(
            conn,
            archive,
            provider=provider,
            source_file=source_file,
            watcher_generation=watcher_generation,
            root_receipt=root_receipt,
            current_receipt=occurrence_receipt,
        )
        occurrence_key = hashlib.sha256(
            _canonical(
                [
                    "native-session-header-occurrence-v1",
                    generation_key,
                    record_start_offset,
                    record_end_offset,
                    raw_record_hash,
                ]
            ).encode()
        ).hexdigest()
        existing = conn.execute(
            "SELECT disposition,session_id,header_context_json,header_context_sha256,validation_error "
            "FROM native_session_header_occurrences WHERE occurrence_key=?",
            (occurrence_key,),
        ).fetchone()
        if existing is not None:
            status = conn.execute(
                "SELECT status FROM native_source_generations WHERE generation_key=?",
                (generation_key,),
            ).fetchone()[0]
            if tuple(existing[1:]) != (session_id, context_json, context_sha, error):
                raise NativeHeaderCatalogError("stored physical header occurrence differs")
            return NativeHeaderCatalogResult(generation_key, occurrence_key, existing[0], status, session_id, False)
        generation = conn.execute(
            "SELECT status,canonical_occurrence_key FROM native_source_generations WHERE generation_key=?",
            (generation_key,),
        ).fetchone()
        if error is not None:
            disposition, next_status = "invalid", "invalid"
        elif generation[1] is None:
            disposition = "canonical" if generation[0] == "active" else "conflict"
            next_status = generation[0]
        else:
            canonical_sha = conn.execute(
                "SELECT header_context_sha256 FROM native_session_header_occurrences WHERE occurrence_key=?",
                (generation[1],),
            ).fetchone()[0]
            disposition = "repeat" if canonical_sha == context_sha else "conflict"
            next_status = generation[0] if disposition == "repeat" or generation[0] == "invalid" else "conflicted"
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO native_session_header_occurrences VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                occurrence_key,
                generation_key,
                disposition,
                occurrence_receipt["manifest_sha256"],
                record_start_offset,
                record_end_offset,
                raw_record_hash,
                session_id,
                context_json,
                context_sha,
                error,
                now,
            ),
        )
        canonical_key = occurrence_key if disposition == "canonical" else generation[1]
        conn.execute(
            "UPDATE native_source_generations SET status=?,canonical_occurrence_key=?,updated_at=? "
            "WHERE generation_key=?",
            (next_status, canonical_key, now, generation_key),
        )
        return NativeHeaderCatalogResult(generation_key, occurrence_key, disposition, next_status, session_id, True)


def native_source_generation_status(conn: Any, generation_key: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT provider,source_file,watcher_generation,root_manifest_sha256,"
        "current_manifest_sha256,status,canonical_occurrence_key "
        "FROM native_source_generations WHERE generation_key=?",
        (generation_key,),
    ).fetchone()
    if row is None:
        return None
    keys = (
        "provider",
        "source_file",
        "watcher_generation",
        "root_manifest_sha256",
        "current_manifest_sha256",
        "status",
        "canonical_occurrence_key",
    )
    return dict(zip(keys, row, strict=True))
