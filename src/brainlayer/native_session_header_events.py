"""Strict durable-queue envelope for native Codex session headers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .native_session_headers import (
    MAX_HEADER_RECORD_BYTES,
    NativeHeaderCatalogError,
    _record_bytes,
    _verified_receipt,
    catalog_native_session_header,
)
from .raw_source_archive import RawSourceArchive

EVENT_KIND = "watcher_native_session_header"
EVENT_VERSION = 1
MAX_EVENT_BYTES = 160 * 1024

_PAYLOAD_KEYS = {
    "kind",
    "version",
    "provider",
    "source_file",
    "watcher_generation",
    "root_receipt",
    "occurrence_receipt",
    "record_start_offset",
    "record_end_offset",
    "raw_record_hash",
}
_QUEUE_KEYS = _PAYLOAD_KEYS | {"source", "queued_at"}
_HEADER_INPUT_KEYS = _PAYLOAD_KEYS - {"kind", "version"}


class NativeSessionHeaderEventError(ValueError):
    """The event cannot safely identify immutable physical header bytes."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise NativeSessionHeaderEventError("native header event is not canonical JSON data") from error


def validate_native_session_header_event(
    event: Mapping[str, Any],
    archive: RawSourceArchive,
    *,
    queued: bool = False,
) -> dict[str, Any]:
    """Return a detached, physically verified event without parsing header metadata."""
    if not isinstance(event, Mapping):
        raise NativeSessionHeaderEventError("native header event must be an object")
    expected_keys = _QUEUE_KEYS if queued else _PAYLOAD_KEYS
    if set(event) != expected_keys:
        raise NativeSessionHeaderEventError("native header event has invalid keys")
    encoded = _canonical_bytes(event)
    if len(encoded) > MAX_EVENT_BYTES:
        raise NativeSessionHeaderEventError("native header event exceeds byte cap")
    if event.get("kind") != EVENT_KIND:
        raise NativeSessionHeaderEventError("native header event has invalid kind")
    if type(event.get("version")) is not int or event["version"] != EVENT_VERSION:
        raise NativeSessionHeaderEventError("native header event has invalid version")
    provider = event.get("provider")
    source_file = event.get("source_file")
    watcher_generation = event.get("watcher_generation")
    if provider != "codex":
        raise NativeSessionHeaderEventError("native header event provider must be codex")
    if not isinstance(source_file, str) or source_file != str(Path(source_file).absolute()):
        raise NativeSessionHeaderEventError("native header event source_file must be absolute")
    if type(watcher_generation) is not int or watcher_generation < 0:
        raise NativeSessionHeaderEventError("native header event watcher_generation must be nonnegative")
    if queued:
        if event.get("source") != "watcher-native-session-header":
            raise NativeSessionHeaderEventError("native header event has invalid queue source")
        if type(event.get("queued_at")) not in {int, float}:
            raise NativeSessionHeaderEventError("native header event queued_at must be numeric")

    root_receipt = event.get("root_receipt")
    occurrence_receipt = event.get("occurrence_receipt")
    try:
        _verified_receipt(archive, root_receipt, provider, source_file)
        _verified_receipt(archive, occurrence_receipt, provider, source_file)
        if not archive.is_prefix(root_receipt, occurrence_receipt):
            raise NativeSessionHeaderEventError("occurrence receipt does not extend immutable generation root")
        _record_bytes(
            archive,
            occurrence_receipt,
            event.get("record_start_offset"),
            event.get("record_end_offset"),
            event.get("raw_record_hash"),
        )
    except NativeHeaderCatalogError as error:
        raise NativeSessionHeaderEventError(str(error)) from error

    detached = json.loads(encoded)
    if detached["record_end_offset"] - detached["record_start_offset"] > MAX_HEADER_RECORD_BYTES:
        raise NativeSessionHeaderEventError("native header record exceeds byte cap")
    return detached


def build_native_session_header_event(
    *,
    archive: RawSourceArchive,
    provider: str,
    source_file: str,
    watcher_generation: int,
    root_receipt: dict[str, Any],
    occurrence_receipt: dict[str, Any],
    record_start_offset: int,
    record_end_offset: int,
    raw_record_hash: str,
) -> dict[str, Any]:
    """Build and physically verify the only supported queue payload."""
    event = {
        "kind": EVENT_KIND,
        "version": EVENT_VERSION,
        "provider": provider,
        "source_file": source_file,
        "watcher_generation": watcher_generation,
        "root_receipt": root_receipt,
        "occurrence_receipt": occurrence_receipt,
        "record_start_offset": record_start_offset,
        "record_end_offset": record_end_offset,
        "raw_record_hash": raw_record_hash,
    }
    return validate_native_session_header_event(event, archive)


def build_native_session_header_event_from_mapping(
    header: Mapping[str, Any],
    *,
    archive: RawSourceArchive,
) -> dict[str, Any]:
    """Reject extra caller fields before constructing the fixed-kind event."""
    if not isinstance(header, Mapping) or set(header) != _HEADER_INPUT_KEYS:
        raise NativeSessionHeaderEventError("native header input has invalid keys")
    return build_native_session_header_event(archive=archive, **dict(header))


def apply_native_session_header_event(
    conn: Any,
    archive: RawSourceArchive,
    event: Mapping[str, Any],
) -> None:
    """Revalidate immutable bytes, then catalog inside the caller's transaction."""
    payload = validate_native_session_header_event(event, archive, queued=True)
    catalog_native_session_header(
        conn,
        archive,
        provider=payload["provider"],
        source_file=payload["source_file"],
        watcher_generation=payload["watcher_generation"],
        root_receipt=payload["root_receipt"],
        occurrence_receipt=payload["occurrence_receipt"],
        record_start_offset=payload["record_start_offset"],
        record_end_offset=payload["record_end_offset"],
        raw_record_hash=payload["raw_record_hash"],
    )
