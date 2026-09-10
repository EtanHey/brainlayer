from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from brainlayer.raw_source_archive import RawSourceArchive
from brainlayer.watcher import _NATIVE_HEADER_STRUCTURAL_KIND, JSONLWatcher, OffsetRegistry, WatchRoot

SESSION_A = "01a08d1e-42f6-7093-83a4-1c1b160f4911"
SESSION_B = "01a08d1e-4373-7631-b908-1a465cb013d3"


def header(session_id: str, **extra: object) -> bytes:
    payload = {"id": session_id, **extra}
    return (json.dumps({"type": "session_meta", "payload": payload}, separators=(",", ":")) + "\n").encode()


def setup_source(tmp_path: Path, content: bytes):
    sessions = tmp_path / "codex" / "sessions"
    sessions.mkdir(parents=True)
    source = sessions / "rollout.jsonl"
    source.write_bytes(content)
    archive = RawSourceArchive(tmp_path / "archive")
    receipt = archive.capture(source, "codex")
    return sessions, source, archive, receipt


def watcher(sessions: Path, registry: Path, archive: RawSourceArchive, flushed: list[list[dict]] | None = None):
    return JSONLWatcher(
        watch_roots=[WatchRoot("codex", sessions)],
        registry_path=registry,
        on_flush=(flushed.append if flushed is not None else lambda _items: None),
        batch_size=1,
        flush_interval_ms=0,
        registry_flush_interval_s=3600,
        raw_archive=archive,
    )


def test_offset_registry_reserves_and_preserves_root_across_reload(tmp_path: Path) -> None:
    raw = header(SESSION_A)
    sessions, source, archive, receipt = setup_source(tmp_path, raw)
    registry_path = tmp_path / "offsets.json"
    registry = OffsetRegistry(registry_path)
    registry.set(str(source), 0, source.stat().st_ino, generation=0)
    registry.reserve_native_header(
        str(source),
        offset=0,
        generation=0,
        provider="codex",
        root_receipt=receipt,
        record_start_offset=0,
        record_end_offset=len(raw),
        raw_record_hash=hashlib.sha256(raw).hexdigest(),
    )
    assert registry.flush() is True
    registry.set(str(source), len(raw), source.stat().st_ino, generation=0, raw_archive_receipt=receipt)
    assert registry.flush() is True

    reloaded = OffsetRegistry(registry_path)
    assert reloaded.get(str(source)) == (len(raw), source.stat().st_ino)
    assert reloaded.native_header_root_receipt(str(source)) == receipt
    assert reloaded.native_header_pending(str(source))["record_end_offset"] == len(raw)


def test_offset_registry_refuses_competing_root_same_generation(tmp_path: Path) -> None:
    _sessions, source, archive, first_receipt = setup_source(tmp_path, header(SESSION_A))
    registry = OffsetRegistry(tmp_path / "offsets.json")
    registry.set(str(source), 0, source.stat().st_ino, generation=4)
    registry.reserve_native_header(
        str(source),
        offset=0,
        generation=4,
        provider="codex",
        root_receipt=first_receipt,
        record_start_offset=0,
        record_end_offset=len(header(SESSION_A)),
        raw_record_hash=hashlib.sha256(header(SESSION_A)).hexdigest(),
    )
    source.write_bytes(header(SESSION_B))
    second_receipt = archive.capture(source, "codex")

    with pytest.raises(ValueError, match="competing native header root"):
        registry.reserve_native_header(
            str(source),
            offset=0,
            generation=4,
            provider="codex",
            root_receipt=second_receipt,
            record_start_offset=0,
            record_end_offset=len(header(SESSION_B)),
            raw_record_hash=hashlib.sha256(header(SESSION_B)).hexdigest(),
        )
    assert registry.native_header_root_receipt(str(source)) == first_receipt


def test_offset_registry_refuses_competing_root_during_whole_entry_merge(tmp_path: Path) -> None:
    _sessions, source, archive, first_receipt = setup_source(tmp_path, header(SESSION_A))
    registry_path = tmp_path / "offsets.json"
    initial = OffsetRegistry(registry_path)
    initial.set(str(source), 0, source.stat().st_ino, generation=4)
    assert initial.flush() is True
    stale = OffsetRegistry(registry_path)
    advancing = OffsetRegistry(registry_path)
    advancing.reserve_native_header(
        str(source),
        offset=0,
        generation=4,
        provider="codex",
        root_receipt=first_receipt,
        record_start_offset=0,
        record_end_offset=len(header(SESSION_A)),
        raw_record_hash=hashlib.sha256(header(SESSION_A)).hexdigest(),
    )
    assert advancing.flush() is True
    source.write_bytes(header(SESSION_B))
    second_receipt = archive.capture(source, "codex")
    stale.reserve_native_header(
        str(source),
        offset=0,
        generation=4,
        provider="codex",
        root_receipt=second_receipt,
        record_start_offset=0,
        record_end_offset=len(header(SESSION_B)),
        raw_record_hash=hashlib.sha256(header(SESSION_B)).hexdigest(),
    )

    assert stale.flush() is False
    assert OffsetRegistry(registry_path).native_header_root_receipt(str(source)) == first_receipt


def test_offset_registry_rewind_resets_root_only_with_new_generation(tmp_path: Path) -> None:
    _sessions, source, archive, receipt = setup_source(tmp_path, header(SESSION_A))
    registry = OffsetRegistry(tmp_path / "offsets.json")
    registry.set(str(source), 0, source.stat().st_ino, generation=2)
    registry.reserve_native_header(
        str(source),
        offset=0,
        generation=2,
        provider="codex",
        root_receipt=receipt,
        record_start_offset=0,
        record_end_offset=len(header(SESSION_A)),
        raw_record_hash=hashlib.sha256(header(SESSION_A)).hexdigest(),
    )
    generation = registry.mark_rewind(str(source), source.stat().st_ino)
    assert generation > 2
    assert registry.native_header_root_receipt(str(source)) is None
    assert registry.native_header_pending(str(source)) is None


def test_header_only_is_structural_and_never_confirmed_without_consumer(tmp_path: Path) -> None:
    sessions, source, archive, receipt = setup_source(tmp_path, header(SESSION_A))
    flushed: list[list[dict]] = []
    current = watcher(sessions, tmp_path / "offsets.json", archive, flushed)

    assert current.poll_once() == 1
    assert flushed == []
    assert current.registry.get(str(source)) == (0, 0)
    assert current.registry.native_header_root_receipt(str(source)) == receipt
    buffered = current.indexer._buffer
    assert len(buffered) == 1
    assert buffered[0]["_structural_kind"] == _NATIVE_HEADER_STRUCTURAL_KIND
    assert buffered[0]["_line_start_offset"] == 0
    assert buffered[0]["_line_end_offset"] == len(header(SESSION_A))
    assert buffered[0]["_raw_archive_receipt"] == receipt


def test_restart_replays_persisted_structural_header_without_offset_ack(tmp_path: Path) -> None:
    sessions, source, archive, receipt = setup_source(tmp_path, header(SESSION_A))
    registry_path = tmp_path / "offsets.json"
    first = watcher(sessions, registry_path, archive)
    first.poll_once()
    assert first.registry.flush() is True

    flushed: list[list[dict]] = []
    restarted = watcher(sessions, registry_path, archive, flushed)
    assert restarted.poll_once() == 1
    assert flushed == []
    assert restarted.registry.get(str(source)) == (0, 0)
    assert restarted.registry.native_header_root_receipt(str(source)) == receipt


def test_prefix_append_keeps_original_native_header_root(tmp_path: Path) -> None:
    sessions, source, archive, first_receipt = setup_source(tmp_path, header(SESSION_A))
    current = watcher(sessions, tmp_path / "offsets.json", archive)
    current.poll_once()
    with source.open("ab") as output:
        output.write(
            b'{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}}\n'
        )
    latest_receipt = archive.capture(source, "codex")
    current.poll_once()

    assert current.registry.native_header_root_receipt(str(source)) == first_receipt
    assert current.registry.native_header_root_receipt(str(source)) != latest_receipt
    assert current.registry.get(str(source)) == (0, 0)


def test_legacy_eof_recovers_first_header_from_verified_snapshot(tmp_path: Path) -> None:
    raw = header(SESSION_A) + b'{"type":"response_item"}\n'
    sessions, source, archive, receipt = setup_source(tmp_path, raw)
    registry_path = tmp_path / "offsets.json"
    registry = OffsetRegistry(registry_path)
    registry.set(str(source), len(raw), source.stat().st_ino, generation=0, raw_archive_receipt=receipt)
    assert registry.flush() is True
    flushed: list[list[dict]] = []
    current = watcher(sessions, registry_path, archive, flushed)

    assert current.poll_once() == 0
    assert flushed == []
    assert current.registry.get(str(source)) == (len(raw), source.stat().st_ino)
    assert current.registry.native_header_root_receipt(str(source)) == receipt
    assert current.registry.native_header_pending(str(source))["record_start_offset"] == 0
    assert current.indexer.has_buffered_structural_source(str(source))


def test_malformed_and_oversized_first_headers_remain_unconfirmed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BRAINLAYER_WATCHER_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    sessions, malformed, archive, _receipt = setup_source(tmp_path / "malformed", b"not-json\n")
    malformed_watcher = watcher(sessions, tmp_path / "malformed-offsets.json", archive)
    malformed_watcher.poll_once()
    assert malformed_watcher.registry.native_header_root_receipt(str(malformed)) is None
    assert malformed_watcher.registry.get(str(malformed)) == (malformed.stat().st_size, malformed.stat().st_ino)

    oversized_raw = header(SESSION_A, padding="x" * (256 * 1024))
    sessions2, oversized, archive2, _receipt2 = setup_source(tmp_path / "oversized", oversized_raw)
    oversized_watcher = watcher(sessions2, tmp_path / "oversized-offsets.json", archive2)
    oversized_watcher.poll_once()
    assert oversized_watcher.registry.native_header_root_receipt(str(oversized)) is None
    assert oversized_watcher.registry.get(str(oversized)) == (0, 0)
    assert not oversized_watcher.indexer.has_buffered_structural_source(str(oversized))
