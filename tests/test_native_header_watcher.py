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
    assert reloaded.native_header_root_basis(str(source)) == "observed_header"
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
    assert current.registry.native_header_root_basis(str(source)) == "observed_header"
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
    assert current.registry.native_header_root_basis(str(source)) == "first_admitted_catalog_root"
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


def test_whole_registry_merge_preserves_stale_reservation_on_newer_offset_winner(tmp_path: Path) -> None:
    raw = header(SESSION_A)
    _sessions, source, _archive, receipt = setup_source(tmp_path, raw)
    registry_path = tmp_path / "offsets.json"
    initial = OffsetRegistry(registry_path)
    initial.set(str(source), 0, source.stat().st_ino, generation=0)
    assert initial.flush() is True
    stale = OffsetRegistry(registry_path)
    advancing = OffsetRegistry(registry_path)
    stale.reserve_native_header(
        str(source),
        offset=0,
        generation=0,
        provider="codex",
        root_receipt=receipt,
        record_start_offset=0,
        record_end_offset=len(raw),
        raw_record_hash=hashlib.sha256(raw).hexdigest(),
    )
    advancing.set(str(source), len(raw), source.stat().st_ino, generation=0, raw_archive_receipt=receipt)
    assert advancing.flush() is True

    assert stale.flush() is True
    merged = OffsetRegistry(registry_path)
    assert merged.get(str(source))[0] == len(raw)
    assert merged.native_header_root_receipt(str(source)) == receipt
    assert merged.native_header_root_basis(str(source)) == "observed_header"


def test_later_session_meta_uses_stable_root_and_retains_new_occurrence(tmp_path: Path) -> None:
    first = header(SESSION_A)
    sessions, source, archive, first_receipt = setup_source(tmp_path, first)
    current = watcher(sessions, tmp_path / "offsets.json", archive)
    assert current.poll_once() == 1
    with source.open("ab") as stream:
        stream.write(header(SESSION_B))
    second_receipt = archive.capture(source, "codex")

    assert current.poll_once() == 1
    assert current.registry.native_header_root_receipt(str(source)) == first_receipt
    assert current.registry.native_header_pending(str(source))["raw_archive_receipt"] == first_receipt
    assert second_receipt != first_receipt
    assert len(current.indexer._buffer) == 2


class DeferredResult(dict[str, int]):
    inserted = 0
    skipped = 0

    def __init__(self, entries: list[dict]) -> None:
        super().__init__()
        self.deferred_entries = entries


def test_callback_failure_isolation_holds_pending_source_and_advances_only_unrelated_success(
    tmp_path: Path,
) -> None:
    codex_dir = tmp_path / "codex" / "sessions"
    claude_dir = tmp_path / "claude" / "projects"
    claude_project = claude_dir / "project"
    codex_dir.mkdir(parents=True)
    claude_project.mkdir(parents=True)
    codex = codex_dir / "rollout.jsonl"
    good = claude_project / "good.jsonl"
    deferred = claude_project / "deferred.jsonl"
    assistant = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        },
    }
    codex.write_bytes(header(SESSION_A) + (json.dumps(assistant) + "\n").encode())
    good.write_text(json.dumps({"type": "assistant", "message": {"content": "good"}}) + "\n")
    deferred.write_text(json.dumps({"type": "assistant", "message": {"content": "later"}}) + "\n")
    calls: list[list[str]] = []

    def callback(items: list[dict]) -> dict[str, int] | DeferredResult:
        sources = [item["_source_file"] for item in items]
        calls.append(sources)
        if len(items) > 1:
            raise RuntimeError("force per-item isolation")
        if sources == [str(deferred)]:
            return DeferredResult(items)
        return {str(good): items[0]["_line_end_offset"]}

    current = JSONLWatcher(
        watch_roots=[WatchRoot("codex", codex_dir), WatchRoot("claude", claude_dir)],
        registry_path=tmp_path / "offsets.json",
        on_flush=callback,
        batch_size=100,
        flush_interval_ms=60_000,
        registry_flush_interval_s=3600,
        raw_archive=RawSourceArchive(tmp_path / "archive"),
    )
    assert current.poll_once() == 4
    current.indexer.flush()

    assert any(call == [str(good)] for call in calls)
    assert any(call == [str(deferred)] for call in calls)
    assert all(str(codex) not in call for call in calls)
    assert current.registry.get(str(good))[0] == good.stat().st_size
    assert current.registry.get(str(deferred))[0] == 0
    assert current.registry.get(str(codex))[0] == 0
    assert len(current.indexer._buffer) == 3


def test_legacy_unknown_is_persisted_per_manifest_and_new_generation_can_supply_valid_header(
    tmp_path: Path,
) -> None:
    raw = b"not-json\n"
    sessions, source, archive, receipt = setup_source(tmp_path, raw)
    registry_path = tmp_path / "offsets.json"
    registry = OffsetRegistry(registry_path)
    registry.set(str(source), len(raw), source.stat().st_ino, generation=0, raw_archive_receipt=receipt)
    assert registry.flush() is True
    current = watcher(sessions, registry_path, archive)

    assert current.poll_once() == 0
    assert current.registry.native_header_recovery_unknown(str(source)) == {
        "reason": "malformed_first_record",
        "manifest_sha256": receipt["manifest_sha256"],
    }
    assert current.poll_once() == 0

    replacement = source.with_suffix(".replacement")
    replacement.write_bytes(header(SESSION_B))
    replacement.replace(source)
    assert current.poll_once() == 1
    assert current.registry.native_header_root_basis(str(source)) == "observed_header"
    assert current.registry.native_header_recovery_unknown(str(source)) is None
    assert current.registry.flush() is True
    reloaded = OffsetRegistry(registry_path)
    assert reloaded.native_header_root_basis(str(source)) == "observed_header"
    assert reloaded.native_header_recovery_unknown(str(source)) is None


def test_changed_version_replaces_unknown_diagnostic_across_flush_and_reload(tmp_path: Path) -> None:
    raw = b"not-json\n"
    _sessions, source, archive, first_receipt = setup_source(tmp_path, raw)
    registry_path = tmp_path / "offsets.json"
    first = OffsetRegistry(registry_path)
    first.set(str(source), len(raw), source.stat().st_ino, generation=3, raw_archive_receipt=first_receipt)
    first.record_native_header_recovery_unknown(
        str(source), generation=3, receipt=first_receipt, reason="malformed_first_record"
    )
    assert first.flush() is True

    with source.open("ab") as stream:
        stream.write(b"still-not-a-header\n")
    second_receipt = archive.capture(source, "codex")
    changed = OffsetRegistry(registry_path)
    changed.record_native_header_recovery_unknown(
        str(source), generation=3, receipt=second_receipt, reason="malformed_first_record"
    )
    assert changed.flush() is True

    assert first_receipt["manifest_sha256"] != second_receipt["manifest_sha256"]
    assert OffsetRegistry(registry_path).native_header_recovery_unknown(str(source)) == {
        "reason": "malformed_first_record",
        "manifest_sha256": second_receipt["manifest_sha256"],
    }


def test_valid_root_wins_over_concurrent_changed_version_unknown_diagnostic(tmp_path: Path) -> None:
    raw = header(SESSION_A)
    _sessions, source, archive, first_receipt = setup_source(tmp_path, raw)
    registry_path = tmp_path / "offsets.json"
    initial = OffsetRegistry(registry_path)
    initial.set(str(source), 0, source.stat().st_ino, generation=5)
    initial.record_native_header_recovery_unknown(
        str(source), generation=5, receipt=first_receipt, reason="malformed_first_record"
    )
    assert initial.flush() is True
    root_writer = OffsetRegistry(registry_path)
    stale_diagnostic_writer = OffsetRegistry(registry_path)

    root_writer.reserve_native_header(
        str(source),
        offset=0,
        generation=5,
        provider="codex",
        root_receipt=first_receipt,
        record_start_offset=0,
        record_end_offset=len(raw),
        raw_record_hash=hashlib.sha256(raw).hexdigest(),
    )
    assert root_writer.flush() is True
    with source.open("ab") as stream:
        stream.write(b'{"type":"response_item"}\n')
    second_receipt = archive.capture(source, "codex")
    stale_diagnostic_writer.record_native_header_recovery_unknown(
        str(source), generation=5, receipt=second_receipt, reason="first_record_not_session_meta"
    )
    assert stale_diagnostic_writer.flush() is True

    reloaded = OffsetRegistry(registry_path)
    assert reloaded.native_header_root_receipt(str(source)) == first_receipt
    assert reloaded.native_header_root_basis(str(source)) == "observed_header"
    assert reloaded.native_header_recovery_unknown(str(source)) is None
