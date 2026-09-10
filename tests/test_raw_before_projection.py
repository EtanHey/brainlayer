from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

import brainlayer.watcher as watcher_module
from brainlayer.raw_source_archive import RawSourceArchive
from brainlayer.watcher import JSONLWatcher, WatchRoot


def make_archive(tmp_path: Path, *, chunk_bytes: int = 32) -> RawSourceArchive:
    return RawSourceArchive(
        tmp_path / "archive",
        chunk_bytes=chunk_bytes,
        max_source_bytes=1024 * 1024,
    )


def make_watcher(
    tmp_path: Path,
    source_root: Path,
    archive: Any,
    flushed: list[dict],
    *,
    provider: str = "synthetic",
) -> JSONLWatcher:
    def confirm(items: list[dict]) -> dict[str, int]:
        flushed.extend(items)
        return {str(item["_source_file"]): item["_line_end_offset"] for item in items}

    return JSONLWatcher(
        watch_roots=[WatchRoot(provider, source_root)],
        registry_path=tmp_path / "offsets.json",
        on_flush=confirm,
        raw_archive=archive,
        batch_size=1,
        flush_interval_ms=1,
        max_lines_per_file=100,
    )


def archived_bytes(store: RawSourceArchive, receipt: dict) -> bytes:
    output = io.BytesIO()
    store.reconstruct(receipt, output)
    return output.getvalue()


def test_archive_precedes_parse_and_owns_malformed_unknown_and_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "session.jsonl"
    raw = b'{"type":"user","message":{"role":"user","content":"ok"}}\n{bad}\n42\npartial'
    source.write_bytes(raw)
    store = make_archive(tmp_path)
    flushed: list[dict] = []
    monkeypatch.setenv("BRAINLAYER_WATCHER_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    watcher = make_watcher(tmp_path, root, store, flushed)

    watcher.poll_once()
    watcher.poll_once()

    assert len(flushed) == 1
    receipt = watcher._archive_receipts[str(source)]
    assert archived_bytes(store, receipt) == raw
    offset, _inode = watcher.registry.get(str(source))
    assert offset < len(raw)
    assert list((tmp_path / "quarantine").glob("*.bad"))


def test_denied_session_is_archived_before_visibility_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "denied.jsonl"
    raw = b'{"type":"user","message":{"role":"user","content":"hidden"}}\n'
    source.write_bytes(raw)
    store = make_archive(tmp_path)
    flushed: list[dict] = []
    monkeypatch.setattr(
        watcher_module,
        "is_denylisted",
        lambda path: path == str(source),
    )
    watcher = make_watcher(tmp_path, root, store, flushed)

    assert watcher.poll_once() == 0

    assert flushed == []
    assert archived_bytes(store, watcher._archive_receipts[str(source)]) == raw
    assert watcher.registry.get(str(source))[0] == 0


def test_actual_brain_worker_class_exclusion_still_archives_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / ".claude" / "projects"
    session = projects / "project" / "session" / "subagents"
    session.mkdir(parents=True)
    source = session / "agent-synthetic.jsonl"
    raw = b'{"type":"user","attributionAgent":"brain-worker","message":{"role":"user","content":"hidden"}}\n'
    source.write_bytes(raw)
    store = make_archive(tmp_path)
    flushed: list[dict] = []
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    watcher = make_watcher(tmp_path, projects, store, flushed, provider="claude")

    assert watcher.poll_once() == 0

    assert flushed == []
    assert archived_bytes(store, watcher._archive_receipts[str(source)]) == raw
    assert watcher.registry.get(str(source))[0] == 0


class FailingArchive:
    def capture(self, _path: str, _provider: str) -> dict:
        raise OSError("archive unavailable")


def test_archive_write_failure_prevents_projection_and_offset_ack(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "session.jsonl"
    source.write_bytes(b'{"type":"user","message":{"role":"user","content":"x"}}\n')
    flushed: list[dict] = []
    watcher = make_watcher(tmp_path, root, FailingArchive(), flushed)

    assert watcher.poll_once() == 0

    assert flushed == []
    assert watcher.registry.get(str(source))[0] == 0
    assert watcher._raw_archive_failures[str(source)]["error_type"] == "OSError"
    assert watcher._file_ingestion_failures[str(source)]["disposition"] == "raw_archive_failed"


class ReadFailArchive:
    def __init__(self, delegate: RawSourceArchive) -> None:
        self.delegate = delegate

    def capture(self, path: str, provider: str) -> dict:
        return self.delegate.capture(path, provider)

    def open_snapshot(self, _receipt: dict) -> Any:
        raise OSError("snapshot read unavailable")


def test_archive_read_failure_restores_tailer_and_prevents_ack(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "session.jsonl"
    source.write_bytes(b'{"type":"user","message":{"role":"user","content":"x"}}\n')
    flushed: list[dict] = []
    watcher = make_watcher(tmp_path, root, ReadFailArchive(make_archive(tmp_path)), flushed)

    assert watcher.poll_once() == 0

    assert flushed == []
    assert watcher.registry.get(str(source))[0] == 0
    assert watcher._tailers[str(source)].offset == 0


class MutateAfterCaptureArchive(RawSourceArchive):
    def __init__(self, *args: Any, source: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.source = source
        self.mutated = False

    def capture(self, path: str | Path, provider: str) -> dict:
        receipt = super().capture(path, provider)
        if not self.mutated:
            self.mutated = True
            with self.source.open("ab") as handle:
                handle.write(b'{"type":"user","message":{"role":"user","content":"later"}}\n')
        return receipt


def test_post_capture_mutation_refuses_projection_until_recaptured(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "session.jsonl"
    first = b'{"type":"user","message":{"role":"user","content":"first"}}\n'
    source.write_bytes(first)
    store = MutateAfterCaptureArchive(
        tmp_path / "archive",
        chunk_bytes=32,
        max_source_bytes=1024 * 1024,
        source=source,
    )
    flushed: list[dict] = []
    watcher = make_watcher(tmp_path, root, store, flushed)

    assert watcher.poll_once() == 0
    assert flushed == []
    assert watcher.registry.get(str(source))[0] == 0
    first_receipt = dict(watcher._archive_receipts[str(source)])
    assert archived_bytes(store, first_receipt) == first
    assert watcher._file_ingestion_failures[str(source)]["disposition"] == ("source_changed_after_raw_archive")

    assert watcher.poll_once() == 2
    assert [item["message"]["content"] for item in flushed] == ["first", "later"]


def test_partial_frame_is_archived_then_replayed_after_completion(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "session.jsonl"
    partial = b'{"type":"user","message":{"role":"user","content":"partial"}}'
    source.write_bytes(partial)
    store = make_archive(tmp_path)
    flushed: list[dict] = []
    watcher = make_watcher(tmp_path, root, store, flushed)

    assert watcher.poll_once() == 0
    first_receipt = dict(watcher._archive_receipts[str(source)])
    assert archived_bytes(store, first_receipt) == partial
    assert watcher.registry.get(str(source))[0] == 0

    with source.open("ab") as handle:
        handle.write(b"\n")
    assert watcher.poll_once() == 1
    assert len(flushed) == 1
    assert archived_bytes(store, watcher._archive_receipts[str(source)]) == partial + b"\n"


def test_oversized_parser_record_is_owned_without_offset_advance(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "session.jsonl"
    raw = b'{"type":"user","message":{"role":"user","content":"' + b"x" * 100 + b'"}}\n'
    source.write_bytes(raw)
    store = make_archive(tmp_path)
    flushed: list[dict] = []
    watcher = make_watcher(tmp_path, root, store, flushed)
    watcher.max_record_bytes = 16

    assert watcher.poll_once() == 0

    assert flushed == []
    assert archived_bytes(store, watcher._archive_receipts[str(source)]) == raw
    assert watcher.registry.get(str(source))[0] == 0


def test_truncation_replays_new_version_and_retains_old_version(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "session.jsonl"
    first = b'{"type":"user","message":{"role":"user","content":"one"}}\n'
    source.write_bytes(first)
    store = make_archive(tmp_path)
    flushed: list[dict] = []
    watcher = make_watcher(tmp_path, root, store, flushed)

    assert watcher.poll_once() == 1
    old_receipt = dict(watcher._archive_receipts[str(source)])
    second = b'{"type":"user","message":{"role":"user","content":"new"}}\n'
    source.write_bytes(second)

    assert watcher.poll_once() == 1
    assert [item["message"]["content"] for item in flushed] == ["one", "new"]
    assert archived_bytes(store, old_receipt) == first
    assert archived_bytes(store, watcher._archive_receipts[str(source)]) == second


def test_archive_state_is_visible_on_health_surface(tmp_path: Path) -> None:
    import json

    root = tmp_path / "sources"
    root.mkdir()
    health = tmp_path / "health.json"
    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("synthetic", root)],
        registry_path=tmp_path / "offsets.json",
        on_flush=lambda _items: None,
        raw_archive=make_archive(tmp_path),
        health_path=health,
    )
    watcher._write_health_snapshot([])
    assert json.loads(health.read_text())["raw_archive_state"] == "ENABLED"

    disabled_health = tmp_path / "disabled-health.json"
    disabled = JSONLWatcher(
        watch_roots=[WatchRoot("synthetic", root)],
        registry_path=tmp_path / "disabled-offsets.json",
        on_flush=lambda _items: None,
        health_path=disabled_health,
    )
    disabled._write_health_snapshot([])
    assert json.loads(disabled_health.read_text())["raw_archive_state"] == "DISABLED_UNWIRED"


def test_archive_is_explicit_and_default_constructor_creates_no_archive(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("synthetic", root)],
        registry_path=tmp_path / "offsets.json",
        on_flush=lambda _items: None,
    )

    assert watcher.raw_archive_enabled is False
    assert watcher.raw_archive is None
    assert not (tmp_path / "archive").exists()
