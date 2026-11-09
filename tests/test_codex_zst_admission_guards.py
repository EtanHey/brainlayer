"""Fail-first admission probes from round A."""

import os

import pytest

from brainlayer.watcher import JSONLWatcher, WatchRoot
from tests.test_codex_zst_watcher import compress as _compress
from tests.test_codex_zst_watcher import watcher
from tests.test_codex_zstd_stream import rollout

compress = _compress


def test_truncated_newest_archive_does_not_starve_siblings(compress, tmp_path, monkeypatch):
    bad = compress("bad.jsonl.zst", rollout("bad prefix " * 8) + b'{"partial":')
    good = [compress(f"good{i}.jsonl.zst", rollout(f"good{i} " * 12)) for i in range(3)]
    for path in good:
        os.utime(path, (1000, 1000))
    os.utime(bad, (5000, 5000))
    instance, entries = watcher(tmp_path, monkeypatch, cap=1)
    try:
        for _ in range(8):
            instance.poll_once()
        assert {e["_source_file"] for e in entries} == {str(p.with_suffix("")) for p in [bad, *good]}
        failure = instance._file_ingestion_failures[str(bad.with_suffix(""))]
        assert failure["disposition"] == "quarantined"
        assert "truncated_final_record" in failure["error"]
        assert failure["file_path"] == str(bad)
        assert failure["file_size_bytes"] == bad.stat().st_size
    finally:
        instance.stop()


def test_invalid_admission_refuses_construction(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_ADMISSION", "invalid")
    with pytest.raises(ValueError, match="must be off or bounded"):
        JSONLWatcher(watch_roots=[WatchRoot("claude", tmp_path)], registry_path=tmp_path / "offsets")


def test_off_file_watch_root_keeps_legacy_skip(tmp_path, monkeypatch):
    path = tmp_path / "file.jsonl"
    path.write_bytes(rollout())
    instance, _ = watcher(tmp_path, monkeypatch, enabled=False)
    instance.watch_roots = [WatchRoot("codex", path)]
    assert instance._discover_jsonl_files() == []


def test_completed_archive_skips_queue_scan(compress, tmp_path, monkeypatch):
    compress("done.jsonl.zst", rollout())
    instance, _ = watcher(tmp_path, monkeypatch)
    try:
        instance.poll_once()
        monkeypatch.setattr("brainlayer.codex_zst.queue_has_capacity", lambda: pytest.fail("unnecessary queue scan"))
        instance.poll_once()
    finally:
        instance.stop()


def test_short_archive_checkpoint_failure_is_terminal(compress, tmp_path, monkeypatch):
    plain = tmp_path / "short.jsonl"
    plain.write_bytes(rollout())
    instance, _ = watcher(tmp_path, monkeypatch)
    try:
        instance.poll_once()
        compress("short.jsonl.zst", b"{}\n")
        plain.unlink()
        instance.poll_once()
        failure = instance._file_ingestion_failures[str(plain)]
        assert "before its decoded checkpoint" in failure["error"]
        monkeypatch.setattr("brainlayer.codex_zst.ZstdDecodedFile", lambda *_: pytest.fail("retried terminal archive"))
        instance.poll_once()
    finally:
        instance.stop()


def test_failure_alarm_names_physical_archive(compress, tmp_path, monkeypatch):
    path = compress("corrupt.jsonl.zst", rollout())
    path.write_bytes(b"not a zstd archive")
    instance, _ = watcher(tmp_path, monkeypatch)
    try:
        instance.poll_once()
        failure = instance._file_ingestion_failures[str(path.with_suffix(""))]
        assert failure["file_path"] == str(path)
        assert failure["file_size_bytes"] == path.stat().st_size
    finally:
        instance.stop()
