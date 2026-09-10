from __future__ import annotations

import json
import signal as signal_module
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

import brainlayer.deploy_drift as deploy_drift
import brainlayer.parent_death as parent_death
import brainlayer.raw_source_archive as archive_module
import brainlayer.watcher as watcher_module
from brainlayer.cli import app, watch
from brainlayer.raw_source_archive import (
    RAW_ARCHIVE_MAX_SOURCE_BYTES_ENV,
    RAW_ARCHIVE_PATH_ENV,
    RawSourceArchive,
    create_raw_source_archive,
    raw_archive_max_source_bytes,
    raw_archive_path,
)


def test_factory_defaults_to_resolved_db_relative_archive(tmp_path: Path) -> None:
    db = tmp_path / "data" / "brainlayer.db"
    store = create_raw_source_archive(db, {})

    assert store.root == db.parent / "raw-source-archive-v1"
    assert store.chunk_bytes == RawSourceArchive.DEFAULT_CHUNK_BYTES
    assert store.max_source_bytes == 8 * 1024 * 1024 * 1024
    assert store.root.is_dir()


def test_factory_accepts_absolute_path_and_strict_cap_override(tmp_path: Path) -> None:
    root = tmp_path / "owned-archive"
    env = {
        RAW_ARCHIVE_PATH_ENV: str(root),
        RAW_ARCHIVE_MAX_SOURCE_BYTES_ENV: "12345",
    }

    store = create_raw_source_archive(tmp_path / "db.sqlite", env)

    assert store.root == root
    assert store.max_source_bytes == 12345


@pytest.mark.parametrize("value", ["", "relative/archive"])
def test_archive_path_override_refuses_ambiguous_location(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match=RAW_ARCHIVE_PATH_ENV):
        raw_archive_path(tmp_path / "db.sqlite", {RAW_ARCHIVE_PATH_ENV: value})


@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", "nan", " 12"])
def test_archive_cap_override_refuses_malformed_values(value: str) -> None:
    with pytest.raises(ValueError, match=RAW_ARCHIVE_MAX_SOURCE_BYTES_ENV):
        raw_archive_max_source_bytes({RAW_ARCHIVE_MAX_SOURCE_BYTES_ENV: value})


def _stub_live_watch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, str]:
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "data" / "watch.db"))
    monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
    monkeypatch.delenv(RAW_ARCHIVE_PATH_ENV, raising=False)
    monkeypatch.delenv(RAW_ARCHIVE_MAX_SOURCE_BYTES_ENV, raising=False)
    monkeypatch.setattr(signal_module, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(parent_death, "install_parent_death_watcher", lambda *_a, **_kw: None)
    monkeypatch.setattr(deploy_drift, "record_launch_from_environment", lambda *_a, **_kw: None)
    built: dict = {}

    class StubWatcher:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.indexer = types.SimpleNamespace(total_flushed=0)

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    monkeypatch.setattr(watcher_module, "JSONLWatcher", StubWatcher)
    source = tmp_path / "sources"
    source.mkdir()
    watch(source=[source], poll_interval=30, batch_size=10, flush_interval=500)
    return built, str(tmp_path / "data" / "raw-source-archive-v1")


def test_live_watch_always_injects_db_relative_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    built, expected_root = _stub_live_watch(tmp_path, monkeypatch)
    output = capsys.readouterr().out

    assert isinstance(built["raw_archive"], RawSourceArchive)
    assert str(built["raw_archive"].root) == expected_root
    compact_output = "".join(output.split())
    assert "ENABLED_REQUIRED" in compact_output
    assert expected_root in compact_output


def test_live_watch_factory_failure_prevents_watcher_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "data" / "watch.db"))
    monkeypatch.setattr(signal_module, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(parent_death, "install_parent_death_watcher", lambda *_a, **_kw: None)
    monkeypatch.setattr(deploy_drift, "record_launch_from_environment", lambda *_a, **_kw: None)
    built = False

    class StubWatcher:
        def __init__(self, **_kwargs):
            nonlocal built
            built = True

    def fail(_db_path: Path) -> RawSourceArchive:
        raise OSError("archive root unavailable")

    monkeypatch.setattr(archive_module, "create_raw_source_archive", fail)
    with pytest.raises(OSError, match="archive root unavailable"):
        watch(
            source=[tmp_path],
            poll_interval=30,
            batch_size=10,
            flush_interval=500,
        )
    assert built is False


def test_watch_backfill_dry_run_reports_without_creating_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    (source / "one.jsonl").write_text(json.dumps({"role": "user", "content": "one"}) + "\n")
    archive_root = tmp_path / "must-not-exist"
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "data" / "brainlayer.db"))

    result = CliRunner().invoke(
        app,
        ["watch-backfill", "--source", str(source), "--dry-run"],
        env={RAW_ARCHIVE_PATH_ENV: str(archive_root)},
        terminal_width=1000,
    )

    assert result.exit_code == 0, result.output
    compact_output = "".join(result.output.split())
    assert "raw_archive=DRY_RUN_NOT_CREATED" in compact_output
    assert str(archive_root) in compact_output
    assert not archive_root.exists()


def test_real_watch_backfill_archives_before_queue_and_reports_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / ".cursor" / "projects" / "repo" / "agent-transcripts" / "agent" / "agent.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps(
            {
                "type": "message",
                "payload": {
                    "role": "user",
                    "content": "Synthetic archive-before-queue backfill record.",
                },
                "timestamp": "2026-09-11T00:00:00Z",
            }
        )
        + "\n"
    )
    db = tmp_path / "data" / "brainlayer.db"
    archive_root = db.parent / "raw-source-archive-v1"
    queue = tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_DB", str(db))
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)

    result = CliRunner().invoke(
        app,
        ["watch-backfill", "--home", str(tmp_path), "--max-cycles", "3"],
        env={"BRAINLAYER_QUEUE_DIR": str(queue)},
        terminal_width=1000,
    )

    assert result.exit_code == 0, result.output
    compact_output = "".join(result.output.split())
    assert f"raw_archive=ENABLED_REQUIREDpath={archive_root}" in compact_output
    assert len(list((archive_root / "manifests").glob("*.json"))) == 1
    assert len(list((archive_root / "objects").iterdir())) == 1
    assert len(list(queue.glob("watcher-*.jsonl"))) == 1


def test_real_backfill_invalid_cap_refuses_before_offsets_or_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    (source / "one.jsonl").write_text(json.dumps({"role": "user", "content": "x"}) + "\n")
    db = tmp_path / "data" / "brainlayer.db"
    registry = tmp_path / "offsets.json"
    monkeypatch.setenv("BRAINLAYER_DB", str(db))

    result = CliRunner().invoke(
        app,
        ["watch-backfill", "--source", str(source), "--registry", str(registry)],
        env={RAW_ARCHIVE_MAX_SOURCE_BYTES_ENV: "invalid"},
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert RAW_ARCHIVE_MAX_SOURCE_BYTES_ENV in str(result.exception)
    assert not registry.exists()
    assert not (db.parent / "raw-source-archive-v1").exists()


def test_real_backfill_oversize_source_fails_loudly_without_offset_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    source_path = source / "oversize.jsonl"
    source_path.write_text(json.dumps({"role": "user", "content": "too large"}) + "\n")
    source_size = source_path.stat().st_size
    cap = source_size - 1
    db = tmp_path / "data" / "brainlayer.db"
    registry = tmp_path / "offsets.json"
    archive_root = db.parent / "raw-source-archive-v1"
    monkeypatch.setenv("BRAINLAYER_DB", str(db))

    result = CliRunner().invoke(
        app,
        ["watch-backfill", "--source", str(source), "--registry", str(registry)],
        env={RAW_ARCHIVE_MAX_SOURCE_BYTES_ENV: str(cap)},
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    message = str(result.exception)
    assert str(source_path) in message
    assert f"size {source_size}" in message
    assert f"cap {cap}" in message
    assert not registry.exists()
    assert not list((archive_root / "manifests").glob("*.json"))
