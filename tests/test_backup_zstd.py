"""zstd DB snapshots beside legacy gzip: round trip, compatibility, retention, gate, and stop.

No production I/O: every DB is a tmp_path fixture and BrainBar's socket is never reached.
"""

# Pytest registers the imported fixture; test arguments intentionally shadow that import.
# ruff: noqa: F811

import datetime as dt
import gzip
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup
from brainlayer import maintenance
from tests.test_backup_daily import _create_source_db
from tests.test_weekly_local_gate import config_for, local_lane  # noqa: F401

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def lane(tmp_path, monkeypatch):
    source, output = tmp_path / "source.db", tmp_path / "backups"
    _create_source_db(source, chunk_count=4)
    output.mkdir()
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setattr(backup, "_brainbar_writer_started_at", lambda *a: 1)
    monkeypatch.setattr(backup, "_important_usage_capacity_bytes", lambda *a: None)
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=10**12))

    def vacuum_into(target, **kwargs):  # noqa: ARG001
        with sqlite3.connect(source) as conn:
            conn.execute("VACUUM INTO ?", (str(target),))

    monkeypatch.setattr(backup, "request_brainbar_vacuum_into", vacuum_into)
    return source, output


def _zstd_bytes(data: bytes) -> bytes:
    return subprocess.run(
        [backup._require_zstd(), "-q", "-c", "--check"], input=data, capture_output=True, check=True
    ).stdout


def _restore_and_check(archive: Path, tmp_path: Path, expected_chunks: int) -> None:
    restored = tmp_path / f"restored-{archive.name}.db"
    backup._decompress_archive_to(archive, restored)
    with sqlite3.connect(restored) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == expected_chunks


def _fake_zstd(tmp_path: Path, monkeypatch) -> Path:
    """A zstd stand-in that records its pid and blocks, so stops can be observed."""
    script = tmp_path / "fake-zstd"
    script.write_text('#!/bin/sh\necho $$ > "$FAKE_ZSTD_PIDFILE"\nexec sleep 30\n')
    script.chmod(0o755)
    pidfile = tmp_path / "fake-zstd.pid"
    monkeypatch.setenv("FAKE_ZSTD_PIDFILE", str(pidfile))
    return pidfile


def _wait_for_pid(pidfile: Path) -> int:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if pidfile.exists() and pidfile.read_text().strip():
            return int(pidfile.read_text())
        time.sleep(0.02)
    raise AssertionError("fake zstd never started")


def _gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


# --- round trip ------------------------------------------------------------------------


def test_zstd_is_the_default_and_round_trips_create_verify_restore(lane, tmp_path):
    source, output = lane
    artifact = backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-20")

    assert artifact.archive_path == output / "2026-10-20.db.zst"
    assert artifact.compressor == "zstd" and artifact.compressor_fallback is None
    assert artifact.compress_duration_seconds is not None
    assert not (output / "2026-10-20.db").exists(), "raw is removed once the zstd archive is proven"
    assert artifact.archive_path.read_bytes()[:4] == b"\x28\xb5\x2f\xfd", "zstd frame magic"

    result = backup.verify_sqlite_backup_artifact(artifact)
    assert result["verified"] is True, result
    assert result["archive_test"] is True and result["compression"] == "zstd"
    assert "gzip_test" not in result
    assert result["sentinel_verified_chunks"] == 4
    _restore_and_check(artifact.archive_path, tmp_path, 4)


def test_full_verify_of_zstd_compares_drive_copy(lane, monkeypatch):
    source, output = lane
    artifact = backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-18")
    uploaded = artifact.archive_path.read_bytes()

    def download(service, *, file_id, destination):  # noqa: ARG001
        destination.write_bytes(uploaded)
        return destination

    monkeypatch.setattr(backup, "download_drive_file_raw", download)
    result = backup.verify_sqlite_backup_artifact(artifact, full=True, service=object(), file_id="f")
    assert result["verified"] is True and result["drive_md5_match"] is True


def test_corrupt_zstd_fails_verification(lane):
    source, output = lane
    artifact = backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-20")
    data = bytearray(artifact.archive_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    artifact.archive_path.write_bytes(bytes(data))
    result = backup.verify_sqlite_backup_artifact(artifact)
    assert result["verified"] is False and result["archive_test"] is False
    assert result["verification_error"]


# --- legacy gzip compatibility ------------------------------------------------------------


def test_legacy_gzip_still_verifies_and_restores_with_zstd_default(lane, tmp_path):
    source, output = lane
    raw = tmp_path / "legacy.db"
    with sqlite3.connect(source) as conn:
        conn.execute("VACUUM INTO ?", (str(raw),))
    legacy = output / "2026-10-09.db.gz"
    legacy.write_bytes(gzip.compress(raw.read_bytes()))
    artifact = backup.SQLiteBackupArtifact(
        gzip_path=legacy,
        uncompressed_path=None,
        sentinel_chunks=4,
        local_retention_deleted=[],
        stale_attempts_deleted=[],
        surviving_attempts=[],
        surviving_attempt_bytes=0,
        surviving_attempt_growth_reserve_bytes=0,
        attempt_reclamation="resumed",
        writer_probe_error=None,
    )
    result = backup.verify_sqlite_backup_artifact(artifact)
    assert result["verified"] is True, result
    assert result["gzip_test"] is True and result["archive_test"] is True and result["compression"] == "gzip"
    _restore_and_check(legacy, tmp_path, 4)


def test_gzip_rollback_switch_writes_gzip(lane, monkeypatch):
    source, output = lane
    monkeypatch.setenv(backup.BACKUP_COMPRESSOR_ENV, "gzip")
    artifact = backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-20")
    assert artifact.archive_path.name == "2026-10-20.db.gz" and artifact.compressor == "gzip"
    assert backup.verify_sqlite_backup_artifact(artifact)["verified"] is True


def test_unknown_compressor_is_refused_before_any_snapshot(lane, monkeypatch):
    source, output = lane
    monkeypatch.setenv(backup.BACKUP_COMPRESSOR_ENV, "brotli")
    monkeypatch.setattr(backup, "request_brainbar_vacuum_into", lambda *a, **kw: pytest.fail("snapshot taken"))
    with pytest.raises(ValueError, match="zstd or gzip"):
        backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-20")


def test_missing_zstd_falls_back_to_gzip_loudly_in_the_receipt(lane, monkeypatch, capsys):
    source, output = lane
    monkeypatch.setattr(backup, "_resolve_zstd", lambda: None)
    log = output / "receipt.jsonl"
    receipt = backup.run_backup(source, staging_dir=output, date_stamp="2026-10-20", upload=False, log_path=log)
    assert receipt["snapshot"].endswith("2026-10-20.db.gz")
    assert receipt["compressor"] == "gzip"
    assert "zstd binary not found" in receipt["compressor_fallback"]
    assert "Backup compressor fallback" in capsys.readouterr().out


def test_zstd_override_must_be_an_absolute_executable(monkeypatch, tmp_path):
    monkeypatch.setenv(backup.BACKUP_ZSTD_PATH_ENV, "zstd")
    with pytest.raises(ValueError, match="absolute executable"):
        backup._resolve_zstd()
    monkeypatch.setenv(backup.BACKUP_ZSTD_PATH_ENV, str(tmp_path / "missing-zstd"))
    with pytest.raises(ValueError, match="absolute executable"):
        backup._selected_compressor()


def test_zstd_resolution_does_not_depend_on_launchd_path(monkeypatch):
    real = backup._resolve_zstd()
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    if os.path.exists("/opt/homebrew/bin/zstd") or os.path.exists("/usr/local/bin/zstd"):
        assert backup._resolve_zstd() in backup.ZSTD_CANDIDATE_PATHS
    else:
        assert backup._resolve_zstd() in {real, None}


# --- raw/archive binding --------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["valid", "corrupt", "mismatched", "not-sqlite"])
def test_leftover_raw_with_zstd_sibling_removed_only_when_bound(lane, kind):
    source, output = lane
    raw = output / "2026-10-16.db"
    raw.write_bytes(source.read_bytes())
    payload = {
        "valid": source.read_bytes(),
        "mismatched": source.read_bytes()[:-1] + bytes([source.read_bytes()[-1] ^ 1]),
        "not-sqlite": b"x" * raw.stat().st_size,
    }.get(kind)
    archive = output / "2026-10-16.db.zst"
    archive.write_bytes(_zstd_bytes(payload) if payload is not None else b"\x28\xb5\x2f\xfd broken")
    deleted, retained = backup._cleanup_raw_snapshots(output)
    if kind == "valid":
        assert deleted == ["2026-10-16.db"] and not raw.exists()
    else:
        assert retained == ["2026-10-16.db"] and raw.read_bytes() == source.read_bytes()
    assert archive.exists()


def test_raw_with_only_legacy_gzip_sibling_still_cleans_up(lane):
    source, output = lane
    raw = output / "2026-10-16.db"
    raw.write_bytes(source.read_bytes())
    (output / "2026-10-16.db.gz").write_bytes(gzip.compress(source.read_bytes()))
    assert backup._cleanup_raw_snapshots(output) == (["2026-10-16.db"], [])


# --- retention ------------------------------------------------------------------------------


def test_local_retention_keeps_newest_verified_copy_of_each_kind(tmp_path):
    names = [f"2026-10-0{d}.db.gz" for d in (1, 2, 3)] + [f"2026-10-0{d}.db.zst" for d in (4, 5, 6, 7)]
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    deleted = backup.prune_local_compressed_snapshots(tmp_path, verified_drive_names=set(names))
    assert deleted == ["2026-10-04.db.zst", "2026-10-02.db.gz", "2026-10-01.db.gz"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "2026-10-03.db.gz",  # newest verified gzip: the rollback format's last verified copy
        "2026-10-05.db.zst",
        "2026-10-06.db.zst",
        "2026-10-07.db.zst",
    ]


def test_local_retention_never_deletes_unverified_zst(tmp_path):
    for name in (
        "2026-10-01.db.zst",
        "2026-10-02.db.gz",
        "2026-10-03.db.zst",
        "2026-10-04.db.zst",
        "2026-10-05.db.zst",
    ):
        (tmp_path / name).write_bytes(b"x")
    deleted = backup.prune_local_compressed_snapshots(
        tmp_path, verified_drive_names={"2026-10-02.db.gz", "2026-10-03.db.zst"}
    )
    assert deleted == []


class _DriveFiles:
    def __init__(self, files):
        self.files_list, self.trashed = files, []

    def list(self, **kwargs):  # noqa: ARG002
        return SimpleNamespace(execute=lambda: {"files": self.files_list})

    def update(self, *, fileId, body, supportsAllDrives):  # noqa: N803, ARG002
        assert body == {"trashed": True}
        self.trashed.append(fileId)
        return SimpleNamespace(execute=lambda: {})


def _prune(monkeypatch, files, *, keep, verified):
    drive = _DriveFiles(files)
    monkeypatch.setattr(backup, "ensure_drive_folder_chain", lambda service, parts: "folder")
    trashed = backup.prune_drive_backups(
        SimpleNamespace(files=lambda: drive),
        folder_parts=["x"],
        retention_policy=backup.DriveRetentionPolicy(keep_latest=keep),
        **({} if verified is ... else {"verified_file_ids": verified}),
    )
    return sorted(drive.trashed), sorted(trashed)


def test_drive_retention_counts_both_formats_and_keeps_newest_verified_of_each(monkeypatch):
    files = [{"id": f"gz{d}", "name": f"2026-10-0{d}.db.gz"} for d in (1, 2, 3)] + [
        {"id": f"zst{d}", "name": f"2026-10-0{d}.db.zst"} for d in (4, 5, 6)
    ]
    ids, names = _prune(monkeypatch, files, keep=2, verified={item["id"] for item in files})
    assert ids == ["gz1", "gz2", "zst4"]
    assert names == ["2026-10-01.db.gz", "2026-10-02.db.gz", "2026-10-04.db.zst"]


def _transition(new_suffix, old_suffix):
    """Sol's r1 repro shape: seven newer objects of the new format fill keep-7, the newest
    old-format upload FAILED verification, and the one before it is the last verified copy."""
    files = [{"id": f"new-{d}", "name": f"2026-10-{d}{new_suffix}"} for d in range(14, 21)]
    files += [
        {"id": "old-failed", "name": f"2026-10-13{old_suffix}"},
        {"id": "old-last-verified", "name": f"2026-10-12{old_suffix}"},
        {"id": "old-older-verified", "name": f"2026-10-11{old_suffix}"},
    ]
    verified = {f"new-{d}" for d in range(14, 21)} | {"old-last-verified", "old-older-verified"}
    return files, verified


@pytest.mark.parametrize(
    "new_suffix,old_suffix",
    [(".db.zst", ".db.gz"), (".db.gz", ".db.zst")],
    ids=["zstd-after-gzip", "gzip-rollback-after-zstd"],
)
def test_drive_retention_never_trashes_last_verified_copy_behind_a_failed_newer_one(
    monkeypatch, new_suffix, old_suffix
):
    files, verified = _transition(new_suffix, old_suffix)
    ids, _ = _prune(monkeypatch, files, keep=backup.DEFAULT_DRIVE_KEEP, verified=verified)
    assert "old-last-verified" not in ids, "the last verified copy of the old format was trashed"
    assert ids == ["old-failed", "old-older-verified"]


@pytest.mark.parametrize("verified", [..., None, set()], ids=["omitted", "unknown", "empty"])
def test_drive_retention_without_verification_evidence_trashes_nothing(monkeypatch, verified):
    files, _ = _transition(".db.zst", ".db.gz")
    assert _prune(monkeypatch, files, keep=2, verified=verified) == ([], [])


def test_drive_retention_trashes_nothing_of_a_format_with_no_verified_copy(monkeypatch):
    files, _ = _transition(".db.zst", ".db.gz")
    verified = {f"new-{d}" for d in range(14, 21)}  # no gzip object has verification evidence
    ids, _ = _prune(monkeypatch, files, keep=2, verified=verified)
    assert ids == [f"new-{d}" for d in range(14, 19)]


def _receipt(file_id, *, verified, **extra):
    return {"drive_file": {"id": file_id, "name": "x"}, "uploaded": True, "verified": verified, **extra}


def test_verified_drive_ids_come_from_the_latest_receipt_per_object(tmp_path):
    log = tmp_path / "backup-daily.log"
    rows = [
        _receipt("a", verified=False, verification_pending=True),
        _receipt("a", verified=True),
        _receipt("b", verified=True),
        _receipt("b", verified=False, verification_error="snapshot aged beyond weekly window"),
        _receipt("c", verified=True, drive_md5_match=False),
        _receipt("d", verified=True, verification_error="md5 mismatch"),
        {"drive_file": "legacy-string", "uploaded": True, "verified": True},
        _receipt("e", verified=True),
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert backup._verified_drive_file_ids(log) == {"a", "e"}


def test_unreadable_receipts_fail_closed(tmp_path):
    log = tmp_path / "backup-daily.log"
    assert backup._verified_drive_file_ids(log) is None  # missing
    log.write_text(
        json.dumps(_receipt("a", verified=True)) + "\nnot json\n" + json.dumps(_receipt("b", verified=True)) + "\n"
    )
    assert backup._verified_drive_file_ids(log) is None  # malformed mid-log


def _progress_noise(count: int, total: int = 5242019358) -> bytes:
    """The exact legacy stdout shape #1418 classifies as known noise (production: 2,393 lines)."""
    return b"".join(f"drive upload progress: {8388608 * (i + 1)}/{total} bytes\n".encode() for i in range(count))


def test_known_progress_noise_before_a_verified_receipt_still_yields_evidence(tmp_path):
    log = tmp_path / "backup-daily.log"
    log.write_bytes(_progress_noise(2393) + (json.dumps(_receipt("last-good", verified=True)) + "\n").encode())
    assert backup._verified_drive_file_ids(log) == {"last-good"}


def test_uncertain_line_after_a_verified_receipt_yields_no_evidence_and_trashes_nothing(tmp_path, monkeypatch):
    """The uncertain line could have been the receipt revoking that verification (Sol #1418/#1421)."""
    log = tmp_path / "backup-daily.log"
    log.write_bytes(
        _progress_noise(2393)
        + (json.dumps(_receipt("old-last-verified", verified=True)) + "\n").encode()
        + b'{"drive_file": {"id": "old-last-verified"}, "uploaded": true, "verified": fa\n'
    )
    evidence = backup._verified_drive_file_ids(log)
    assert evidence is None
    files, _ = _transition(".db.zst", ".db.gz")
    assert _prune(monkeypatch, files, keep=2, verified=evidence) == ([], [])


def test_run_backup_feeds_receipt_verified_ids_to_drive_retention(tmp_path, monkeypatch):
    from tests.test_backup_daily import _stub_verified_backup_run

    snapshot = tmp_path / "2026-10-20.db.zst"
    snapshot.write_bytes(b"synthetic")
    log = tmp_path / "receipt.jsonl"
    log.write_text(json.dumps(_receipt("old-last-verified", verified=True)) + "\n")
    _stub_verified_backup_run(backup, monkeypatch, snapshot, SimpleNamespace())
    seen = {}
    monkeypatch.setattr(backup, "prune_drive_backups", lambda *a, **kw: seen.update(kw) or [])
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_RETENTION", "1")
    result = backup.run_backup(db_path=tmp_path / "fixture.db", staging_dir=tmp_path, log_path=log)
    assert result["verified"] is True
    assert {"old-last-verified", result["drive_file"]["id"]} <= seen["verified_file_ids"]


# --- receipt readers ------------------------------------------------------------------------


def test_receipt_readers_accept_both_suffixes(tmp_path):
    from brainlayer.health_check import _jsonl_backup_attempt_time

    for suffix in (".db.zst", ".db.gz"):
        assert backup._parse_snapshot_date(f"2026-10-20{suffix}") == dt.date(2026, 10, 20)
        assert backup._snapshot_created_time({"snapshot": f"/b/2026-10-20{suffix}"}) is not None
        assert _jsonl_backup_attempt_time({"snapshot": f"/b/2026-10-20{suffix}"}) is not None
    assert backup._parse_snapshot_date("2026-10-20.db") is None
    assert backup._parse_snapshot_date("2026-10-20.db.zst.tmp") is None
    (tmp_path / ".weekly-backup-pin.json").write_text("{not json")
    (tmp_path / "2026-10-20.db.zst").write_bytes(b"z")
    (tmp_path / "2026-10-19.db.gz").write_bytes(b"g")
    assert backup._protected_snapshot_names(tmp_path) == {"2026-10-20.db.zst", "2026-10-19.db.gz"}


def test_upload_declares_zstd_content_type(tmp_path, monkeypatch):
    archive = tmp_path / "2026-10-20.db.zst"
    archive.write_bytes(b"z")
    seen = {}

    class Stop(Exception):
        pass

    def post(url, *, headers, data, timeout):  # noqa: ARG001
        seen.update(headers)
        raise Stop

    monkeypatch.setattr(backup.requests, "post", post)
    with pytest.raises(Stop):
        backup.upload_file_to_drive_raw(archive, "folder", SimpleNamespace(token="t"), machine_id="test-machine")
    assert seen["X-Upload-Content-Type"] == "application/zstd"


def test_experiment_snapshot_materializes_from_zstd_backup(tmp_path, monkeypatch):
    from brainlayer.eval.experiment_store import materialize_snapshot

    monkeypatch.setattr(backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=10**12))
    source = tmp_path / "source.db"
    _create_source_db(source, chunk_count=2)
    archive = tmp_path / "2026-10-20.db.zst"
    archive.write_bytes(_zstd_bytes(source.read_bytes()))
    target = materialize_snapshot(backup_gzip_path=archive, snapshot_db_path=tmp_path / "exp" / "snap.db")
    assert target.read_bytes() == source.read_bytes()


# --- weekly gate ----------------------------------------------------------------------------


def test_weekly_gate_accepts_verified_local_zstd_snapshot(local_lane, tmp_path, monkeypatch):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    assert receipt["snapshot"].endswith(".db.zst") and receipt["compressor"] == "zstd"
    assert receipt["local_verified"] is True and receipt["local_integrity_pragma"] == "integrity_check"
    monkeypatch.setattr(maintenance, "_run_bounded_weekly_backup", lambda *_: pytest.fail("second snapshot"))
    assert maintenance._weekly_backup(config_for(local_lane, tmp_path))["snapshot"] == receipt["snapshot"]
    observed = json.loads(local_lane["log_path"].read_text().splitlines()[-1])
    assert observed["compress_duration_seconds"] is not None


# --- stop / kill ----------------------------------------------------------------------------


def test_compressor_child_stays_in_the_callers_process_group(tmp_path, monkeypatch):
    """The calendar stops a step with killpg; a child in a new session would escape it."""
    pidfile = _fake_zstd(tmp_path, monkeypatch)
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=10**12))
    seen = {}

    def observe():
        pid = _wait_for_pid(pidfile)
        seen["pgid"] = os.getpgid(pid)
        os.kill(pid, signal.SIGKILL)

    watcher = threading.Thread(target=observe)
    watcher.start()
    with pytest.raises(RuntimeError, match="exited"):
        backup._run_zstd(str(tmp_path / "fake-zstd"), ["-3"], output=tmp_path / "out.zst", floor_dir=tmp_path)
    watcher.join()
    assert seen["pgid"] == os.getpgid(0)


def test_stop_signal_during_compression_kills_and_reaps_the_child(tmp_path, monkeypatch):
    pidfile = _fake_zstd(tmp_path, monkeypatch)
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=10**12))
    previous = signal.signal(signal.SIGTERM, backup._raise_backup_stopped)
    pid = {}

    def stop():
        pid["value"] = _wait_for_pid(pidfile)
        os.kill(os.getpid(), signal.SIGTERM)

    watcher = threading.Thread(target=stop)
    try:
        watcher.start()
        with pytest.raises(backup.BackupStoppedError):
            backup._run_zstd(str(tmp_path / "fake-zstd"), ["-3"], output=tmp_path / "out.zst", floor_dir=tmp_path)
    finally:
        watcher.join()
        signal.signal(signal.SIGTERM, previous)
    assert _gone(pid["value"]), "compressor outlived the stopped backup"


def test_floor_breach_during_compression_kills_the_child(tmp_path, monkeypatch):
    pidfile = _fake_zstd(tmp_path, monkeypatch)
    started = threading.Event()
    real_free = {"value": 10**12}

    def disk_usage(*a):
        if started.is_set():
            real_free["value"] = 0
        return SimpleNamespace(free=real_free["value"])

    monkeypatch.setattr(backup.shutil, "disk_usage", disk_usage)
    threading.Thread(target=lambda: (_wait_for_pid(pidfile), started.set())).start()
    with pytest.raises(RuntimeError, match="raw free space floor breached"):
        backup._run_zstd(str(tmp_path / "fake-zstd"), ["-3"], output=tmp_path / "out.zst", floor_dir=tmp_path)
    assert _gone(_wait_for_pid(pidfile))


def test_calendar_process_group_stop_leaves_no_orphan_compressor(tmp_path, monkeypatch):
    """End to end: a backup child in its own group is killpg'd, as the calendar stops a step."""
    pidfile = _fake_zstd(tmp_path, monkeypatch)
    code = (
        "import pathlib, sys\n"
        "from brainlayer import backup_daily as b\n"
        "t = pathlib.Path(sys.argv[1])\n"
        "b._run_zstd(str(t / 'fake-zstd'), ['-3'], output=t / 'out.zst', floor_dir=t)\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "BRAINBAR_SOCKET_PATH": "/nonexistent/brainbar.sock",
        "BRAINLAYER_FORBID_BRAINBAR_SOCKET": "1",
    }
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], env=env, start_new_session=True)
    try:
        compressor = _wait_for_pid(pidfile)
        assert os.getpgid(compressor) == child.pid
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert _gone(compressor), "compressor orphaned after the process-group stop"
