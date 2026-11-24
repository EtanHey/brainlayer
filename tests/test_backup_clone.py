"""Real SQLite writer/checkpoint interleavings; source payload reads use a separate process."""

import json
import os
import plistlib
import select
import signal
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from brainlayer import backup_clone as clone

needs_clone_control = pytest.mark.skipif(
    not (hasattr(sqlite3.Connection, "setconfig") and hasattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE")),
    reason="Clone snapshots require checkpoint-on-close control; this runtime uses VACUUM fallback",
)


ACTOR = """
import sqlite3,sys,json
c=sqlite3.connect(sys.argv[1], isolation_level=None, timeout=0)
c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA wal_autocheckpoint=0')
c.execute('CREATE TABLE chunks(id INTEGER PRIMARY KEY, value INTEGER)')
c.execute('CREATE TABLE mirror(id INTEGER PRIMARY KEY, value INTEGER)')
c.execute('BEGIN'); c.execute('INSERT INTO chunks VALUES(1,1)'); c.execute('INSERT INTO mirror VALUES(1,1)'); c.commit()
print('READY',flush=True)
for line in sys.stdin:
    command=line.strip()
    if command=='write':
        try:
            c.execute('BEGIN IMMEDIATE'); c.execute('UPDATE chunks SET value=value+1'); c.execute('UPDATE mirror SET value=value+1'); c.commit()
            print('COMMITTED',flush=True)
        except sqlite3.OperationalError:
            c.rollback(); print('BUSY',flush=True)
    elif command in {'PASSIVE','TRUNCATE'}:
        print(json.dumps(c.execute('PRAGMA wal_checkpoint('+command+')').fetchone()),flush=True)
c.close()
"""


@pytest.mark.parametrize("filesystem,expected", [("apfs", True), ("hfs", False)])
def test_support_probe_resolves_staging_directory_to_volume(tmp_path, monkeypatch, filesystem, expected):
    from types import SimpleNamespace

    source = tmp_path / "db"
    source.touch()
    monkeypatch.setattr(clone.sys, "platform", "darwin")

    def run(argv, **kwargs):
        assert kwargs["timeout"] == 5
        if argv[0] == "/bin/df":
            assert argv == ["/bin/df", "-P", str(tmp_path)]
            return SimpleNamespace(
                stdout=b"Filesystem 512-blocks Used Available Capacity Mounted on\n/dev/disk3s5 1 1 1 1% /System/Volumes/Data\n"
            )
        assert argv == ["/usr/sbin/diskutil", "info", "-plist", "/dev/disk3s5"]
        return SimpleNamespace(stdout=plistlib.dumps({"FilesystemType": filesystem}))

    monkeypatch.setattr(clone.subprocess, "run", run)
    assert clone.supported(source, tmp_path) is expected


def ask(process, command=None):
    if command is not None:
        process.stdin.write(command + "\n")
        process.stdin.flush()
    assert select.select([process.stdout], [], [], 5)[0], "SQLite actor did not answer"
    answer = process.stdout.readline().strip()
    assert answer, "SQLite actor exited early"
    return answer


@pytest.fixture
def source_actor(monkeypatch):
    root = Path.cwd() / ".build.noindex" / "pytest"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        path = Path(directory)
        source = path / "source.db"
        actor = subprocess.Popen(
            [sys.executable, "-c", ACTOR, str(source)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert ask(actor) == "READY"
            monkeypatch.setattr(clone, "supported", lambda *args: True)
            yield source, path / "snapshot.db", actor
        finally:
            actor.stdin.close()
            try:
                actor.wait(timeout=5)
            finally:
                if actor.poll() is None:
                    actor.kill()
                    actor.wait()
                actor.stdout.close()


def external_copy(source, target):
    # Portable analogue of cp-c for consistency tests. Real APFS proof is separate.
    subprocess.run(
        [sys.executable, "-c", "import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2])", str(source), str(target)],
        check=True,
        timeout=5,
    )


@needs_clone_control
def test_writer_and_checkpoint_between_file_clones_cannot_tear_snapshot(source_actor, monkeypatch):
    source, target, actor = source_actor
    events = []

    def copy(path, destination):
        external_copy(path, destination)
        events.append(ask(actor, "write"))
        assert events[-1] == "BUSY"
        if path == source:
            checkpoint = json.loads(ask(actor, "PASSIVE"))
            assert checkpoint[0] == 0

    monkeypatch.setattr(clone, "_copy", copy)
    info = clone.snapshot(source, target)
    assert info.st_ino == target.stat().st_ino
    assert events == ["BUSY", "BUSY"]
    assert ask(actor, "write") == "COMMITTED"  # Source is released before validation/compression.
    with sqlite3.connect(f"{target.as_uri()}?immutable=1", uri=True) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("SELECT value FROM chunks").fetchone() == (1,)
        assert connection.execute("SELECT value FROM mirror").fetchone() == (1,)
    assert not Path(f"{target}-wal").exists()


@needs_clone_control
def test_broken_barrier_torn_pair_is_rejected_not_published(source_actor, monkeypatch):
    source, target, actor = source_actor
    connect = sqlite3.connect

    class BrokenBarrier:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, sql):
            if sql == "BEGIN IMMEDIATE":
                return None  # Mutant: the exact missing lock that permits a torn file pair.
            return self.connection.execute(sql)

    monkeypatch.setattr(clone.sqlite3, "connect", lambda *a, **kw: BrokenBarrier(connect(*a, **kw)))

    def copy(path, destination):
        external_copy(path, destination)
        if path == source:
            assert ask(actor, "write") == "COMMITTED"
            assert json.loads(ask(actor, "TRUNCATE")) == [0, 0, 0]

    monkeypatch.setattr(clone, "_copy", copy)
    with pytest.raises(clone.CloneUnavailable, match="identity changed"):
        clone.snapshot(source, target)
    assert not target.exists()


@needs_clone_control
def test_copy_failure_releases_barrier_and_removes_private_attempt(source_actor, monkeypatch):
    source, target, actor = source_actor

    def fail(*args):
        raise OSError("synthetic clonefile failure")

    monkeypatch.setattr(clone, "_copy", fail)
    with pytest.raises(clone.CloneUnavailable, match="synthetic clonefile failure"):
        clone.snapshot(source, target)
    assert ask(actor, "write") == "COMMITTED"
    assert not target.exists()
    assert not list(target.parent.glob(".brainlayer-clone-*"))


def test_non_apfs_never_opens_source(source_actor, monkeypatch):
    source, target, _ = source_actor
    monkeypatch.setattr(clone, "supported", lambda *a: False)
    monkeypatch.setattr(clone.sqlite3, "connect", lambda *a, **kw: pytest.fail("opened unsupported source"))
    with pytest.raises(clone.CloneUnavailable, match="not on one APFS"):
        clone.snapshot(source, target)


@needs_clone_control
def test_large_wal_falls_back_without_copying(source_actor, monkeypatch):
    source, target, actor = source_actor
    monkeypatch.setattr(clone, "_copy", lambda *a: pytest.fail("unbounded WAL cloned"))
    with pytest.raises(clone.CloneUnavailable, match="outgrew admitted clone recovery"):
        clone.snapshot(source, target, max_recovery_bytes=1)
    assert ask(actor, "write") == "COMMITTED"


def test_existing_target_is_never_replaced(source_actor):
    source, target, _ = source_actor
    target.write_bytes(b"peer")
    with pytest.raises(FileExistsError):
        clone.snapshot(source, target)
    assert target.read_bytes() == b"peer"


@needs_clone_control
def test_last_source_connection_does_not_checkpoint_on_close(source_actor, monkeypatch):
    source, target, _ = source_actor
    orphan = source.parent / "orphan.db"
    external_copy(source, orphan)
    wal = Path(f"{orphan}-wal")
    external_copy(Path(f"{source}-wal"), wal)
    before = clone._identity(wal)
    monkeypatch.setattr(clone, "_copy", external_copy)
    clone.snapshot(orphan, target)
    assert clone._identity(wal) == before


@needs_clone_control
def test_stop_during_clone_releases_writer_and_private_files(source_actor, monkeypatch):
    source, target, actor = source_actor

    def stop(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(clone, "_copy", stop)
    with pytest.raises(KeyboardInterrupt):
        clone.snapshot(source, target)
    assert ask(actor, "write") == "COMMITTED"
    assert not target.exists()
    assert not list(target.parent.glob(".brainlayer-clone-*"))


@pytest.fixture
def artifact_lane(source_actor, monkeypatch):
    from brainlayer import backup_daily as backup

    source, _, actor = source_actor
    output = source.parent / "backups"
    output.mkdir()
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(output / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_BACKUP_COMPRESSOR", "gzip")
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST", "m1")
    monkeypatch.setattr(clone, "_copy", external_copy)
    monkeypatch.setattr(backup, "_brainbar_writer_started_at", lambda *args: 1)
    monkeypatch.setattr(backup, "_probe_raw_reclamation", lambda *args: (False, "fixture"))
    monkeypatch.setattr(backup, "_should_run_full_verify", lambda date: True)

    def vacuum(target, **kwargs):
        with sqlite3.connect(source) as connection:
            connection.execute("VACUUM INTO ?", (str(target),))
        return target.stat()

    monkeypatch.setattr(backup, "request_brainbar_vacuum_into", vacuum)
    return backup, source, output, actor


def test_clone_peak_does_not_charge_a_raw_database_or_restore():
    from brainlayer import backup_daily as backup

    db_size, compressed = 26 * 1024**3, 12 * 1024**3
    for full in (False, True):
        wal, divergence = 100 * 1024**2, 2 * 1024**3
        peak = backup._peak_backup_required_bytes(
            db_size, compressed, full=full, clone_mode=True, clone_wal_bytes=wal, clone_divergence_bytes=divergence
        )
        assert peak == compressed + wal + divergence + backup.MIN_RAW_FREE_BYTES + backup.COPY_CHUNK_BYTES
        assert peak < backup._peak_backup_required_bytes(db_size, compressed, full=full)


@needs_clone_control
def test_clone_artifact_integrity_and_streaming_full_verify_without_restore(artifact_lane, monkeypatch):
    backup, source, output, _ = artifact_lane
    artifact = backup.create_sqlite_backup_artifact(source, output, socket_path=source.parent / "unused.sock")
    assert artifact.snapshot_method == "apfs_clone"
    assert artifact.local_integrity == "ok"
    assert artifact.local_integrity_pragma == "integrity_check"
    assert artifact.raw_sha256 is not None
    assert artifact.peak_allocated_bytes is not None
    assert not list(output.glob("*.db"))
    monkeypatch.setattr(backup, "_decompress_archive_to", lambda *args: pytest.fail("restored payload allocated"))
    monkeypatch.setattr(
        backup, "download_drive_file_raw", lambda *args, **kwargs: pytest.fail("download file allocated")
    )
    monkeypatch.setattr(
        backup,
        "_drive_stream_hashes",
        lambda *args, **kwargs: (artifact.raw_sha256, backup._md5_file(artifact.gzip_path)),
    )
    result = backup.verify_sqlite_backup_artifact(artifact, full=True, service=object(), file_id="synthetic")
    assert result["verified"] is True
    assert result["verification_proof"] == "stream_sha256_of_integrity_checked_raw"


@needs_clone_control
def test_clone_archive_publication_cannot_replace_a_peer_file(artifact_lane, monkeypatch):
    backup, source, output, _ = artifact_lane
    copy = backup._copy_with_raw_floor
    peer = output / "2026-11-22.db.gz"

    def racing_peer(*args, **kwargs):
        copy(*args, **kwargs)
        peer.write_bytes(b"peer archive")

    monkeypatch.setattr(backup, "_copy_with_raw_floor", racing_peer)
    with pytest.raises(FileExistsError):
        backup.create_sqlite_backup_artifact(
            source, output, date_stamp="2026-11-22", socket_path=source.parent / "unused.sock"
        )
    assert peer.read_bytes() == b"peer archive"
    assert not list(output.glob("*.db"))


@needs_clone_control
def test_clone_failure_falls_back_to_vacuum_with_reason(artifact_lane, monkeypatch):
    backup, source, output, _ = artifact_lane

    def fail(*args):
        raise OSError("synthetic cp-c failure")

    monkeypatch.setattr(clone, "_copy", fail)
    artifact = backup.create_sqlite_backup_artifact(source, output, socket_path=source.parent / "unused.sock")
    assert artifact.snapshot_method == "vacuum"
    assert "synthetic cp-c failure" in artifact.clone_fallback
    assert artifact.local_integrity == "ok"


def test_weekly_vacuum_fallback_also_streams_without_restore(artifact_lane, monkeypatch):
    backup, source, output, _ = artifact_lane
    monkeypatch.setattr(clone, "supported", lambda *args: False)
    artifact = backup.create_sqlite_backup_artifact(
        source, output, socket_path=source.parent / "unused.sock", integrity_full=False
    )
    assert artifact.raw_sha256 is not None
    monkeypatch.setattr(backup, "_decompress_archive_to", lambda *args: pytest.fail("restored payload allocated"))
    monkeypatch.setattr(backup, "download_drive_file_raw", lambda *args, **kwargs: pytest.fail("download allocated"))
    monkeypatch.setattr(
        backup,
        "_drive_stream_hashes",
        lambda *args, **kwargs: (artifact.raw_sha256, backup._md5_file(artifact.gzip_path)),
    )
    assert backup.verify_sqlite_backup_artifact(artifact, full=True, service=object(), file_id="synthetic")["verified"]


def test_non_apfs_uses_vacuum_and_logs_reason(artifact_lane, monkeypatch, capsys):
    backup, source, output, _ = artifact_lane
    monkeypatch.setattr(clone, "supported", lambda *args: False)
    artifact = backup.create_sqlite_backup_artifact(source, output, socket_path=source.parent / "unused.sock")
    assert artifact.snapshot_method == "vacuum"
    assert "APFS clone support not confirmed" in capsys.readouterr().out


@needs_clone_control
@pytest.mark.parametrize("raises_error", [False, True])
def test_clone_integrity_failure_is_not_published_and_uses_fallback(artifact_lane, monkeypatch, raises_error):
    backup, source, output, _ = artifact_lane
    check = backup._sqlite_pragma_check
    seen = []

    def corrupt_clone(path, pragma, **kwargs):
        seen.append(path)
        if path.name == "snapshot.db" and raises_error:
            raise sqlite3.DatabaseError("synthetic corrupt clone")
        return "synthetic corrupt clone" if path.name == "snapshot.db" else check(path, pragma, **kwargs)

    monkeypatch.setattr(backup, "_sqlite_pragma_check", corrupt_clone)
    # Fixed date, distinct from the clean run's 2026-11-24 below: with today's date, this test failed on 24 Nov
    # itself because both runs targeted the same archive name.
    artifact = backup.create_sqlite_backup_artifact(
        source, output, date_stamp="2026-11-23", socket_path=source.parent / "unused.sock"
    )
    assert artifact.snapshot_method == "vacuum"
    assert "integrity_check failed" in artifact.clone_fallback
    assert artifact.local_integrity == "ok"
    assert not list(output.glob(".brainlayer-clone-*"))
    from brainlayer import job_alerts

    assert "integrity_check failed" in job_alerts.active_alerts()["backup-clone-integrity"]
    monkeypatch.setattr(backup, "_sqlite_pragma_check", check)
    backup.create_sqlite_backup_artifact(
        source, output, date_stamp="2026-11-24", socket_path=source.parent / "unused.sock"
    )
    assert "backup-clone-integrity" not in job_alerts.active_alerts()


def test_fallback_rechecks_legacy_peak_plus_host_floor_before_vacuum(artifact_lane, monkeypatch):
    from types import SimpleNamespace

    backup, source, output, _ = artifact_lane
    monkeypatch.setattr(backup, "_database_logical_size_bytes", lambda path: 26 * 1024**3)
    monkeypatch.setattr(backup, "_estimated_gzip_bytes", lambda *args: 12 * 1024**3)
    monkeypatch.setattr(
        backup.disk_space,
        "measure_disk_space",
        lambda *args, **kwargs: SimpleNamespace(
            free_now_bytes=30 * 1024**3, available_bytes=30 * 1024**3, source="fixture"
        ),
    )
    monkeypatch.setattr(
        clone, "snapshot", lambda *args, **kwargs: (_ for _ in ()).throw(clone.CloneUnavailable("failed"))
    )
    monkeypatch.setattr(
        backup, "request_brainbar_vacuum_into", lambda *args, **kwargs: pytest.fail("unbudgeted fallback")
    )
    with pytest.raises(RuntimeError, match="VACUUM fallback needs"):
        backup.create_sqlite_backup_artifact(source, output, socket_path=source.parent / "unused.sock")


def test_large_volume_growth_is_allowed_until_absolute_host_reserve(artifact_lane, monkeypatch):
    from types import SimpleNamespace

    backup, source, output, _ = artifact_lane
    free = [100 * 1024**3]
    monkeypatch.setattr(
        backup.disk_space, "measure_disk_space", lambda *args, **kwargs: SimpleNamespace(free_now_bytes=free[0])
    )
    guard = clone.SpaceGuard(output, output / "archive.zst", minimum_free=42 * 1024**3)
    free[0] = 50 * 1024**3
    guard(force=True)
    assert guard.peak_bytes == 50 * 1024**3
    free[0] = 42 * 1024**3 - 1
    with pytest.raises(clone.CloneUnavailable, match="free-space reserve breached"):
        guard(force=True)


def test_divergence_budget_is_write_rate_times_expected_snapshot_seconds(source_actor, monkeypatch):
    source, _, _ = source_actor
    monkeypatch.setenv("BRAINLAYER_BACKUP_CLONE_WRITE_BYTES_PER_SECOND", "8192")
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_EXPECTED_SNAPSHOT_SECONDS", "120")
    budget = clone.budget(source)
    assert budget["clone_divergence_bytes"] == 8192 * 120
    assert budget["clone_wal_bytes"] == Path(f"{source}-wal").stat().st_size + clone.WAL_GROWTH_SLACK_BYTES


@needs_clone_control
def test_floor_trip_during_compression_retries_capacity_checked_vacuum(artifact_lane, monkeypatch):
    backup, source, output, _ = artifact_lane
    copy = backup._copy_with_raw_floor
    calls = []

    def floor_once(*args, **kwargs):
        calls.append(True)
        copy(*args, **kwargs)
        if len(calls) == 1:
            raise clone.CloneUnavailable("absolute free-space reserve breached")

    monkeypatch.setattr(backup, "_copy_with_raw_floor", floor_once)
    artifact = backup.create_sqlite_backup_artifact(source, output, socket_path=source.parent / "unused.sock")
    assert artifact.snapshot_method == "vacuum"
    assert "absolute free-space reserve" in artifact.clone_fallback
    assert len(calls) == 2
    assert backup.verify_sqlite_backup_artifact(artifact)["verified"] is True


@needs_clone_control
def test_floor_trip_without_fallback_capacity_defers(artifact_lane, monkeypatch):
    from types import SimpleNamespace

    from brainlayer.backup_heavy_gate import BackupGateDeferred

    backup, source, output, _ = artifact_lane
    monkeypatch.setenv("BRAINLAYER_DB", str(source))
    free = [10**12]
    monkeypatch.setattr(
        backup.disk_space,
        "measure_disk_space",
        lambda *a, **kw: SimpleNamespace(
            free_now_bytes=free[0], available_bytes=free[0], source="fixture", describe=lambda: "fixture volume"
        ),
    )

    def floor(*args, **kwargs):
        free[0] = 1024**2
        raise clone.CloneUnavailable("absolute free-space reserve breached")

    monkeypatch.setattr(backup, "_copy_with_raw_floor", floor)
    monkeypatch.setattr(backup, "request_brainbar_vacuum_into", lambda *a, **kw: pytest.fail("unfunded fallback"))
    with pytest.raises(BackupGateDeferred, match="Insufficient free space"):
        backup.run_backup(source, staging_dir=output, upload=False)
    assert not list(output.glob("*.db"))


@needs_clone_control
def test_clone_floor_fallback_keeps_cumulative_sleep_and_observer(artifact_lane, monkeypatch):
    backup, source, output, _ = artifact_lane
    clocks = {"inclusive": 0, "awake": 0}
    tracker_type = backup.backup_sleep.SleepTracker
    monkeypatch.setattr(
        backup.backup_sleep,
        "SleepTracker",
        lambda: tracker_type(inclusive=lambda: clocks["inclusive"], awake=lambda: clocks["awake"]),
    )
    copy = backup._copy_with_raw_floor
    vacuum = backup.request_brainbar_vacuum_into
    copies = []
    observations = []

    def copy_with_sleep(*args, **kwargs):
        copy(*args, **kwargs)
        copies.append(True)
        if len(copies) == 1:
            clocks["inclusive"] += 3600
            raise clone.CloneUnavailable("absolute free-space reserve breached")

    def vacuum_with_sleep(*args, **kwargs):
        result = vacuum(*args, **kwargs)
        clocks["inclusive"] += 120
        return result

    monkeypatch.setattr(backup, "_copy_with_raw_floor", copy_with_sleep)
    monkeypatch.setattr(backup, "request_brainbar_vacuum_into", vacuum_with_sleep)
    artifact = backup.create_sqlite_backup_artifact(
        source, output, socket_path=source.parent / "unused.sock", snapshot_telemetry_callback=observations.append
    )
    assert artifact.snapshot_method == "vacuum"
    assert artifact.snapshot_wall_seconds == artifact.snapshot_slept_seconds == 3720
    assert [span["seconds"] for span in artifact.snapshot_sleep_intervals] == [3600, 120]
    assert observations[-1]["snapshot_slept_seconds"] == 3720


@needs_clone_control
def test_clone_observer_failure_after_publication_still_cleans_owned_raw(artifact_lane):
    backup, source, output, _ = artifact_lane

    def failed_observer(metrics):
        raise ValueError("synthetic observation failure")

    with pytest.raises(ValueError, match="synthetic observation failure"):
        backup.create_sqlite_backup_artifact(
            source,
            output,
            socket_path=source.parent / "unused.sock",
            snapshot_telemetry_callback=failed_observer,
        )
    assert not list(output.glob("*.db"))
    assert not list(output.glob("*.db.gz"))


def test_supervised_capacity_deferral_keeps_durable_job_alert(artifact_lane, monkeypatch):
    from brainlayer import job_alerts
    from brainlayer.backup_heavy_gate import BackupGateDeferred

    backup, source, output, _ = artifact_lane
    monkeypatch.setenv(backup.BACKUP_SUPERVISED_CHILD_ENV, "1")
    monkeypatch.setenv("BRAINLAYER_DB", str(source))
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(output / "alerts.json"))

    def deferred(*args, **kwargs):
        raise BackupGateDeferred("absolute reserve: needs 66 GiB, has 60 GiB")

    monkeypatch.setattr(backup, "run_backup", deferred)
    assert backup._run_backup_process(100) == 75
    assert "needs 66 GiB, has 60 GiB" in job_alerts.active_alerts()["backup-daily"]


def test_missing_checkpoint_control_uses_vacuum_without_cloning(artifact_lane, monkeypatch):
    backup, source, output, _ = artifact_lane
    monkeypatch.delattr(clone.sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", raising=False)
    monkeypatch.setattr(clone, "_copy", lambda *args: pytest.fail("cloning without source close protection"))
    artifact = backup.create_sqlite_backup_artifact(source, output, socket_path=source.parent / "unused.sock")
    assert artifact.snapshot_method == "vacuum"
    assert "cannot disable source checkpoint on close" in artifact.clone_fallback
    assert artifact.local_integrity == "ok"
    assert backup.verify_sqlite_backup_artifact(artifact)["verified"] is True


@needs_clone_control
@pytest.mark.parametrize("sig", [signal.SIGALRM, signal.SIGTERM, signal.SIGINT])
def test_signal_at_integrity_callback_entry_stops_without_fallback(artifact_lane, monkeypatch, sig):
    backup, source, output, _ = artifact_lane
    with sqlite3.connect(source) as connection:
        connection.executemany("INSERT INTO chunks VALUES(?,1)", ((i,) for i in range(2, 20002)))
    monkeypatch.setattr(backup, "_backup_interrupt", None, raising=False)
    monkeypatch.setattr(backup, "request_brainbar_vacuum_into", lambda *a, **k: pytest.fail("fallback after stop"))
    fired = []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code.co_name == "progress"
            and frame.f_code.co_filename == backup.__file__
            and not fired
        ):
            fired.append(True)
            signal.raise_signal(sig)
        return trace

    previous_handler = signal.signal(
        sig, backup._raise_backup_timeout if sig == signal.SIGALRM else backup._raise_backup_stopped
    )
    previous_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        expected = backup.BackupTimeoutError if sig == signal.SIGALRM else backup.BackupStoppedError
        with pytest.raises(expected):
            backup.create_sqlite_backup_artifact(source, output, socket_path=source.parent / "unused.sock")
    finally:
        sys.settrace(previous_trace)
        signal.signal(sig, previous_handler)
    assert fired
    assert not list(output.glob(".brainlayer-clone-*"))


@needs_clone_control
def test_direct_clone_timeout_is_not_wrapped(source_actor, monkeypatch):
    from brainlayer import backup_daily as backup

    source, target, _ = source_actor
    monkeypatch.setattr(clone, "_copy", external_copy)

    def timeout(path):
        raise backup.BackupTimeoutError("deadline")

    with pytest.raises(backup.BackupTimeoutError):
        clone.snapshot(source, target, validate=timeout)


def test_stale_clone_sweep_retains_live_recent_linked_and_unknown(tmp_path, monkeypatch):
    import time

    from brainlayer import backup_daily as backup

    old = time.time() - backup._configured_backup_attempt_max_age_seconds() - 100
    for name in ("dead", "live", "recent", "unknown", "linked"):
        directory = tmp_path / (".brainlayer-clone-" + name)
        directory.mkdir()
        raw = directory / "snapshot.db"
        raw.write_bytes(b"owned")
        if name == "live":
            (directory / ".owner").write_text(str(os.getpid()))
        if name == "unknown":
            (directory / "personal.txt").write_text("keep")
        if name == "linked":
            os.link(raw, tmp_path / "peer")
        if name != "recent":
            for p in directory.iterdir():
                os.utime(p, (old, old))
            os.utime(directory, (old, old))
    link = tmp_path / ".brainlayer-clone-symlink"
    link.symlink_to(tmp_path / ".brainlayer-clone-dead", target_is_directory=True)
    uid = os.getuid()
    with monkeypatch.context() as patch:
        patch.setattr(os, "getuid", lambda: uid + 1)
        assert backup._sweep_stale_clones(tmp_path) == []
    with (tmp_path / ".backup.lock").open("a+b") as lock:
        backup.fcntl.flock(lock, backup.fcntl.LOCK_EX | backup.fcntl.LOCK_NB)
        backup._sweep_clones_before_admission(tmp_path)
        assert (tmp_path / ".brainlayer-clone-dead").exists()
    assert backup._sweep_stale_clones(tmp_path) == [".brainlayer-clone-dead"]
    assert link.is_symlink()
    assert all((tmp_path / (".brainlayer-clone-" + n)).exists() for n in ("live", "recent", "unknown", "linked"))


def test_surviving_clone_growth_is_in_gate_admission(artifact_lane, monkeypatch):
    from types import SimpleNamespace

    from brainlayer import backup_heavy_gate as gate

    backup, source, output, _ = artifact_lane
    monkeypatch.setenv("BRAINLAYER_BACKUP_STAGING_DIR", str(output))
    monkeypatch.setattr(backup, "get_db_path", lambda: source)
    monkeypatch.setattr(backup, "_database_logical_size_bytes", lambda path: 1000)
    monkeypatch.setattr(backup, "_peak_backup_required_bytes", lambda *a, **k: 200)
    monkeypatch.setattr(backup.disk_space, "measure_disk_space", lambda *a, **k: SimpleNamespace(free_now_bytes=500))
    assert gate._disk_reason("m1") is None
    directory = output / ".brainlayer-clone-recent"
    directory.mkdir()
    (directory / "snapshot.db").write_bytes(b"x" * 1000)
    assert backup._owned_temp_inventory(output) == [(directory.name, 1000)]
    assert backup._clone_growth_reserve_bytes(output, 1000) == 1000
    assert "needs" in gate._disk_reason("m1")
    (output / "2026-11-23.db").write_bytes(b"raw")
    assert backup._clone_growth_reserve_bytes(output, 1000) == 2000
