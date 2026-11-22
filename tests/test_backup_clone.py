"""Real SQLite writer/checkpoint interleavings; source payload reads use a separate process."""

import json
import plistlib
import select
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from brainlayer import backup_clone as clone

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


def test_large_wal_falls_back_without_copying(source_actor, monkeypatch):
    source, target, actor = source_actor
    monkeypatch.setattr(clone, "MAX_WAL_BYTES", 1)
    monkeypatch.setattr(clone, "_copy", lambda *a: pytest.fail("unbounded WAL cloned"))
    with pytest.raises(clone.CloneUnavailable, match="bounded clone recovery"):
        clone.snapshot(source, target)
    assert ask(actor, "write") == "COMMITTED"


def test_existing_target_is_never_replaced(source_actor):
    source, target, _ = source_actor
    target.write_bytes(b"peer")
    with pytest.raises(FileExistsError):
        clone.snapshot(source, target)
    assert target.read_bytes() == b"peer"


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
        peak = backup._peak_backup_required_bytes(db_size, compressed, full=full, clone_mode=True)
        assert (
            peak
            == compressed
            + clone.MAX_WAL_BYTES
            + clone.COW_MARGIN_BYTES
            + backup.MIN_RAW_FREE_BYTES
            + backup.COPY_CHUNK_BYTES
        )
        assert peak < backup._peak_backup_required_bytes(db_size, compressed, full=full)


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
    artifact = backup.create_sqlite_backup_artifact(source, output, socket_path=source.parent / "unused.sock")
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


def test_clone_integrity_failure_is_not_published_and_uses_fallback(artifact_lane, monkeypatch):
    backup, source, output, _ = artifact_lane
    check = backup._sqlite_pragma_check
    seen = []

    def corrupt_clone(path, pragma, **kwargs):
        seen.append(path)
        return "synthetic corrupt clone" if path.name == "snapshot.db" else check(path, pragma, **kwargs)

    monkeypatch.setattr(backup, "_sqlite_pragma_check", corrupt_clone)
    artifact = backup.create_sqlite_backup_artifact(source, output, socket_path=source.parent / "unused.sock")
    assert artifact.snapshot_method == "vacuum"
    assert "integrity_check failed" in artifact.clone_fallback
    assert artifact.local_integrity == "ok"
    assert not list(output.glob(".brainlayer-clone-*"))


def test_fallback_rechecks_legacy_peak_plus_host_floor_before_vacuum(artifact_lane, monkeypatch):
    from types import SimpleNamespace

    backup, source, output, _ = artifact_lane
    monkeypatch.setattr(backup, "_database_logical_size_bytes", lambda path: 26 * 1024**3)
    monkeypatch.setattr(backup, "_estimated_gzip_bytes", lambda *args: 12 * 1024**3)
    monkeypatch.setattr(
        backup.disk_space,
        "measure_disk_space",
        lambda *args, **kwargs: SimpleNamespace(
            free_now_bytes=20 * 1024**3, available_bytes=20 * 1024**3, source="fixture"
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


def test_cow_growth_guard_fails_closed(artifact_lane, monkeypatch):
    from types import SimpleNamespace

    backup, source, output, _ = artifact_lane
    free = [10 * 1024**3]
    monkeypatch.setattr(
        backup.disk_space, "measure_disk_space", lambda *args, **kwargs: SimpleNamespace(free_now_bytes=free[0])
    )
    guard = clone.SpaceGuard(output, output / "archive.zst", minimum_free=2 * 1024**3)
    free[0] -= clone.COW_MARGIN_BYTES + clone.MAX_WAL_BYTES + 1
    with pytest.raises(clone.CloneUnavailable, match="CoW growth"):
        guard(force=True)
