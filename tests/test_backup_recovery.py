import gzip
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as daily
from brainlayer import backup_recovery as recovery


class Drive:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.queries = []

    def files(self):
        return self

    def list(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(execute=lambda: next(self.pages))


def candidate(**changes):
    return {
        "id": "file",
        "name": "2026-10-09.db.gz",
        "size": "42",
        "parents": ["folder"],
        "trashed": False,
        "md5Checksum": "a" * 32,
        "appProperties": {daily.DRIVE_MACHINE_PROPERTY: "test-machine"},
        **changes,
    }


def find(service):
    return recovery.find_verified_drive_match(
        service,
        folder_id="folder",
        name="2026-10-09.db.gz",
        size=42,
        md5="a" * 32,
        machine_id="test-machine",
    )


def test_candidate_matching_reads_all_pages_and_preserves_incomplete_objects():
    service = Drive(
        [
            {"files": [candidate(size="3", md5Checksum=None)], "nextPageToken": "next"},
            {"files": [candidate()]},
        ]
    )
    assert find(service)["id"] == "file"
    assert service.queries[1]["pageToken"] == "next"


@pytest.mark.parametrize(
    "item", [candidate(parents=["other"]), candidate(trashed=True), candidate(appProperties={}), candidate(size=None)]
)
def test_unknown_candidate_is_rejected(item):
    with pytest.raises(RuntimeError, match="unknown"):
        find(Drive([{"files": [item]}]))


def test_multiple_complete_matches_fail_instead_of_picking_one():
    with pytest.raises(RuntimeError, match="ambiguous"):
        find(Drive([{"files": [candidate(), candidate(id="second")]}]))
    assert find(Drive([{"files": [candidate(md5Checksum="b" * 32)]}])) is None


def test_folder_resolution_never_creates_missing_or_ambiguous_folders():
    assert (
        recovery.resolve_existing_folder(Drive([{"files": [{"id": "folder", "name": "archive"}]}]), ["archive"])
        == "folder"
    )
    for items in ([], [{"id": "one", "name": "archive"}, {"id": "two", "name": "archive"}]):
        with pytest.raises(RuntimeError, match="ambiguous"):
            recovery.resolve_existing_folder(Drive([{"files": items}]), ["archive"])


def source_pair(tmp_path, monkeypatch):
    monkeypatch.setenv(daily.BACKUP_LOG_PROVENANCE_ENV, "pytest")
    raw = tmp_path / "2026-10-09.db"
    with sqlite3.connect(raw) as db:
        db.execute("CREATE TABLE chunks (id TEXT)")
        db.execute("INSERT INTO chunks VALUES ('one')")
    compressed = tmp_path / "2026-10-09.db.gz"
    compressed.write_bytes(gzip.compress(raw.read_bytes()))
    receipt = {
        "attempted_at": "2026-10-09T00:17:07Z",
        "snapshot": str(compressed),
        "machine_id": "test-machine",
        "drive_folder": "archive",
        "bytes": compressed.stat().st_size,
        "sentinel_snapshot_chunks": 1,
        "uploaded": False,
        "verified": False,
        "backup_log_provenance": "pytest",
    }
    log = tmp_path / "backup.log"
    log.write_text(json.dumps(receipt) + "\n")
    return raw, compressed, receipt, log


def read_source(tmp_path, raw, log):
    return recovery.read_source_receipt(
        log,
        line_number=1,
        snapshot_date="2026-10-09",
        staging_dir=tmp_path,
        machine_id="test-machine",
        folder_parts=["archive"],
        expected_raw_bytes=raw.stat().st_size,
    )


def test_receipt_provenance_preserves_original_time_and_line_hash(tmp_path, monkeypatch):
    raw, compressed, receipt, log = source_pair(tmp_path, monkeypatch)
    original = log.read_bytes()
    result, facts = read_source(tmp_path, raw, log)
    assert result == receipt
    assert facts["source_attempted_at"] == receipt["attempted_at"]
    assert facts["source_receipt_sha256"] == hashlib.sha256(original).hexdigest()
    assert log.read_bytes() == original
    assert facts["bytes"] == compressed.stat().st_size


@pytest.mark.parametrize(
    "changes",
    [
        {"machine_id": "other"},
        {"snapshot": "/elsewhere/2026-10-09.db.gz"},
        {"bytes": True},
        {"sentinel_snapshot_chunks": 0},
        {"uploaded": True},
        {"attempted_at": "2026-10-09T00:17:07"},
        {"backup_log_provenance": "real"},
    ],
)
def test_receipt_rejects_unknown_or_mismatched_provenance(tmp_path, monkeypatch, changes):
    raw, _, receipt, log = source_pair(tmp_path, monkeypatch)
    log.write_text(json.dumps({**receipt, **changes}) + "\n")
    with pytest.raises(ValueError):
        read_source(tmp_path, raw, log)


def test_remote_verification_requires_checksum_parent_and_exact_identity():
    class Service:
        def files(self):
            return self

        def get(self, **kwargs):
            return SimpleNamespace(execute=lambda: self.metadata)

    service = Service()
    for changes in ({}, {"md5Checksum": None}, {"parents": ["wrong"]}, {"id": "wrong"}):
        service.metadata = candidate(**changes)
        kwargs = dict(
            file_id="file",
            expected_name="2026-10-09.db.gz",
            expected_size=42,
            expected_machine_id="test-machine",
            expected_md5="a" * 32,
            expected_parent="folder",
        )
        if changes:
            with pytest.raises(RuntimeError):
                daily.verify_drive_upload(service, **kwargs)
        else:
            assert daily.verify_drive_upload(service, **kwargs) == candidate()


def recovery_setup(tmp_path, monkeypatch, *, reuse=True):
    raw, compressed, source, log = source_pair(tmp_path, monkeypatch)
    alerts = []
    remote = candidate(
        size=str(compressed.stat().st_size), md5Checksum=hashlib.md5(compressed.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(recovery, "report", lambda key, reason: alerts.append(reason))
    monkeypatch.setattr(daily, "build_drive_service", lambda: object())
    monkeypatch.setattr(daily, "get_drive_credentials", lambda: object())
    monkeypatch.setattr(recovery, "resolve_existing_folder", lambda *args: "folder")
    monkeypatch.setattr(daily, "assert_drive_folder_owned_by_machine", lambda *args, **kwargs: None)
    monkeypatch.setattr(recovery, "find_verified_drive_match", lambda *args, **kwargs: remote if reuse else None)
    monkeypatch.setattr(daily, "verify_drive_upload", lambda *args, **kwargs: dict(remote))
    for name in ("create_sqlite_backup_artifact", "prune_drive_backups", "prune_local_gzip_snapshots"):
        monkeypatch.setattr(daily, name, lambda *args, **kwargs: pytest.fail("snapshot/retention forbidden"))
    kwargs = dict(
        snapshot_date="2026-10-09",
        source_receipt_line=1,
        expected_raw_bytes=raw.stat().st_size,
        staging_dir=tmp_path,
        log_path=log,
        machine_id="test-machine",
        folder_parts=["archive"],
    )
    return raw, compressed, source, log, alerts, remote, kwargs


def test_recovery_reuses_verified_copy_and_fsyncs_before_clearing(tmp_path, monkeypatch):
    raw, compressed, source, log, alerts, _, kwargs = recovery_setup(tmp_path, monkeypatch)
    original = log.read_bytes()
    raw_bytes, gzip_bytes = raw.read_bytes(), compressed.read_bytes()
    monkeypatch.setattr(daily, "upload_file_to_drive_raw", lambda *args, **kwargs: pytest.fail("reuse must not upload"))
    real_append = recovery._append_durable_receipt
    events = []
    monkeypatch.setattr(recovery.os, "fsync", lambda fd: events.append("fsync"))
    monkeypatch.setattr(recovery, "report", lambda key, reason: events.append("clear" if reason is None else "fail"))
    result = recovery.recover_existing_artifact(**kwargs)
    assert events == ["fsync", "clear"]
    assert result["verified"] and result["uploaded"]
    assert result["drive_action"] == "reuse_existing"
    assert result["source_attempted_at"] == source["attempted_at"]
    assert result["attempted_at"] != source["attempted_at"]
    assert log.read_bytes().startswith(original)
    assert raw.read_bytes() == raw_bytes and compressed.read_bytes() == gzip_bytes


def test_recovery_uploads_unique_name_from_verified_descriptor(tmp_path, monkeypatch):
    raw, compressed, _, log, alerts, remote, kwargs = recovery_setup(tmp_path, monkeypatch, reuse=False)
    names = []

    def upload(path, folder, credentials, **options):
        import os

        names.append(options["drive_name"])
        assert os.pread(options["source_descriptor"], compressed.stat().st_size, 0) == compressed.read_bytes()
        remote.update(name=options["drive_name"])
        remote["appProperties"]["brainlayer_recovery_attempt"] = options["attempt_id"]
        return dict(remote)

    monkeypatch.setattr(daily, "upload_file_to_drive_raw", upload)
    result = recovery.recover_existing_artifact(**kwargs)
    assert names == [f"2026-10-09-recovery-{result['attempt_id']}.db.gz"]
    assert daily._parse_snapshot_date(names[0]).isoformat() == "2026-10-09"
    assert alerts == [None]
    assert raw.exists() and compressed.exists()


@pytest.mark.parametrize("gate", ["size", "sentinel", "local", "remote", "durable", "swap"])
def test_recovery_failed_gate_preserves_alert_and_artifacts(tmp_path, monkeypatch, gate):
    raw, compressed, _, log, alerts, _, kwargs = recovery_setup(tmp_path, monkeypatch)
    if gate == "size":
        kwargs["expected_raw_bytes"] += 1
    elif gate == "sentinel":
        receipt = json.loads(log.read_text())
        receipt["sentinel_snapshot_chunks"] = 2
        log.write_text(json.dumps(receipt) + "\n")
    else:

        def fail(*args, **options):
            raise RuntimeError("gate failed")

        if gate == "local":
            monkeypatch.setattr(daily, "verify_preserved_artifact_pair", fail)
        elif gate == "remote":
            monkeypatch.setattr(daily, "verify_drive_upload", fail)
        elif gate == "durable":
            monkeypatch.setattr(recovery.os, "fsync", fail)
        else:

            def swap(*args, **options):
                raw.rename(raw.with_suffix(".preserved"))
                raw.write_bytes(b"replacement")
                return candidate()

            monkeypatch.setattr(recovery, "find_verified_drive_match", swap)
    with pytest.raises((ValueError, RuntimeError)):
        recovery.recover_existing_artifact(**kwargs)
    assert None not in alerts
    assert raw.exists() and compressed.exists()
    if gate != "durable":
        assert json.loads(log.read_text().splitlines()[-1])["verified"] is False


def test_recovery_nonblocking_lock_refuses_work(tmp_path, monkeypatch):
    import fcntl

    _, _, _, _, alerts, _, kwargs = recovery_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(daily, "verify_preserved_artifact_pair", lambda *args, **kwargs: pytest.fail("locked"))
    with (tmp_path / ".backup.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(daily.BackupAlreadyRunningError):
            recovery.recover_existing_artifact(**kwargs)
    assert alerts and None not in alerts


def test_recovered_old_snapshot_never_satisfies_fresh_backup_gate(tmp_path):
    import datetime as dt

    from brainlayer.health_check import _jsonl_backup_attempt_time

    now = dt.datetime.now(dt.UTC)
    old = now - dt.timedelta(days=5)
    receipt = dict(
        operation="recover_existing_artifact",
        attempted_at=now.isoformat(),
        source_attempted_at=old.isoformat(),
        db="synthetic.db",
        uploaded=True,
        verified=True,
        drive_file={"id": "file"},
    )
    log = tmp_path / "backup.log"
    log.write_text(json.dumps(receipt) + "\n")
    assert daily._recent_verified_backup_for_reuse(log, "synthetic.db", now=now, max_age_hours=24) is None
    assert _jsonl_backup_attempt_time(receipt) == old
    receipt.pop("source_attempted_at")
    assert _jsonl_backup_attempt_time(receipt) is None
