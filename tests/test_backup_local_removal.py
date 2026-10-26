"""Receipt-bound archive removal through the job; all files belong to tmp_path."""

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup


@pytest.fixture
def removal_lane(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setattr(backup, "local_backup_date", lambda *_: dt.date(2026, 10, 26))
    monkeypatch.setattr(backup, "get_drive_credentials", lambda: object())
    monkeypatch.setattr(backup, "build_drive_service", lambda *_: object())
    monkeypatch.setattr(backup, "ensure_drive_folder_chain", lambda *_: "folder")
    monkeypatch.setattr(backup, "assert_drive_folder_owned_by_machine", lambda *a, **kw: None)
    monkeypatch.setattr(backup, "verify_drive_upload", lambda *a, **kw: None)
    monkeypatch.setattr(
        backup,
        "upload_file_to_drive_raw",
        lambda path, *a, **kw: {
            "id": path.name,
            "name": path.name,
            "size": str(path.stat().st_size),
            "md5Checksum": "local-md5",
        },
    )
    monkeypatch.setattr(
        backup,
        "verify_sqlite_backup_artifact",
        lambda *a, **kw: {"verified": True, "verification_mode": "quick", "local_md5": "local-md5"},
    )
    snapshot = tmp_path / "2026-10-26.db.zst"

    def create(*args, **kwargs):
        snapshot.write_bytes(b"archive")
        return SimpleNamespace(
            gzip_path=snapshot, uncompressed_path=None, sentinel_chunks=1, local_retention_deleted=[]
        )

    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", create)
    return dict(
        db_path=tmp_path / "live.db",
        staging_dir=tmp_path,
        log_path=tmp_path / "receipts.jsonl",
        date_stamp="2026-10-26",
    )


@pytest.mark.parametrize(
    "pin,verified,opt_out,reason",
    [
        ("today", True, False, "weekly_pin_pending"),
        ("past", True, False, None),
        ("unreadable", True, False, "pin_unreadable"),
        (None, False, False, "not_verified"),
        (None, True, True, "opt_out"),
    ],
)
def test_run_records_removal_or_skip(removal_lane, monkeypatch, pin, verified, opt_out, reason):
    staging = removal_lane["staging_dir"]
    snapshot = staging / "2026-10-26.db.zst"
    if pin:
        (staging / ".weekly-backup-pin.json").write_text(
            "broken"
            if pin == "unreadable"
            else json.dumps({"date": "2026-10-26" if pin == "today" else "2026-10-25", "snapshot": str(snapshot)})
        )
    monkeypatch.setattr(
        backup, "verify_sqlite_backup_artifact", lambda *a, **kw: {"verified": verified, "local_md5": "local-md5"}
    )
    result = backup.run_backup(**removal_lane, remove_local_after_upload=not opt_out)
    assert result["local_removed"] is (reason is None)
    assert snapshot.exists() is (reason is not None)
    assert result.get("local_removed_skip_reason") == reason
    persisted = json.loads(removal_lane["log_path"].read_text().splitlines()[-1])
    assert persisted.get("local_removed_skip_reason") == reason


def _old_receipt(lane, name="2026-10-21.db.zst", **changes):
    path = lane["staging_dir"] / name
    path.write_bytes(b"older archive")
    return dict(
        db=str(lane["db_path"]),
        snapshot=str(path),
        local_md5="local-md5",
        uploaded=True,
        verified=True,
        machine_id="test-machine",
        drive_folder="/".join(backup.default_drive_folder_parts("test-machine")),
        backup_log_provenance=backup._backup_log_provenance(),
        drive_file={"id": name, "name": name, "size": str(path.stat().st_size), "md5Checksum": "local-md5"},
        **changes,
    )


@pytest.mark.parametrize("pin", [None, "unreadable", "past"])
def test_upload_only_sweeps_verified_older_archives_after_pin_expiry(removal_lane, pin):
    staging = removal_lane["staging_dir"]
    verified = _old_receipt(removal_lane)
    unverified = _old_receipt(removal_lane, "2026-10-25.db.gz")
    unverified["verified"] = False
    today = _old_receipt(removal_lane, "2026-10-26.db.zst")
    today["verified"] = False
    raw = staging / "2026-10-21.db"
    raw.write_bytes(b"raw")
    removal_lane["db_path"].write_bytes(b"live")
    if pin:
        (staging / ".weekly-backup-pin.json").write_text(
            "broken" if pin == "unreadable" else json.dumps({"date": "2026-10-21", "snapshot": verified["snapshot"]})
        )
    removal_lane["log_path"].write_text("".join(json.dumps(r) + "\n" for r in [verified, unverified, today]))
    result = backup.run_backup(**removal_lane, phase="upload")
    assert result["upload_state"] == "nothing_pending"
    assert not Path(verified["snapshot"]).exists()
    assert Path(unverified["snapshot"]).exists()
    assert Path(today["snapshot"]).exists()
    assert raw.read_bytes() == b"raw" and removal_lane["db_path"].read_bytes() == b"live"
    receipts = [json.loads(line) for line in removal_lane["log_path"].read_text().splitlines()]
    latest = {r["archive"]: r for r in receipts if r.get("event") == "local_sweep"}
    assert latest[verified["snapshot"]]["local_removed"] is True
    assert latest[unverified["snapshot"]]["local_removed_skip_reason"] == "not_verified"


@pytest.mark.parametrize("damage", ["name", "size", "id", "no_metadata", "revoked", "journal", "symlink", "opt_out"])
def test_sweep_requires_matching_latest_drive_receipt(removal_lane, damage):
    receipt = _old_receipt(removal_lane)
    path = Path(receipt["snapshot"])
    if damage in {"name", "size"}:
        receipt["drive_file"][damage] = "wrong"
    elif damage == "id":
        receipt["drive_file"]["id"] = ["malformed"]
    elif damage == "no_metadata":
        receipt.pop("drive_file")
    elif damage == "symlink":
        target = path.with_name("foreign")
        path.rename(target)
        path.symlink_to(target)
    lines = [receipt]
    if damage == "revoked":
        lines.append({**receipt, "verified": False})
    removal_lane["log_path"].write_text(
        "".join(json.dumps(r) + "\n" for r in lines) + ("uncertain line\n" if damage == "journal" else "")
    )
    backup.run_backup(**removal_lane, phase="upload", remove_local_after_upload=damage != "opt_out")
    assert path.exists()


def test_sweep_before_capacity_failure(removal_lane, monkeypatch):
    receipt = _old_receipt(removal_lane)
    log = removal_lane["log_path"]
    log.write_text(json.dumps(receipt) + "\n")

    def no_capacity(*args, **kwargs):
        assert not Path(receipt["snapshot"]).exists()
        raise RuntimeError("capacity")

    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", no_capacity)
    with pytest.raises(RuntimeError, match="capacity"):
        backup.run_backup(**removal_lane)


def test_end_sweep_is_persisted_in_final_receipt(removal_lane, monkeypatch):
    def upload(path, *args, **kwargs):
        old = _old_receipt(removal_lane)
        backup._append_json_log(removal_lane["log_path"], old)
        return {"id": "uploaded", "name": path.name, "size": str(path.stat().st_size), "md5Checksum": "local-md5"}

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    result = backup.run_backup(**removal_lane)
    assert result["local_gzip_retention_deleted"] == ["2026-10-21.db.zst"]
    assert not (removal_lane["staging_dir"] / "2026-10-21.db.zst").exists()
    final = json.loads(removal_lane["log_path"].read_text().splitlines()[-1])
    assert final["local_gzip_retention_deleted"] == result["local_gzip_retention_deleted"]


@pytest.mark.parametrize("snapshot_key", [None, "prior-machine/2026-10-21.db.zst"])
def test_drive_id_revocation_is_not_undone_by_either_sweep(removal_lane, snapshot_key):
    receipt = _old_receipt(removal_lane)
    revoke = {"drive_file": receipt["drive_file"], "uploaded": True, "verified": False, "verification_error": "md5"}
    if snapshot_key:
        revoke["snapshot"] = snapshot_key
    log = removal_lane["log_path"]
    log.write_text(json.dumps(receipt) + "\n" + json.dumps(revoke) + "\n")
    backup.run_backup(**removal_lane, phase="upload")
    assert Path(receipt["snapshot"]).exists()
    assert receipt["drive_file"]["id"] not in backup._verified_drive_file_ids(log)
    events = [json.loads(line) for line in log.read_text().splitlines()][2:]
    assert len(events) == 1  # Same reason is deduped across start/end sweeps.
    assert events[0]["event"] == "local_sweep"
    assert events[0]["local_removed_skip_reason"] == "not_verified"
    # Even a sweep carrying accidental verification fields cannot supersede remote proof.
    backup._append_json_log(log, {**receipt, "event": "local_sweep"})
    assert receipt["drive_file"]["id"] not in backup._verified_drive_file_ids(log)
    assert not (
        {"snapshot", "drive_file", "uploaded", "verified", "attempted_at", "backup_log_provenance"} & events[0].keys()
    )


def test_sweep_does_not_backdate_observability_or_brainbar_log(removal_lane):
    from brainlayer.observability_backup import _daily_snapshot

    old = _old_receipt(removal_lane)
    new = _old_receipt(removal_lane, "2026-10-26.db.zst")
    for receipt, day in [(old, "2026-10-21"), (new, "2026-10-26")]:
        receipt.update(
            backup_log_provenance="real",
            attempted_at=f"{day}T01:00:00+00:00",
            snapshot_created_at=f"{day}T01:00:00+00:00",
        )
    log = removal_lane["log_path"]
    log.write_text(json.dumps(old) + "\n" + json.dumps(new) + "\n")
    backup._sweep_verified_local_archives(
        log, removal_lane["staging_dir"], {"db": str(removal_lane["db_path"])}, remove_local=True
    )
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert not Path(old["snapshot"]).exists()
    assert _daily_snapshot(rows)[0]["last_at"] == "2026-10-26T01:00:00Z"
    # These are the fields consumed by BackupLogReader.lastRun(.databaseBackup).
    last_run = next(r for r in reversed(rows) if r.get("backup_log_provenance") == "real" and r.get("attempted_at"))
    assert last_run["attempted_at"] == new["attempted_at"]
    assert rows[-1]["event"] == "local_sweep"
    rows.append(
        {
            "attempted_at": "2026-10-27T01:00:00+00:00",
            "backup_log_provenance": "real",
            "error_type": "RuntimeError",
            "error": "capacity",
            "uploaded": False,
            "verified": False,
        }
    )
    assert _daily_snapshot(rows)[0]["last_at"] == "2026-10-26T01:00:00Z"


@pytest.mark.parametrize(
    "guard",
    [
        "verified",
        "uploaded",
        "drive_id",
        "drive_name",
        "drive_size",
        "same_db",
        "staging_dir",
        "regular_file",
        "standard_name",
        "before_today",
        "live_db",
        "archive_date",
    ],
)
def test_each_sweep_deletion_guard_retains_archive(removal_lane, monkeypatch, guard):
    receipt = _old_receipt(removal_lane)
    path = Path(receipt["snapshot"])
    context = {"db": str(removal_lane["db_path"])}
    if guard in {"verified", "uploaded"}:
        receipt[guard] = False
        # Isolate the receipt flag from the separate latest-per-object guard.
        monkeypatch.setattr(backup, "_verified_drive_file_ids", lambda _: {receipt["drive_file"]["id"]})
    elif guard in {"drive_name", "drive_size"}:
        receipt["drive_file"][guard.removeprefix("drive_")] = "wrong"
    elif guard == "same_db":
        receipt["db"] = "other-db"
    elif guard in {"staging_dir", "standard_name", "archive_date"}:
        destination = path.parent / (
            "elsewhere"
            if guard == "staging_dir"
            else "2026-10-21.db"
            if guard == "archive_date"
            else "2026-10-21-x.db.zst"
        )
        if guard == "staging_dir":
            destination.mkdir()
            destination /= path.name
        path.rename(destination)
        path = destination
        receipt["snapshot"] = str(path)
        receipt["drive_file"]["name"] = path.name
    elif guard == "regular_file":
        path.unlink()
        path.mkdir()
        receipt["drive_file"]["size"] = str(path.stat().st_size)
    elif guard == "before_today":
        today = path.with_name("2026-10-26.db.zst")
        path.rename(today)
        path = today
        receipt["snapshot"] = str(path)
        receipt["drive_file"]["name"] = path.name
    elif guard == "live_db":
        context["db"] = receipt["db"] = str(path)
    rows = [receipt]
    if guard == "drive_id":
        rows.append({"drive_file": receipt["drive_file"], "verified": False})
    log = removal_lane["log_path"]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert backup._sweep_verified_local_archives(log, removal_lane["staging_dir"], context, remove_local=True) == []
    assert path.exists()
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(r.get("local_removed") for r in rows)
    assert not any(r.get("local_removed_skip_reason") == "delete_failed" for r in rows)


@pytest.mark.parametrize("md5", ["match", "mismatch", "missing_local", "missing_drive"])
@pytest.mark.parametrize("sweep", [False, True])
def test_deletion_requires_recorded_drive_and_local_md5(removal_lane, monkeypatch, md5, sweep):
    local = {} if md5 == "missing_local" else {"local_md5": "local-md5"}
    drive = {} if md5 == "missing_drive" else {"md5Checksum": "wrong" if md5 == "mismatch" else "local-md5"}
    if sweep:
        receipt = _old_receipt(removal_lane)
        receipt.pop("local_md5")
        receipt.update(local)
        receipt["drive_file"].pop("md5Checksum")
        receipt["drive_file"].update(drive)
        removal_lane["log_path"].write_text(json.dumps(receipt) + "\n")
        backup.run_backup(**removal_lane, phase="upload")
        path = Path(receipt["snapshot"])
        rows = [json.loads(line) for line in removal_lane["log_path"].read_text().splitlines()]
        result = rows[-1]
    else:
        monkeypatch.setattr(backup, "verify_sqlite_backup_artifact", lambda *a, **kw: {"verified": True, **local})
        monkeypatch.setattr(
            backup,
            "upload_file_to_drive_raw",
            lambda path, *a, **kw: {"id": path.name, "name": path.name, "size": str(path.stat().st_size), **drive},
        )
        result = backup.run_backup(**removal_lane)
        path = removal_lane["staging_dir"] / "2026-10-26.db.zst"
    assert path.exists() is (md5 != "match")
    assert result["local_removed"] is (md5 == "match")
    assert result["local_removed_skip_reason"] == (None if md5 == "match" else "md5_unverified")


@pytest.mark.parametrize("sweep", [False, True])
def test_unlink_failure_does_not_abort_new_backup(removal_lane, monkeypatch, capsys, sweep):
    blocked = removal_lane["staging_dir"] / ("2026-10-21.db.zst" if sweep else "2026-10-26.db.zst")
    if sweep:
        receipt = _old_receipt(removal_lane)
        removal_lane["log_path"].write_text(json.dumps(receipt) + "\n")
    unlink = Path.unlink

    def refuse(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("test archive locked")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse)
    result = backup.run_backup(**removal_lane)
    assert result["uploaded"] and result["verified"]
    assert blocked.exists()
    rows = [json.loads(line) for line in removal_lane["log_path"].read_text().splitlines()]
    receipt = next(r for r in reversed(rows) if r.get("archive", r.get("snapshot")) == str(blocked))
    assert receipt["local_removed"] is False
    assert receipt["local_removed_skip_reason"] == "delete_failed"
    assert "test archive locked" in capsys.readouterr().out
