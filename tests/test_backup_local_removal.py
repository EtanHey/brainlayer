"""Receipt-and-live-Drive-bound LOCAL cleanup; all data belongs to tmp_path."""

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup

REAL_LIST_DRIVE_BACKUP_FILES = backup._list_drive_backup_files


@pytest.mark.parametrize("platform,brand", [("linux", "Apple M4"), ("darwin", "Apple M3 Pro")])
def test_unknown_host_keeps_everything_and_alerts(removal_lane, monkeypatch, platform, brand):
    from brainlayer import backup_heavy_gate, job_alerts

    _write(removal_lane, [_old_receipt(removal_lane)])
    monkeypatch.delenv("BRAINLAYER_BACKUP_HEAVY_HOST")
    monkeypatch.setattr(backup_heavy_gate.sys, "platform", platform)
    monkeypatch.setattr(backup_heavy_gate, "_sysctl", lambda *_: brand)
    alerts = removal_lane["staging_dir"] / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alerts))
    result = backup.run_backup(**removal_lane)
    assert len(list(removal_lane["staging_dir"].glob("*.db.zst"))) == 3
    assert "Unknown backup retention host" in result["retention_error"]
    assert "Unknown backup retention host" in job_alerts.active_alerts(alerts)["backup-retention"]
    assert json.loads(removal_lane["log_path"].read_text().splitlines()[-1]) == result


def test_unreadable_pin_keeps_everything_and_alerts(removal_lane, monkeypatch):
    from brainlayer import job_alerts

    _write(removal_lane, [_old_receipt(removal_lane)])
    (removal_lane["staging_dir"] / ".weekly-backup-pin.json").write_text("broken")
    alerts = removal_lane["staging_dir"] / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alerts))
    result = backup.run_backup(**removal_lane)
    assert len(list(removal_lane["staging_dir"].glob("*.db.zst"))) == 3
    assert result["local_retention_inventory"] == "unmeasurable"
    assert "weekly pin" in job_alerts.active_alerts(alerts)["backup-retention"]
    assert json.loads(removal_lane["log_path"].read_text().splitlines()[-1]) == result


def test_run_backup_retention_service_has_no_write_authority(removal_lane, monkeypatch):
    from unittest.mock import Mock

    target = _old_receipt(removal_lane)
    _write(removal_lane, [target])
    service = Mock()
    resource = service.files.return_value
    resource.list.return_value.execute.side_effect = lambda: {"files": _live(removal_lane)}
    for verb in ("update", "delete", "patch", "emptyTrash", "create", "copy"):
        getattr(resource, verb).side_effect = AssertionError("Drive write attempted: " + verb)
    monkeypatch.setattr(backup, "build_drive_service", lambda *_: service)
    monkeypatch.setattr(backup, "_list_drive_backup_files", REAL_LIST_DRIVE_BACKUP_FILES)
    result = backup.run_backup(**removal_lane)
    assert result["uploaded"] and result["verified"]
    assert not Path(target["snapshot"]).exists()
    assert resource.list.called
    for verb in ("update", "delete", "patch", "emptyTrash", "create", "copy"):
        getattr(resource, verb).assert_not_called()


@pytest.fixture
def removal_lane(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST", "m4")
    monkeypatch.setattr(backup, "local_backup_date", lambda *_: dt.date(2026, 10, 26))
    monkeypatch.setattr(backup, "get_drive_credentials", lambda: object())
    monkeypatch.setattr(backup, "build_drive_service", lambda *_: object())
    monkeypatch.setattr(backup, "ensure_drive_folder_chain", lambda *_: "folder")
    monkeypatch.setattr(backup, "_find_drive_folder_ids", lambda *a, **kw: ["folder"])
    monkeypatch.setattr(backup, "assert_drive_folder_owned_by_machine", lambda *a, **kw: None)
    monkeypatch.setattr(backup, "verify_drive_upload", lambda *a, **kw: None)
    monkeypatch.setattr(
        backup,
        "upload_file_to_drive_raw",
        lambda path, *a, **kw: {
            "id": path.name,
            "name": path.name,
            "size": str(path.stat().st_size),
            "md5Checksum": backup._md5_file(path),
        },
    )
    snapshot = tmp_path / "2026-10-26.db.zst"
    monkeypatch.setattr(
        backup,
        "verify_sqlite_backup_artifact",
        lambda *a, **kw: {
            "verified": True,
            "local_md5": backup._md5_file(snapshot),
        },
    )

    def create(*args, **kwargs):
        snapshot.write_bytes(b"archive")
        return SimpleNamespace(
            gzip_path=snapshot, uncompressed_path=None, sentinel_chunks=1, local_retention_deleted=[]
        )

    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", create)
    lane = dict(
        db_path=tmp_path / "live.db",
        staging_dir=tmp_path,
        log_path=tmp_path / "receipts.jsonl",
        date_stamp="2026-10-26",
    )
    monkeypatch.setattr(backup, "_list_drive_backup_files", lambda *a, **kw: _live(lane))
    return lane


def _old_receipt(lane, name="2026-10-21.db.zst", **changes):
    path = lane["staging_dir"] / name
    path.write_bytes(b"older archive")
    md5 = backup._md5_file(path)
    return dict(
        db=str(lane["db_path"]),
        snapshot=str(path),
        local_md5=md5,
        uploaded=True,
        verified=True,
        machine_id="test-machine",
        drive_folder="/".join(backup.default_drive_folder_parts("test-machine")),
        backup_log_provenance=backup._backup_log_provenance(),
        drive_file={"id": name, "name": name, "size": str(path.stat().st_size), "md5Checksum": md5},
        **changes,
    )


def _write(lane, rows, *, keeper=True):
    # Explicit newer verified local copy makes the older target deletion-eligible.
    if keeper:
        rows = [*rows, _old_receipt(lane, "2026-10-24.db.zst")]
    lane["log_path"].write_text("".join(json.dumps(row) + "\n" for row in rows))


def _live(lane):
    if not lane["log_path"].exists():
        return []
    rows = backup._read_backup_receipts(lane["log_path"])
    return [
        dict(r["drive_file"], parents=["folder"], appProperties={"brainlayer_machine": "test-machine"})
        for r in rows
        if r.get("event") != "local_sweep" and isinstance(r.get("drive_file"), dict)
    ]


def _sweep(lane, *, context=None):
    return backup._sweep_verified_local_archives(
        lane["log_path"],
        lane["staging_dir"],
        context or {"db": str(lane["db_path"]), "machine_id": "test-machine"},
        remove_local=True,
        live_files=_live(lane),
        folder_id="folder",
    )


@pytest.mark.parametrize(
    "pin,verified,opt_out,reason",
    [
        ("today", True, False, "weekly_pin_pending"),
        ("past", True, False, "weekly_pin_pending"),
        ("unreadable", True, False, "pin_unreadable"),
        (None, False, False, "not_verified"),
        (None, True, True, "opt_out"),
        (None, True, False, "retention_keep"),
    ],
)
def test_run_records_newest_kept_or_skip(removal_lane, monkeypatch, pin, verified, opt_out, reason):
    snapshot = removal_lane["staging_dir"] / "2026-10-26.db.zst"
    if pin:
        (snapshot.parent / ".weekly-backup-pin.json").write_text(
            "broken"
            if pin == "unreadable"
            else json.dumps({"date": "2026-10-26" if pin == "today" else "2026-10-25", "snapshot": str(snapshot)})
        )
    monkeypatch.setattr(
        backup,
        "verify_sqlite_backup_artifact",
        lambda *a, **kw: {"verified": verified, "local_md5": backup._md5_file(snapshot)},
    )
    result = backup.run_backup(**removal_lane, remove_local_after_upload=not opt_out)
    assert not result["local_removed"] and snapshot.exists()
    assert result["local_removed_skip_reason"] == reason
    assert json.loads(removal_lane["log_path"].read_text().splitlines()[-1])["local_removed_skip_reason"] == reason


@pytest.mark.parametrize("pin", [None, "unreadable", "past"])
def test_upload_only_cleanup_obeys_persistent_pin_and_live_proof(removal_lane, pin):
    target = _old_receipt(removal_lane)
    unknown = _old_receipt(removal_lane, "2026-10-25.db.gz")
    unknown["verified"] = False
    if pin:
        (removal_lane["staging_dir"] / ".weekly-backup-pin.json").write_text(
            "broken" if pin == "unreadable" else json.dumps({"date": "2026-10-21", "snapshot": target["snapshot"]})
        )
    _write(removal_lane, [target, unknown])
    result = backup.run_backup(**removal_lane, phase="upload")
    assert result["upload_state"] == "nothing_pending"
    assert Path(target["snapshot"]).exists() is (pin is not None)
    assert Path(unknown["snapshot"]).exists()


def test_sweep_before_capacity_failure(removal_lane, monkeypatch):
    old = _old_receipt(removal_lane)
    _write(removal_lane, [old])

    def no_capacity(*args, **kwargs):
        assert not Path(old["snapshot"]).exists()
        raise RuntimeError("capacity")

    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", no_capacity)
    with pytest.raises(RuntimeError, match="capacity"):
        backup.run_backup(**removal_lane)


def test_end_sweep_is_persisted_in_final_receipt(removal_lane, monkeypatch):
    def upload(path, *args, **kwargs):
        old = _old_receipt(removal_lane)
        backup._append_json_log(removal_lane["log_path"], old)
        return {
            "id": path.name,
            "name": path.name,
            "size": str(path.stat().st_size),
            "md5Checksum": backup._md5_file(path),
        }

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    result = backup.run_backup(**removal_lane)
    assert result["local_gzip_retention_deleted"] == ["2026-10-21.db.zst"]
    assert not (removal_lane["staging_dir"] / "2026-10-21.db.zst").exists()
    assert json.loads(removal_lane["log_path"].read_text().splitlines()[-1]) == result


def test_post_upload_inventory_failure_is_persisted_without_masking_success(removal_lane, monkeypatch):
    from unittest.mock import Mock

    from brainlayer import job_alerts

    # One old verified local is within the cap before upload; only the final
    # fresh listing can fail after this run adds a second verified local copy.
    _write(removal_lane, [_old_receipt(removal_lane)], keeper=False)
    listing = Mock(side_effect=OSError("inventory unavailable after upload"))
    monkeypatch.setattr(backup, "_list_drive_backup_files", listing)
    alerts = removal_lane["staging_dir"] / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alerts))
    result = backup.run_backup(**removal_lane)
    assert result["uploaded"] and result["verified"] and "error" not in result
    assert result["retention_error_type"] == "OSError"
    assert json.loads(removal_lane["log_path"].read_text().splitlines()[-1]) == result
    assert "backup-retention" in job_alerts.active_alerts(alerts)
    listing.assert_called_once()


@pytest.mark.parametrize("snapshot_key", [None, "prior-machine/2026-10-21.db.zst"])
def test_live_listing_does_not_undo_latest_receipt_revocation(removal_lane, snapshot_key):
    old = _old_receipt(removal_lane)
    revoked = {"drive_file": old["drive_file"], "uploaded": True, "verified": False, "verification_error": "md5"}
    if snapshot_key:
        revoked["snapshot"] = snapshot_key
    _write(removal_lane, [old, revoked])
    _sweep(removal_lane)
    assert Path(old["snapshot"]).exists()
    assert old["drive_file"]["id"] not in backup._verified_drive_file_ids(removal_lane["log_path"])
    rows = backup._read_backup_receipts(removal_lane["log_path"])
    event = next(r for r in rows if r.get("archive") == old["snapshot"])
    assert event["local_removed_skip_reason"] == "not_verified"
    assert not (
        {"snapshot", "drive_file", "uploaded", "verified", "attempted_at", "backup_log_provenance"} & event.keys()
    )


def test_sweep_does_not_backdate_observability_or_brainbar_log(removal_lane):
    from brainlayer.observability_backup import _daily_snapshot

    old = _old_receipt(removal_lane)
    new = _old_receipt(removal_lane, "2026-10-26.db.zst")
    for row, day in [(old, "2026-10-21"), (new, "2026-10-26")]:
        row.update(
            backup_log_provenance="real",
            attempted_at=f"{day}T01:00:00+00:00",
            snapshot_created_at=f"{day}T01:00:00+00:00",
        )
    _write(removal_lane, [old, new], keeper=False)
    _sweep(removal_lane)
    rows = backup._read_backup_receipts(removal_lane["log_path"])
    assert not Path(old["snapshot"]).exists()
    assert _daily_snapshot(rows)[0]["last_at"] == "2026-10-26T01:00:00Z"
    last = next(r for r in reversed(rows) if r.get("backup_log_provenance") == "real" and r.get("attempted_at"))
    assert last["attempted_at"] == new["attempted_at"]


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
def test_each_sweep_deletion_boundary_retains_archive(removal_lane, guard):
    old = _old_receipt(removal_lane)
    path = Path(old["snapshot"])
    context = {"db": str(removal_lane["db_path"]), "machine_id": "test-machine"}
    if guard in {"verified", "uploaded"}:
        old[guard] = False
    elif guard in {"drive_name", "drive_size"}:
        old["drive_file"][guard.removeprefix("drive_")] = "wrong"
    elif guard == "same_db":
        old["db"] = "other-db"
    elif guard in {"staging_dir", "standard_name", "archive_date", "before_today"}:
        destination = path.parent / (
            "elsewhere"
            if guard == "staging_dir"
            else "2026-10-21-x.db.zst"
            if guard == "standard_name"
            else "2026-10-21.db"
            if guard == "archive_date"
            else "2026-10-26.db.zst"
        )
        if guard == "staging_dir":
            destination.mkdir()
            destination /= path.name
        path.rename(destination)
        path = destination
        old["snapshot"], old["drive_file"]["name"] = str(path), path.name
    elif guard == "regular_file":
        path.unlink()
        path.mkdir()
        old["drive_file"]["size"] = str(path.stat().st_size)
    elif guard == "live_db":
        old["db"] = context["db"] = str(path)
    rows = [old]
    if guard == "drive_id":
        rows.append({"drive_file": old["drive_file"], "verified": False})
    _write(removal_lane, rows)
    assert path.name not in _sweep(removal_lane, context=context)
    assert path.exists()


@pytest.mark.parametrize("md5", ["match", "mismatch", "missing_local", "missing_drive"])
def test_deletion_requires_recorded_and_live_matching_md5(removal_lane, md5):
    old = _old_receipt(removal_lane)
    if md5 == "mismatch":
        old["drive_file"]["md5Checksum"] = "f" * 32
    elif md5 == "missing_local":
        old.pop("local_md5")
    elif md5 == "missing_drive":
        old["drive_file"].pop("md5Checksum")
    _write(removal_lane, [old])
    _sweep(removal_lane)
    assert Path(old["snapshot"]).exists() is (md5 != "match")


@pytest.mark.parametrize("sweep", [False, True])
def test_unlink_failure_keeps_backup_success_and_local_bytes(removal_lane, monkeypatch, capsys, sweep):
    blocked = removal_lane["staging_dir"] / ("2026-10-21.db.zst" if sweep else "2026-10-26.db.zst")
    if sweep:
        _write(removal_lane, [_old_receipt(removal_lane)])
    unlink = Path.unlink

    def refuse(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("test archive locked")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse)
    result = backup.run_backup(**removal_lane)
    assert result["uploaded"] and result["verified"] and blocked.exists()
    rows = backup._read_backup_receipts(removal_lane["log_path"])
    row = next(r for r in reversed(rows) if r.get("archive", r.get("snapshot")) == str(blocked))
    assert row["local_removed"] is False
    assert row["local_removed_skip_reason"] == ("delete_failed" if sweep else "retention_keep")
    if sweep:
        assert "test archive locked" in capsys.readouterr().out


def test_sweep_retains_same_size_archive_rewritten_after_upload(removal_lane):
    old = _old_receipt(removal_lane)
    _write(removal_lane, [old])
    path = Path(old["snapshot"])
    path.write_bytes(b"OLDER ARCHIVE")
    assert path.stat().st_size == int(old["drive_file"]["size"])
    assert _sweep(removal_lane) == []
    assert path.read_bytes() == b"OLDER ARCHIVE"
    rows = backup._read_backup_receipts(removal_lane["log_path"])
    row = next(r for r in rows if r.get("archive") == str(path))
    assert row["local_removed_skip_reason"] == "md5_current_mismatch"
