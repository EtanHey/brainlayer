"""Approved database retention counts and deletion boundaries (module's scoped owner)."""

import json
import os
from unittest.mock import Mock

import pytest

from brainlayer import backup_daily as b


def daily(day, *, machine="MacBook-Pro", folder="folder", suffix=".db.zst"):
    return {
        "id": f"{machine}-{day}{suffix}",
        "name": f"2026-11-{day:02}{suffix}",
        "size": "8",
        "md5Checksum": f"{day:032x}",
        "appProperties": {"brainlayer_machine": machine},
        "parents": [folder],
    }


def receipt(item):
    return {
        "db": "/fixture.db",
        "machine_id": item["appProperties"]["brainlayer_machine"],
        "snapshot": "/backups/" + item["name"],
        "uploaded": True,
        "verified": True,
        "local_md5": item["md5Checksum"],
        "drive_file": item.copy(),
    }


def prune(items, receipts, **kwargs):
    service = Mock()
    service.files.return_value.list.return_value.execute.return_value = {"files": items}
    deleted = b.prune_drive_backups(
        service,
        machine_id="MacBook-Pro",
        selected_folder_id="folder",
        verified_receipts=receipts,
        **kwargs,
    )
    service.files.return_value.delete.assert_not_called()
    for call in service.files.return_value.update.call_args_list:
        assert call.kwargs["body"] == {"trashed": True}
    return deleted, service


def test_drive_counts_only_verified_dailies_and_keeps_seven():
    items = [daily(day) for day in range(1, 11)]
    verified = [receipt(item) for item in items[:9]]
    deleted, _ = prune(items, verified)
    assert set(deleted) == {items[0]["name"], items[1]["name"]}
    assert items[-1]["name"] not in deleted  # unknown newest never displaces a good copy


def test_drive_protects_other_machine_folder_one_off_and_weekly_pin():
    items = [daily(day) for day in range(1, 11)]
    foreign = daily(1, machine="Locals-MacBook-Pro")
    elsewhere = daily(1, folder="other")
    elsewhere["id"] = "other-folder"
    named = {**daily(1), "id": "named", "name": "2026-11-01-pre-migration.db.zst"}
    items += [foreign, elsewhere, named]
    deleted, service = prune(items, [receipt(item) for item in items], protected_names={items[0]["name"]})
    assert set(deleted) == {items[1]["name"], items[2]["name"]}
    ids = {call.kwargs["fileId"] for call in service.files.return_value.update.call_args_list}
    assert not ids & {foreign["id"], elsewhere["id"], named["id"]}


@pytest.mark.parametrize("change", ["md5", "size", "name", "revoked"])
def test_drive_stale_or_revoked_receipt_never_authorizes_trash(change):
    items = [daily(day) for day in range(1, 10)]
    proofs = [receipt(item) for item in items]
    if change == "md5":
        items[0]["md5Checksum"] = "f" * 32
    elif change == "size":
        items[0]["size"] = "9"
    elif change == "name":
        items[0]["name"] = "2026-10-01.db.zst"
    else:
        proofs += [{**proofs[0], "verified": False}]
    deleted, _ = prune(items, proofs)
    assert items[0]["name"] not in deleted


def test_ids_without_md5_receipts_never_authorize_trash():
    service = Mock()
    items = [daily(day) for day in range(1, 10)]
    service.files.return_value.list.return_value.execute.return_value = {"files": items}
    assert (
        b.prune_drive_backups(
            service, machine_id="MacBook-Pro", selected_folder_id="folder", verified_file_ids={i["id"] for i in items}
        )
        == []
    )
    service.files.return_value.update.assert_not_called()


@pytest.mark.parametrize(
    "name",
    [
        "2026-11-01.db",
        "2026-11-01-verified.db.gz",
        "2026-11-01.db.zst.age",
        "2026-11-31.db.zst",
        "other-2026-11-01.db.gz",
    ],
)
def test_drive_only_exact_valid_daily_names_are_candidates(name):
    item = {**daily(1), "name": name}
    items = [item] + [daily(d) for d in range(2, 11)]
    deleted, _ = prune(items, [receipt(i) for i in items], retention_policy=b.DriveRetentionPolicy(1))
    assert name not in deleted


def test_drive_never_trashes_its_newest_verified_even_with_keep_one():
    items = [daily(day) for day in range(1, 11)]
    deleted, _ = prune(items, [receipt(i) for i in items[:5]], retention_policy=b.DriveRetentionPolicy(1))
    assert set(deleted) == {i["name"] for i in items[:4]}
    assert daily(5)["name"] not in deleted


def test_unreadable_pin_keeps_local_files(tmp_path):
    (tmp_path / ".weekly-backup-pin.json").write_text("broken")
    log = tmp_path / "receipt.jsonl"
    log.write_text(json.dumps({"uploaded": True, "verified": True}) + "\n")
    assert b._sweep_verified_local_archives(log, tmp_path, {"db": "/fixture.db"}, remove_local=True) == []


def test_local_unverified_day_never_authorizes_deleting_older_copy(tmp_path):
    names = [f"2026-11-{day:02}.db.zst" for day in range(1, 6)]
    for name in names:
        (tmp_path / name).write_bytes(b"fixture")
    b.prune_local_compressed_snapshots(tmp_path, keep_latest=3, verified_drive_names=set(names[1:]))
    assert (tmp_path / names[0]).exists()


def test_local_counts_verified_not_unknown_newest(tmp_path):
    names = [f"2026-11-{day:02}.db.zst" for day in range(1, 6)]
    for name in names:
        (tmp_path / name).write_bytes(b"fixture")
    deleted = b.prune_local_compressed_snapshots(tmp_path, keep_latest=3, verified_drive_names=set(names[:4]))
    assert deleted == [names[0]]
    assert all((tmp_path / name).exists() for name in names[1:])


@pytest.mark.parametrize("host,count", [("m1", 3), ("m4", 1)])
def test_local_config_uses_shared_host_detection(monkeypatch, host, count):
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST", host)
    assert b._local_retention_keep() == count


def test_local_count_can_be_configured(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST", "m4")
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOCAL_KEEP_M4", "2")
    assert b._local_retention_keep() == 2


def test_drive_retention_defaults_on_and_honors_opt_out(monkeypatch):
    monkeypatch.delenv("BRAINLAYER_BACKUP_DRIVE_RETENTION", raising=False)
    assert b._drive_retention_enabled()
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_RETENTION", "0")
    assert not b._drive_retention_enabled()


@pytest.mark.parametrize("host,keep", [("m1", 3), ("m4", 1)])
def test_real_local_sweep_keeps_host_count_and_unknown_newest(tmp_path, monkeypatch, host, keep):
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST", host)
    log = tmp_path / "receipt.jsonl"
    db = tmp_path / "fixture.db"
    rows = []
    for day in range(1, 8):
        item = daily(day)
        path = tmp_path / item["name"]
        path.write_bytes(b"fixture!")
        item["md5Checksum"] = b._md5_file(path)
        rows.append(dict(receipt(item), db=str(db), snapshot=str(path)))
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    unknown = tmp_path / "2026-11-08.db.zst"
    unknown.write_bytes(b"unknown")
    removed = b._sweep_verified_local_archives(log, tmp_path, {"db": str(db)}, remove_local=True)
    assert len(removed) == 7 - keep
    assert {p.name for p in tmp_path.glob("*.db.zst")} == {unknown.name} | {
        daily(d)["name"] for d in range(8 - keep, 8)
    }


def test_weekly_pin_remains_protected_after_its_date(tmp_path):
    (tmp_path / ".weekly-backup-pin.json").write_text(
        json.dumps({"date": "2026-10-29", "snapshot": "/backups/2026-10-29.db.zst"})
    )
    assert b._protected_snapshot_names(tmp_path) == {"2026-10-29.db.zst"}


def test_partial_drive_trash_failure_reports_completed_actions():
    items = [daily(day) for day in range(1, 11)]
    service = Mock()
    service.files.return_value.list.return_value.execute.return_value = {"files": items}
    service.files.return_value.update.return_value.execute.side_effect = [{}, OSError("offline")]
    with pytest.raises(b.BackupRetentionError) as caught:
        b.prune_drive_backups(
            service,
            machine_id="MacBook-Pro",
            selected_folder_id="folder",
            verified_receipts=[receipt(i) for i in items],
        )
    assert caught.value.trashed == [daily(3)["name"]]
    service.files.return_value.delete.assert_not_called()


def test_upload_of_older_day_reports_if_the_final_sweep_removed_it(tmp_path, monkeypatch):
    from tests.test_backup_daily import _stub_verified_backup_run

    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST", "m4")
    db, log = tmp_path / "fixture.db", tmp_path / "receipt.jsonl"
    snapshot = tmp_path / daily(1)["name"]
    snapshot.write_bytes(b"fixture!")
    rows = []
    for day in range(2, 5):
        item = daily(day)
        path = tmp_path / item["name"]
        path.write_bytes(b"fixture!")
        item["md5Checksum"] = b._md5_file(path)
        rows.append(dict(receipt(item), db=str(db), snapshot=str(path)))
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _stub_verified_backup_run(b, monkeypatch, snapshot, Mock())
    uploaded = daily(1)
    uploaded["md5Checksum"] = b._md5_file(snapshot)
    monkeypatch.setattr(b, "upload_file_to_drive_raw", lambda *a, **kw: uploaded)
    monkeypatch.setattr(
        b, "verify_sqlite_backup_artifact", lambda *a, **kw: {"verified": True, "local_md5": uploaded["md5Checksum"]}
    )
    monkeypatch.setattr(b, "prune_drive_backups", lambda *a, **kw: [])
    result = b.run_backup(
        db_path=db, staging_dir=tmp_path, log_path=log, date_stamp="2026-11-01", machine_id="MacBook-Pro"
    )
    assert result["verified"] and result["local_removed"]
    assert result["local_removed_skip_reason"] is None
    assert not snapshot.exists()


def test_one_off_report_does_not_delete_or_open_database(tmp_path):
    path = tmp_path / "brainlayer-pre-migration.db"
    path.write_bytes(b"never open this as sqlite")
    result = {"verified": True, "snapshot": str(tmp_path / "2026-11-22.db.zst")}
    os.utime(path, (0, 0))
    rows = b._one_off_copies(tmp_path, result)
    assert rows == [{"path": str(path), "bytes": path.stat().st_size, "superseded_by": result["snapshot"]}]
    assert path.exists()


def test_one_off_requires_a_newer_verified_backup(tmp_path):
    path = tmp_path / "brainlayer-pre-migration.db"
    path.write_bytes(b"retained")
    result = {"verified": False, "snapshot": str(tmp_path / "2026-11-22.db.zst")}
    assert b._one_off_copies(tmp_path, result) == []
    result["verified"] = True
    assert b._one_off_copies(tmp_path, result) == []  # copy is newer than snapshot's conservative midnight
    assert path.exists()


def test_drive_trash_failure_is_logged_and_alerted_without_failing_backup(tmp_path, monkeypatch):
    from brainlayer import job_alerts
    from tests.test_backup_daily import _stub_verified_backup_run

    snapshot = tmp_path / "2026-11-22.db.gz"
    snapshot.write_bytes(b"synthetic")
    service = Mock()
    _stub_verified_backup_run(b, monkeypatch, snapshot, service)
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_RETENTION", "1")
    monkeypatch.setattr(b, "prune_drive_backups", Mock(side_effect=OSError("trash unavailable")))
    alert = tmp_path / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alert))
    result = b.run_backup(db_path=tmp_path / "fixture.db", staging_dir=tmp_path, log_path=tmp_path / "log.jsonl")
    assert result["uploaded"] and result["verified"]
    assert "error" not in result
    assert result["retention_error_type"] == "OSError"
    assert "backup-retention" in job_alerts.active_alerts(alert)
    assert any(json.loads(line).get("retention_error") for line in (tmp_path / "log.jsonl").read_text().splitlines())
    assert snapshot.exists()
