"""Caller-level regressions adapted from ID-R1-SOL1/SOL2 synthetic reproductions."""

from unittest.mock import Mock

import pytest

from brainlayer import backup_daily as b
from brainlayer import observability_backup
from tests.test_backup_daily import _stub_verified_backup_run
from tests.test_backup_machine_identity import _drive_fixture


def duplicate_tree_service():
    """A normal tree and a separate archived tree, both with legacy M4 data."""
    service = Mock()
    folders = [
        ("primary-root", "Brain Drive", None),
        ("archived-root", "Brain Drive", "unrelated-archive"),
        ("primary-archive", "06_ARCHIVE", "primary-root"),
        ("archived-archive", "06_ARCHIVE", "archived-root"),
        ("primary-backups", "backups", "primary-archive"),
        ("archived-backups", "backups", "archived-archive"),
        ("primary-db", "brainlayer-db", "primary-backups"),
        ("archived-db", "brainlayer-db", "archived-backups"),
    ]
    snapshots = {
        "primary-db": [{"id": f"current-{day}", "name": f"2026-10-{day:02}.db.gz"} for day in range(12, 19)],
        "archived-db": [{"id": "archived-copy", "name": "2026-10-01.db.gz"}],
    }
    import re

    def listing(**kwargs):
        q = kwargs["q"]
        name = re.search(r"name = '([^']+)'", q)
        parent = re.search(r"'([^']+)' in parents", q)
        if name:
            rows = [
                {"id": fid} for fid, fname, pid in folders if fname == name[1] and (parent is None or pid == parent[1])
            ]
        else:
            rows = snapshots.get(parent[1], [])
        return Mock(execute=Mock(return_value={"files": rows}))

    service.files.return_value.list.side_effect = listing
    return service


@pytest.mark.parametrize("tagged", [False, True])
def test_retention_keeps_separate_archive_tree_after_verified_upload(tmp_path, monkeypatch, tagged):
    service = duplicate_tree_service()
    if tagged:
        listing = service.files.return_value.list.side_effect

        def tagged_listing(**kwargs):
            response = listing(**kwargs)
            for item in response.execute.return_value["files"]:
                if "name" in item:
                    item["appProperties"] = {"brainlayer_machine": "MacBook-Pro"}
            return response

        service.files.return_value.list.side_effect = tagged_listing
    snapshot = tmp_path / "2026-10-18.db.gz"
    snapshot.write_bytes(b"synthetic")
    ensure, check_owner = b.ensure_drive_folder_chain, b.assert_drive_folder_owned_by_machine
    _stub_verified_backup_run(b, monkeypatch, snapshot, service)
    monkeypatch.setattr(b, "ensure_drive_folder_chain", ensure)
    monkeypatch.setattr(b, "assert_drive_folder_owned_by_machine", check_owner)
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_RETENTION", "1")
    result = b.run_backup(
        db_path=tmp_path / "fixture.db",
        staging_dir=tmp_path,
        log_path=tmp_path / "receipt.jsonl",
        machine_id="MacBook-Pro",
    )
    assert result["uploaded"] and result["verified"]
    assert result["retention_deleted"] == []
    service.files.return_value.update.assert_not_called()
    service.files.return_value.create.assert_not_called()
    service.files.return_value.delete.assert_not_called()
    assert not any("archived-db" in c.kwargs["q"] for c in service.files.return_value.list.call_args_list)


def test_prior_conflict_preserves_confirmed_upload_and_recovery_receipt(tmp_path, monkeypatch):
    snapshot = tmp_path / "2026-10-18.db.gz"
    snapshot.write_bytes(b"synthetic")
    service = _drive_fixture(foreign_owner=True)
    ensure, check_owner = b.ensure_drive_folder_chain, b.assert_drive_folder_owned_by_machine
    _stub_verified_backup_run(b, monkeypatch, snapshot, service)
    monkeypatch.setattr(b, "ensure_drive_folder_chain", ensure)
    monkeypatch.setattr(b, "assert_drive_folder_owned_by_machine", check_owner)
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_RETENTION", "1")
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PROVENANCE", "real")
    md5 = b._md5_file(snapshot)
    monkeypatch.setattr(b, "verify_sqlite_backup_artifact", lambda *a, **kw: {"verified": True, "local_md5": md5})
    monkeypatch.setattr(
        b,
        "upload_file_to_drive_raw",
        lambda *a, **kw: {
            "id": "drive-file-id",
            "name": snapshot.name,
            "size": str(snapshot.stat().st_size),
            "md5Checksum": md5,
        },
    )
    receipt_path = tmp_path / "receipt.jsonl"
    with pytest.raises(b.DriveFolderOwnedByOtherMachineError):
        b.run_backup(
            db_path=tmp_path / "fixture.db",
            staging_dir=tmp_path,
            log_path=receipt_path,
            machine_id="Locals-MacBook-Pro",
        )
    receipts = [r for r in b._read_backup_receipts(receipt_path) if r.get("event") != "local_sweep"]
    assert all("snapshot" in r and "attempted_at" in r for r in receipts)
    assert len(receipts) == 4
    assert receipts[2]["uploaded"] is True and receipts[2]["verified"] is True
    assert receipts[0]["local_verified"] is True and receipts[0]["uploaded"] is False
    assert receipts[1]["uploaded"] is True and receipts[1]["verification_pending"] is True
    assert "resumable_upload" not in receipts[1]
    receipt = receipts[-1]
    assert receipt["uploaded"] and receipt["verified"]
    assert receipt["drive_file"]["id"] == "drive-file-id"
    assert receipt["local_removed"] and not snapshot.exists()
    assert receipt["retention_error_type"] == "DriveFolderOwnedByOtherMachineError"
    assert receipt["retention_error"] == receipt["error"]
    assert receipt["error_code"] == "drive_folder_owned_by_other_machine"
    assert observability_backup._daily_snapshot([receipt])[0]["verified"] is True
    service.files.return_value.update.assert_not_called()
    service.files.return_value.create.assert_not_called()


def test_foreign_owner_on_late_prior_page_prevents_all_trash():
    service = _drive_fixture()
    listing = service.files.return_value.list.side_effect

    def paged(**kwargs):
        if kwargs["q"] == "'prior' in parents and trashed = false":
            if kwargs["pageToken"] is None:
                return Mock(execute=Mock(return_value={"files": [], "nextPageToken": "late"}))
            assert kwargs["pageToken"] == "late"
            return Mock(
                execute=Mock(
                    return_value={
                        "files": [
                            {
                                "id": "foreign",
                                "name": "2026-01-01.db.gz",
                                "appProperties": {"brainlayer_machine": "foreign-machine"},
                            }
                        ]
                    }
                )
            )
        return listing(**kwargs)

    service.files.return_value.list.side_effect = paged
    with pytest.raises(b.DriveFolderOwnedByOtherMachineError):
        b.prune_drive_backups(
            service, machine_id="Locals-MacBook-Pro", retention_policy=b.DriveRetentionPolicy(keep_latest=1)
        )
    service.files.return_value.update.assert_not_called()
    service.files.return_value.create.assert_not_called()


def test_explicit_folder_does_not_expand_to_default_or_prior(monkeypatch):
    service = Mock()
    service.files.return_value.list.return_value.execute.return_value = {
        "files": [
            {"id": "old", "name": "2026-10-16.db.gz"},
            {"id": "new", "name": "2026-10-17.db.gz"},
        ]
    }
    ensure = Mock(return_value="custom-id")
    discovery = Mock(side_effect=AssertionError("must not discover other folders"))
    monkeypatch.setattr(b, "ensure_drive_folder_chain", ensure)
    monkeypatch.setattr(b, "list_machine_drive_backups", discovery)
    assert b.prune_drive_backups(
        service,
        folder_parts=["custom", "only"],
        machine_id="Locals-MacBook-Pro",
        retention_policy=b.DriveRetentionPolicy(1),
        verified_file_ids={"new"},
    ) == ["2026-10-16.db.gz"]
    ensure.assert_called_once_with(service, ["custom", "only"])
    assert service.files.return_value.list.call_args.kwargs["q"] == "'custom-id' in parents and trashed = false"
    service.files.return_value.update.assert_called_once_with(
        fileId="old", body={"trashed": True}, supportsAllDrives=True
    )
    discovery.assert_not_called()


def test_disabled_retention_does_not_read_foreign_prior(tmp_path, monkeypatch):
    snapshot = tmp_path / "2026-10-18.db.gz"
    snapshot.write_bytes(b"synthetic")
    service = _drive_fixture(foreign_owner=True)
    _stub_verified_backup_run(b, monkeypatch, snapshot, service)
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_RETENTION", "0")
    result = b.run_backup(
        db_path=tmp_path / "fixture.db",
        staging_dir=tmp_path,
        log_path=tmp_path / "receipt.jsonl",
        machine_id="Locals-MacBook-Pro",
    )
    assert result["uploaded"] and result["verified"]
    service.files.return_value.list.assert_not_called()
    service.files.return_value.update.assert_not_called()


@pytest.mark.parametrize("parents", [[], ["a", "b"], "backups"])
def test_ambiguous_prior_parent_refuses_all_mutation(parents):
    service = _drive_fixture()
    service.files.return_value.get.return_value.execute.return_value = {"parents": parents}
    with pytest.raises(ValueError, match="unambiguous"):
        b.prune_drive_backups(service, machine_id="Locals-MacBook-Pro")
    service.files.return_value.update.assert_not_called()
    service.files.return_value.create.assert_not_called()


@pytest.mark.parametrize("missing_prior", [False, True])
def test_retention_scopes_paginated_prior_siblings_and_deduplicates(missing_prior):
    service = _drive_fixture(missing_prior=missing_prior)
    listing = service.files.return_value.list.side_effect

    def scoped(**kwargs):
        if "name = 'brainlayer-db-Locals-MBP.lan'" in kwargs["q"]:
            assert "'backups' in parents" in kwargs["q"]
            rows = [] if missing_prior else [{"id": "prior"}, {"id": "prior"}]
            token = kwargs.get("pageToken")
            return Mock(execute=Mock(return_value={"files": rows, **({"nextPageToken": "next"} if not token else {})}))
        return listing(**kwargs)

    service.files.return_value.list.side_effect = scoped
    removed = b.prune_drive_backups(
        service,
        machine_id="Locals-MacBook-Pro",
        retention_policy=b.DriveRetentionPolicy(keep_latest=1),
        verified_file_ids={"new"},
    )
    assert removed == ([] if missing_prior else ["2026-10-17.db.gz"])
    assert service.files.return_value.update.call_count == int(not missing_prior)
    service.files.return_value.create.assert_not_called()
    service.files.return_value.delete.assert_not_called()
    assert sum(
        c.kwargs["q"] == "'prior' in parents and trashed = false"
        for c in service.files.return_value.list.call_args_list
    ) == int(not missing_prior)
