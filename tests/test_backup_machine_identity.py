"""Synthetic identity/Drive fixtures: never contact Drive or the installed writer."""

import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from brainlayer import backup_daily, observability_backup


def test_scheduled_path_keeps_identity_when_network_hostname_changes(monkeypatch):
    monkeypatch.setattr(backup_daily.sys, "platform", "darwin")
    monkeypatch.setenv("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")

    def scutil(argv, **kwargs):
        if argv[0] != "/usr/sbin/scutil":
            raise FileNotFoundError(argv[0])
        return SimpleNamespace(returncode=0, stdout="Locals-MacBook-Pro\n")

    monkeypatch.setattr(backup_daily.subprocess, "run", scutil)
    resolved = []
    for hostname in ("Locals-MBP.lan", "hotel-router-name.lan"):
        monkeypatch.setattr(backup_daily.socket, "gethostname", lambda: hostname)
        resolved.append(backup_daily.resolve_machine_id(env={}))
    assert resolved == ["Locals-MacBook-Pro", "Locals-MacBook-Pro"]


@pytest.mark.parametrize("failure", [FileNotFoundError(), backup_daily.subprocess.TimeoutExpired("scutil", 5), None])
def test_macos_does_not_silently_use_network_name_when_stable_identity_fails(monkeypatch, failure):
    monkeypatch.setattr(backup_daily.sys, "platform", "darwin")

    def scutil(*args, **kwargs):
        if failure is not None:
            raise failure
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(backup_daily.subprocess, "run", scutil)
    hostname = Mock(return_value="untrusted-router.lan")
    monkeypatch.setattr(backup_daily.socket, "gethostname", hostname)
    with pytest.raises(backup_daily.InvalidMachineIdError, match="stable"):
        backup_daily.resolve_machine_id(env={})
    hostname.assert_not_called()


def test_explicit_identity_remains_authoritative_and_linux_fallback_works(monkeypatch):
    monkeypatch.setattr(backup_daily.sys, "platform", "linux")
    monkeypatch.setattr(backup_daily.subprocess, "run", Mock(side_effect=FileNotFoundError()))
    monkeypatch.setattr(backup_daily.socket, "gethostname", lambda: "ci-host.local")
    assert backup_daily.resolve_machine_id(env={"BRAINLAYER_MACHINE_ID": "explicit"}) == "explicit"
    assert backup_daily.resolve_machine_id(env={}) == "ci-host"


def _drive_fixture(*, foreign_owner=False, missing_prior=False):
    service = Mock()
    files = service.files.return_value
    folders = [
        ("brain", "Brain Drive", None),
        ("archive", "06_ARCHIVE", "brain"),
        ("backups", "backups", "archive"),
        ("stable", "brainlayer-db-Locals-MacBook-Pro", "backups"),
    ]
    if not missing_prior:
        folders.append(("prior", "brainlayer-db-Locals-MBP.lan", "backups"))
    snapshots = {
        "stable": [
            {"id": "new", "name": "2026-10-18.db.gz", "appProperties": {"brainlayer_machine": "Locals-MacBook-Pro"}}
        ],
        "prior": [
            {
                "id": "old",
                "name": "2026-10-17.db.gz",
                "appProperties": {"brainlayer_machine": "other-host" if foreign_owner else "Locals-MBP.lan"},
            }
        ],
    }

    def listing(**kwargs):
        query = kwargs["q"]
        assert "trashed = false" in query
        name = re.search(r"name = '([^']+)'", query)
        parent = re.search(r"'([^']+)' in parents", query)
        parent_id = parent[1] if parent else None
        if name:
            result = [
                {"id": fid, "name": fname} for fid, fname, pid in folders if fname == name[1] and pid == parent_id
            ]
        else:
            result = snapshots.get(parent_id, [])
        return Mock(execute=Mock(return_value={"files": result}))

    files.get.return_value.execute.return_value = {"parents": ["backups"]}
    files.list.side_effect = listing
    return service


def test_discovery_reads_stable_and_known_prior_folders_without_mutating_them():
    service = _drive_fixture()
    inventory = backup_daily.list_machine_drive_backups(service, machine_id="Locals-MacBook-Pro")
    assert {item["id"] for item in inventory} == {"new", "old"}
    files = service.files.return_value
    files.create.assert_not_called()
    files.update.assert_not_called()
    files.delete.assert_not_called()


def test_missing_prior_folder_is_not_created():
    service = _drive_fixture(missing_prior=True)
    inventory = backup_daily.list_machine_drive_backups(service, machine_id="Locals-MacBook-Pro")
    assert [item["id"] for item in inventory] == ["new"]
    service.files.return_value.create.assert_not_called()


def test_discovery_rejects_foreign_machine_in_prior_folder():
    service = _drive_fixture(foreign_owner=True)
    with pytest.raises(backup_daily.DriveFolderOwnedByOtherMachineError):
        backup_daily.prune_drive_backups(service, machine_id="Locals-MacBook-Pro")
    service.files.return_value.update.assert_not_called()


def test_opt_in_retention_sees_prior_ids_without_creating_or_moving_folders():
    service = _drive_fixture()
    removed = backup_daily.prune_drive_backups(
        service,
        machine_id="Locals-MacBook-Pro",
        retention_policy=backup_daily.DriveRetentionPolicy(keep_latest=1),
        verified_file_ids={"new"},
    )
    assert removed == ["2026-10-17.db.gz"]
    service.files.return_value.update.assert_called_once_with(
        fileId="old", body={"trashed": True}, supportsAllDrives=True
    )
    service.files.return_value.create.assert_not_called()
    service.files.return_value.delete.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_verified_backup_routes_opt_in_retention_to_prior_folder_inventory(tmp_path, monkeypatch, enabled):
    from tests.test_backup_daily import _stub_verified_backup_run

    snapshot = tmp_path / "2026-10-18.db.gz"
    snapshot.write_bytes(b"synthetic-backup")
    service = _drive_fixture()
    ensure = backup_daily.ensure_drive_folder_chain
    check_owner = backup_daily.assert_drive_folder_owned_by_machine
    _stub_verified_backup_run(backup_daily, monkeypatch, snapshot, service)
    monkeypatch.setattr(backup_daily, "ensure_drive_folder_chain", ensure)
    monkeypatch.setattr(backup_daily, "assert_drive_folder_owned_by_machine", check_owner)
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "Locals-MacBook-Pro")
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_RETENTION", "1" if enabled else "0")
    # Receipt evidence that the listed 10-18 object verified; retention trashes only behind it.
    (tmp_path / "backup.log").write_text(
        json.dumps({"drive_file": {"id": "new"}, "uploaded": True, "verified": True}) + "\n"
    )
    result = backup_daily.run_backup(
        db_path=tmp_path / "synthetic.db",
        staging_dir=tmp_path,
        log_path=tmp_path / "backup.log",
        remove_local_after_upload=False,
        retention_policy=backup_daily.DriveRetentionPolicy(keep_latest=1),
    )
    assert result["uploaded"] and result["verified"]
    assert result["retention_deleted"] == (["2026-10-17.db.gz"] if enabled else [])
    assert result["drive_folder"].endswith("brainlayer-db-Locals-MacBook-Pro")
    assert service.files.return_value.update.call_count == int(enabled)


def test_discovery_paginates_folder_and_snapshot_results_and_deduplicates():
    service = Mock()
    service.files.return_value.list.side_effect = [
        Mock(execute=Mock(return_value=page))
        for page in [
            {"files": [{"id": "folder"}], "nextPageToken": "folder-page-2"},
            {"files": [{"id": "folder"}]},
            {"files": [{"id": "first", "name": "2026-10-17.db.gz"}], "nextPageToken": "file-page-2"},
            {"files": [{"id": "second", "name": "2026-10-18.db.gz"}]},
        ]
    ]
    # The canonical Mac's historical folder and untagged snapshots remain valid.
    monkey_parts = Mock(return_value=["brainlayer-db"])
    from unittest.mock import patch

    with patch.object(backup_daily, "default_drive_folder_parts", monkey_parts):
        inventory = backup_daily.list_machine_drive_backups(service, machine_id="MacBook-Pro")
    assert {item["id"] for item in inventory} == {"first", "second"}
    assert [call.kwargs["pageToken"] for call in service.files.return_value.list.call_args_list] == [
        None,
        "folder-page-2",
        None,
        "file-page-2",
    ]
    service.files.return_value.create.assert_not_called()


def test_recovery_receipts_remain_visible_across_prior_identity():
    record = {
        "backup_log_provenance": "real",
        "machine_id": "Locals-MBP.lan",
        "drive_folder": "Brain Drive/06_ARCHIVE/backups/brainlayer-db-Locals-MBP.lan",
        "attempted_at": "2026-10-17T20:21:05Z",
        "uploaded": True,
        "verified": True,
        "drive_file": {"id": "old", "name": "2026-10-17.db.gz"},
    }
    snapshot, error, all_errors = observability_backup._daily_snapshot([record])
    assert snapshot["destination"] == "2026-10-17.db.gz"
    assert snapshot["verified"] is True
    assert error is None and not all_errors


@pytest.mark.parametrize("drive_file", [{"id": "prior-file", "name": "2026-10-17.db.gz"}, "2026-10-17.db.gz"])
def test_prior_identity_drive_only_receipt_uses_artifact_date_not_retry_time(drive_file):
    record = {
        "backup_log_provenance": "real",
        "machine_id": "Locals-MBP.lan",
        "drive_folder": "Brain Drive/06_ARCHIVE/backups/brainlayer-db-Locals-MBP.lan",
        "attempted_at": "2026-10-18T23:59:59Z",
        "uploaded": True,
        "verified": True,
        "drive_file": drive_file,
    }
    snapshot, error, all_errors = observability_backup._daily_snapshot([record])
    assert snapshot == {"last_at": "2026-10-17T00:00:00Z", "destination": "2026-10-17.db.gz", "verified": True}
    assert error is None and not all_errors


@pytest.mark.parametrize("created", [None, "invalid", "2026-10-17T20:00:00"])
def test_explicit_unknown_creation_time_never_falls_back_to_prior_drive_filename(created):
    assert (
        backup_daily._snapshot_created_time(
            {
                "snapshot_created_at": created,
                "drive_file": {"id": "prior-file", "name": "2026-10-17.db.gz"},
                "attempted_at": "2026-10-18T23:59:59Z",
            }
        )
        is None
    )
