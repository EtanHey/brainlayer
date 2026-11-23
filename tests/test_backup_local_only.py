"""Local retention is authorized by the same live, verified Drive snapshot only."""

import datetime as dt
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from brainlayer import backup_daily as b
from brainlayer.backup_retention import drive_retention_report
from brainlayer.backup_retention_invariant import inspect_backup_daily_retention_invariant
from tests.test_backup_retention import daily, receipt


@pytest.mark.parametrize(
    "statement",
    [
        "from .jsonl_backup import _prune_drive_backups_to_trash as prune; prune(service)",
        "f = service.files; f().update(fileId='x', body={'trashed': True})",
        "files.update(fileId='x', body={'trashed': True})",
    ],
)
@pytest.mark.parametrize("policy_module", [False, True])
def test_guard_rejects_cross_lane_and_indirect_writes(statement, policy_module):
    from brainlayer import backup_retention

    module = backup_retention if policy_module else b
    source = Path(module.__file__).read_text() + "\ndef forbidden(service, files):\n    " + statement + "\n"
    assert inspect_backup_daily_retention_invariant(source, policy_module=policy_module)


def test_pure_retention_module_has_no_drive_writes():
    from brainlayer import backup_retention

    assert (
        inspect_backup_daily_retention_invariant(Path(backup_retention.__file__).read_text(), policy_module=True) == []
    )


@pytest.mark.parametrize("brand,expected", [("Apple M1 Pro", "m1"), ("Apple M4 Max", "m4")])
def test_strict_host_accepts_recognised_families(monkeypatch, brand, expected):
    from brainlayer import backup_heavy_gate

    monkeypatch.delenv("BRAINLAYER_BACKUP_HEAVY_HOST", raising=False)
    monkeypatch.setattr(backup_heavy_gate.sys, "platform", "darwin")
    monkeypatch.setattr(backup_heavy_gate, "_sysctl", lambda *_: brand)
    assert backup_heavy_gate._host(strict=True) == expected


def test_second_host_probe_failure_keeps_archives_and_alerts(tmp_path, monkeypatch):
    import subprocess

    from brainlayer import backup_heavy_gate, job_alerts

    log, context, files = local_lane(tmp_path, monkeypatch)
    monkeypatch.delenv("BRAINLAYER_BACKUP_HEAVY_HOST")
    monkeypatch.setattr(backup_heavy_gate.sys, "platform", "darwin")
    monkeypatch.setattr(backup_heavy_gate, "_sysctl", Mock(side_effect=subprocess.TimeoutExpired("sysctl", 5)))
    alerts = tmp_path / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alerts))
    assert (
        b._sweep_verified_local_archives(
            log, tmp_path, context, remove_local=True, live_files=files, folder_id="folder"
        )
        == []
    )
    assert len(list(tmp_path.glob("*.db.zst"))) == 8
    assert "backup-retention" in job_alerts.active_alerts(alerts)


def local_lane(tmp_path, monkeypatch, host="m4"):
    machine = "MacBook-Pro" if host == "m4" else "Locals-MacBook-Pro"
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST", host)
    db, log = tmp_path / "fixture.db", tmp_path / "receipt.jsonl"
    rows, files = [], []
    for day in range(1, 9):
        item = daily(day, machine=machine)
        path = tmp_path / item["name"]
        path.write_bytes(b"fixture!")
        item["md5Checksum"] = b._md5_file(path)
        rows.append(dict(receipt(item), db=str(db), snapshot=str(path)))
        files.append(item)
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return log, {"db": str(db), "machine_id": machine}, files


def test_receipts_without_live_listing_never_delete_local_files(tmp_path, monkeypatch):
    log, context, _ = local_lane(tmp_path, monkeypatch)
    before = log.read_bytes()
    assert b._sweep_verified_local_archives(log, tmp_path, context, remove_local=True) == []
    assert len(list(tmp_path.glob("*.db.zst"))) == 8
    assert log.read_bytes() == before


def test_filename_only_legacy_api_never_removes_archives(tmp_path, monkeypatch):
    _, _, files = local_lane(tmp_path, monkeypatch)
    assert (
        b.prune_local_compressed_snapshots(tmp_path, keep_latest=1, verified_drive_names={i["name"] for i in files})
        == []
    )
    assert len(list(tmp_path.glob("*.db.zst"))) == 8


def test_newer_day_never_reclaims_an_unverified_older_stall(tmp_path, monkeypatch):
    _, _, files = local_lane(tmp_path, monkeypatch)
    path = tmp_path / files[0]["name"]
    old = {
        "snapshot": str(path),
        "local_verified": False,
        "verified": False,
        "resumable_upload": b._upload_identity(path),
    }
    newer = {"snapshot": str(tmp_path / "2026-11-09.db.zst"), "uploaded": True, "verified": True}
    b._remove_superseded_stalls([old], newer, "2026-11-09")
    assert path.exists()


@pytest.mark.parametrize("host,keep", [("m1", 3), ("m4", 1)])
def test_local_counts_require_current_listing_and_own_md5(tmp_path, monkeypatch, host, keep):
    log, context, files = local_lane(tmp_path, monkeypatch, host)
    removed = b._sweep_verified_local_archives(
        log, tmp_path, context, remove_local=True, live_files=files, folder_id="folder"
    )
    assert len(removed) == 8 - keep
    assert sorted(p.name for p in tmp_path.glob("*.db.zst")) == [daily(day)["name"] for day in range(9 - keep, 9)]


def test_local_count_spans_both_compression_formats(tmp_path, monkeypatch):
    log, context, files = local_lane(tmp_path, monkeypatch, "m1")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    for day, (item, row) in enumerate(zip(files, rows), 1):
        if day % 2 == 0:
            old = tmp_path / item["name"]
            name = item["name"].replace(".zst", ".gz")
            old.rename(tmp_path / name)
            item["name"] = row["drive_file"]["name"] = name
            row["snapshot"] = str(tmp_path / name)
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    b._sweep_verified_local_archives(log, tmp_path, context, remove_local=True, live_files=files, folder_id="folder")
    assert sorted(p.name for p in tmp_path.glob("*.db.*")) == [
        "2026-11-06.db.gz",
        "2026-11-07.db.zst",
        "2026-11-08.db.gz",
    ]


@pytest.mark.parametrize("damage", ["missing", "id", "md5Checksum", "size", "name", "machine", "folder", "trashed"])
def test_live_drive_copy_must_match_exact_snapshot(tmp_path, monkeypatch, damage):
    log, context, files = local_lane(tmp_path, monkeypatch)
    target = tmp_path / files[0]["name"]
    if damage == "missing":
        files.pop(0)
    elif damage == "id":
        files[0]["id"] = ["malformed"]
    elif damage == "machine":
        files[0]["appProperties"] = {"brainlayer_machine": "another-machine"}
    elif damage == "folder":
        files[0]["parents"] = ["other-folder"]
    elif damage == "trashed":
        files[0]["trashed"] = True
    else:
        files[0][damage] = "changed"
    b._sweep_verified_local_archives(log, tmp_path, context, remove_local=True, live_files=files, folder_id="folder")
    assert target.exists()


def test_pin_one_off_and_unknown_local_copy_survive(tmp_path, monkeypatch):
    log, context, files = local_lane(tmp_path, monkeypatch)
    pin = files[0]["name"]
    (tmp_path / ".weekly-backup-pin.json").write_text(
        json.dumps({"date": "2026-11-01", "snapshot": str(tmp_path / pin)})
    )
    unknown, one_off = tmp_path / "2026-11-09.db.zst", tmp_path / "brainlayer-pre-migration.db"
    unknown.write_bytes(b"unknown")
    one_off.write_bytes(b"personal copy")
    b._sweep_verified_local_archives(log, tmp_path, context, remove_local=True, live_files=files, folder_id="folder")
    assert (tmp_path / pin).exists() and unknown.exists() and one_off.exists()


def test_malformed_pin_refuses_all_local_cleanup_with_live_proof(tmp_path, monkeypatch):
    log, context, files = local_lane(tmp_path, monkeypatch)
    (tmp_path / ".weekly-backup-pin.json").write_text("broken")
    assert (
        b._sweep_verified_local_archives(
            log, tmp_path, context, remove_local=True, live_files=files, folder_id="folder"
        )
        == []
    )
    assert len(list(tmp_path.glob("*.db.zst"))) == 8


def test_local_inventory_stat_error_keeps_archives_and_logs_reason(tmp_path, monkeypatch, capsys):
    log, context, files = local_lane(tmp_path, monkeypatch)
    target = tmp_path / files[0]["name"]
    real_stat, calls = Path.stat, []

    def unstable(path, *args, **kwargs):
        if path == target:
            calls.append(path)
            if len(calls) == 2:
                raise OSError("injected inventory stat failure")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", unstable)
    assert (
        b._sweep_verified_local_archives(
            log, tmp_path, context, remove_local=True, live_files=files, folder_id="folder"
        )
        == []
    )
    assert target.exists() and "Local retention deferred" in capsys.readouterr().out


def test_archive_replaced_during_hash_is_never_unlinked(tmp_path, monkeypatch):
    import os

    log, context, files = local_lane(tmp_path, monkeypatch)
    target = tmp_path / files[0]["name"]
    replacement = tmp_path / "replacement.noindex"
    replacement.write_bytes(b"changed!")  # Same size; the inode binds the hash to the unlinked path.
    md5 = b._md5_file

    def swap_after_hash(path):
        digest = md5(path)
        if path == target:
            os.replace(replacement, target)
        return digest

    monkeypatch.setattr(b, "_md5_file", swap_after_hash)
    b._sweep_verified_local_archives(log, tmp_path, context, remove_local=True, live_files=files, folder_id="folder")
    assert not replacement.exists(), "the injected swap must run inside the hash-to-unlink window"
    assert target.exists() and target.read_bytes() == b"changed!"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(
        row.get("archive") == str(target) and row.get("local_removed_skip_reason") == "local_identity_changed"
        for row in rows
    )


def test_legacy_drive_entry_cannot_write_even_when_env_enabled(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_RETENTION", "1")
    service = Mock()
    files = [daily(day) for day in range(1, 11)]
    service.files.return_value.list.return_value.execute.return_value = {"files": files}
    for method in ("update", "delete", "patch"):
        getattr(service.files.return_value, method).side_effect = AssertionError("no Drive write authority")
    assert (
        b.prune_drive_backups(
            service,
            machine_id="MacBook-Pro",
            selected_folder_id="folder",
            verified_receipts=[receipt(i) for i in files],
        )
        == []
    )
    service.files.assert_not_called()
    assert b._drive_retention_enabled() is False


def test_database_drive_write_invariant_checks_the_real_source():
    source = Path(b.__file__).read_text()
    assert inspect_backup_daily_retention_invariant(source) == []


@pytest.mark.parametrize(
    "statement",
    [
        "service.files().update(fileId='x', body={'trashed': True})",
        "files = service.files(); files.update(fileId='x', body={'name': 'renamed'})",
        "files = service.files(); write = files.update; write(fileId='x', body={'parents': []})",
        "getattr(service.files(), 'update')(fileId='x', body={'trashed': True})",
        "service.files().delete(fileId='x')",
        "service.files().patch(fileId='x')",
        "http.request('DELETE', 'https://www.googleapis.com/drive/v3/files/x')",
    ],
)
def test_invariant_rejects_drive_write_mutants(statement):
    source = Path(b.__file__).read_text() + "\ndef forbidden(service, http):\n    " + statement + "\n"
    assert inspect_backup_daily_retention_invariant(source)


def test_report_is_pure_and_compares_both_policies():
    files = [daily(day) for day in range(1, 11)]
    for name, size in [("2026-10-31.db.zst", 10), ("2026-09-30.db.gz", 20), ("2026-08-10.db", 30)]:
        files.append({**daily(1), "id": name, "name": name, "size": str(size)})
    before = json.dumps(files, sort_keys=True)
    report = drive_retention_report(files, machine_id="MacBook-Pro", folder_id="folder", as_of=dt.date(2026, 11, 22))
    assert report["mode"] == "report-only" and report["daily"]["count"] == 13
    assert report["keep_7"]["would_remove"]["count"] == 0  # No verified receipts authorize candidates.
    assert report["keep_7"]["kept"]["count"] == 13
    assert report["keep_7_plus_monthly"]["extra"]["count"] == 3
    assert report["keep_7_plus_monthly"]["extra"]["bytes"] == 60
    assert json.dumps(files, sort_keys=True) == before
