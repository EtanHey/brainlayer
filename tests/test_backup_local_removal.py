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
        lambda path, *a, **kw: {"id": path.name, "name": path.name, "size": str(path.stat().st_size)},
    )
    monkeypatch.setattr(
        backup, "verify_sqlite_backup_artifact", lambda *a, **kw: {"verified": True, "verification_mode": "quick"}
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
    monkeypatch.setattr(backup, "verify_sqlite_backup_artifact", lambda *a, **kw: {"verified": verified})
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
        uploaded=True,
        verified=True,
        machine_id="test-machine",
        drive_folder="/".join(backup.default_drive_folder_parts("test-machine")),
        backup_log_provenance=backup._backup_log_provenance(),
        drive_file={"id": name, "name": name, "size": str(path.stat().st_size)},
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
    latest = {r["snapshot"]: r for r in receipts if "snapshot" in r}
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
        return {"id": "uploaded", "name": path.name, "size": str(path.stat().st_size)}

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    result = backup.run_backup(**removal_lane)
    assert result["local_gzip_retention_deleted"] == ["2026-10-21.db.zst"]
    assert not (removal_lane["staging_dir"] / "2026-10-21.db.zst").exists()
    final = json.loads(removal_lane["log_path"].read_text().splitlines()[-1])
    assert final["local_gzip_retention_deleted"] == result["local_gzip_retention_deleted"]
