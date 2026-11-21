"""Gap 22/23 recovery receipts retain measured progress and verification phase."""

# Imported pytest fixtures are intentionally shadowed by test arguments.
# ruff: noqa: F811

import datetime as dt
import json
from pathlib import Path

import pytest

from brainlayer import backup_daily as backup
from brainlayer.observability_backup import _local_snapshot
from tests.test_weekly_local_gate import local_lane  # noqa: F401

URL = "https://www.googleapis.com/upload/drive/v3/files?upload_id=isolated"


def latest(args):
    return json.loads(args["log_path"].read_text().splitlines()[-1])


def saved(args, *, uploaded=False, dated=True):
    receipt = backup.run_backup(**args, phase="snapshot")
    identity = backup._upload_identity(Path(receipt["snapshot"]))
    receipt.update(bytes_confirmed=25, resumable_upload={"url": URL, "folder_id": "folder", **identity})
    if dated:
        receipt["bytes_confirmed_at"] = (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=30)).isoformat()
    if uploaded:
        receipt.update(uploaded=True, verified=False, drive_file={"id": "completed"})
        receipt.pop("resumable_upload")
    backup._append_json_log(args["log_path"], receipt)
    return receipt


def network(monkeypatch):
    monkeypatch.setattr(backup, "get_drive_credentials", lambda: object())
    monkeypatch.setattr(backup, "build_drive_service", lambda: object())
    monkeypatch.setattr(backup, "ensure_drive_folder_chain", lambda *a: "folder")
    monkeypatch.setattr(backup, "assert_drive_folder_owned_by_machine", lambda *a, **kw: None)


@pytest.mark.parametrize(
    "confirmed,dated,new_session",
    [(25, True, False), (30, True, False), (25, False, False), (0, True, True), (0, True, False)],
)
def test_resume_checkpoint_dates_only_new_remote_bytes(local_lane, monkeypatch, confirmed, dated, new_session):
    prior = saved(local_lane, dated=dated)
    network(monkeypatch)
    observed = []
    url = URL + "-new" if new_session else URL

    def upload(path, *a, **kw):
        assert kw["resume_session_url"] == URL
        for count in [0, confirmed]:
            kw["progress_callback"](url, count)
            observed.append(latest(local_lane))
        raise backup.DriveUploadStalledError(
            bytes_confirmed=confirmed, total_bytes=path.stat().st_size, attempts=1, session_url=url
        )

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    with pytest.raises(backup.DriveUploadStalledError):
        backup.run_backup(**local_lane, phase="upload")
    if not new_session:
        assert observed[0]["bytes_confirmed"] == 25
        assert observed[0].get("bytes_confirmed_at") == prior.get("bytes_confirmed_at")
    if not new_session and confirmed <= 25:
        assert observed[-1].get("bytes_confirmed_at") == prior.get("bytes_confirmed_at")
    else:
        assert observed[-1]["bytes_confirmed_at"] != prior.get("bytes_confirmed_at")
    expected = confirmed if new_session else max(25, confirmed)
    assert observed[-1]["bytes_confirmed"] == expected
    assert latest(local_lane)["bytes_confirmed"] == expected


@pytest.mark.parametrize("uploaded", [True, False])
def test_first_recovery_receipt_names_verification_before_network(local_lane, monkeypatch, uploaded):
    saved(local_lane, uploaded=uploaded)
    network(monkeypatch)
    observed = []

    def service():
        receipt = latest(local_lane)
        observed.append(receipt)
        assert receipt.get("verification_pending", False) is uploaded
        assert receipt["upload_state"] == "pending"
        assert receipt["uploaded"] is uploaded
        assert _local_snapshot([receipt]).get("verification_pending", False) is uploaded
        raise RuntimeError("synthetic network stop")

    monkeypatch.setattr(backup, "build_drive_service", service)
    with pytest.raises(RuntimeError, match="synthetic network stop"):
        backup.run_backup(**local_lane, phase="upload")
    assert len(observed) == 1
    assert "verification_pending" not in latest(local_lane), "terminal receipt shape remains backward-compatible"
