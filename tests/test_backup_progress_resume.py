"""Gap 22/23 recovery receipts retain measured progress and verification phase."""

# Imported pytest fixtures are intentionally shadowed by test arguments.
# ruff: noqa: F811

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup
from brainlayer.observability_backup import _local_snapshot
from tests.test_weekly_local_gate import local_lane  # noqa: F401
from tests.test_weekly_local_gate_r2 import local_clock  # noqa: F401

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
        for count in [confirmed]:
            kw["progress_callback"](url, count)
            observed.append(latest(local_lane))
        raise backup.DriveUploadStalledError(
            bytes_confirmed=confirmed, total_bytes=path.stat().st_size, attempts=1, session_url=url
        )

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    with pytest.raises(backup.DriveUploadStalledError):
        backup.run_backup(**local_lane, phase="upload")
    if not new_session and confirmed == 25:
        assert observed[-1].get("bytes_confirmed_at") == prior.get("bytes_confirmed_at")
    else:
        assert observed[-1]["bytes_confirmed_at"] != prior.get("bytes_confirmed_at")
    expected = confirmed
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


@pytest.mark.parametrize("change", ["destination", "artifact", "expired"])
def test_fresh_snapshot_never_inherits_resumed_checkpoint(local_lane, local_clock, monkeypatch, change):
    local_clock("UTC", dt.datetime(2026, 11, 20, 12))
    prior = saved(local_lane)
    prior["bytes_confirmed_at"] = "2026-11-21T08:00:00+00:00"
    if change == "destination":
        prior["resumable_upload"]["folder_id"] = "old-folder"
    backup._append_json_log(local_lane["log_path"], prior)
    local_clock("UTC", dt.datetime(2026, 11, 21, 10))
    network(monkeypatch)
    old = Path(prior["snapshot"])

    def folder(*a):
        if change == "artifact":
            old.write_bytes(b"synthetic changed artifact")
        return "folder"

    def upload(path, *a, **kw):
        if path == old:
            assert change == "expired"
            raise backup.DriveSessionExpiredError("synthetic expiry")
        raise RuntimeError("synthetic failure before first upload byte")

    monkeypatch.setattr(backup, "ensure_drive_folder_chain", folder)
    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    with pytest.raises(RuntimeError, match="before first upload byte"):
        backup.run_backup(**local_lane, phase="all")
    rows = [json.loads(line) for line in local_lane["log_path"].read_text().splitlines()]
    fresh = [row for row in rows if row.get("snapshot") and row["snapshot"] != str(old)]
    assert {row["upload_state"] for row in fresh} >= {"pending", "failed"}
    for row in fresh:
        assert "bytes_confirmed" not in row, row
        assert "bytes_confirmed_at" not in row, row
    displayed = _local_snapshot(fresh)
    assert "bytes_confirmed" not in displayed
    assert "bytes_confirmed_at" not in displayed


@pytest.mark.parametrize("confirmed", [10, 0])
def test_real_lower_resume_observation_replaces_saved_checkpoint(local_lane, monkeypatch, confirmed):
    prior = saved(local_lane)
    network(monkeypatch)

    def upload(path, *a, **kw):
        kw["progress_callback"](URL, confirmed)  # A real Drive observation, not initialization.
        row = latest(local_lane)
        assert row["bytes_confirmed"] == confirmed
        assert row["bytes_confirmed_at"] != prior["bytes_confirmed_at"]
        raise backup.DriveUploadStalledError(
            bytes_confirmed=confirmed, total_bytes=path.stat().st_size, attempts=1, session_url=URL
        )

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    with pytest.raises(backup.DriveUploadStalledError):
        backup.run_backup(**local_lane, phase="upload")
    assert latest(local_lane)["bytes_confirmed"] == confirmed


@pytest.mark.parametrize("confirmed", [10, 0])
def test_resumed_uploader_only_publishes_after_drive_checks_the_offset(tmp_path, monkeypatch, confirmed):
    path = tmp_path / "synthetic.db.gz"
    path.write_bytes(b"x" * 100)
    calls = []

    class Session:
        def put(self, *a, **kw):
            calls.append(kw["headers"]["Content-Range"])
            if len(calls) > 1:
                raise RuntimeError("synthetic stop after probe")
            return SimpleNamespace(
                status_code=308,
                headers={"Range": f"bytes=0-{confirmed - 1}"} if confirmed else {},
                raise_for_status=lambda: None,
            )

        def close(self):
            pass

    observed = []

    def progress(url, count):
        assert calls, "initialization has not queried Drive and is not a checkpoint"
        observed.append(count)

    monkeypatch.setattr(backup.requests, "Session", Session)
    with pytest.raises(RuntimeError, match="synthetic stop after probe"):
        backup.upload_file_to_drive_raw(
            path,
            "folder",
            SimpleNamespace(token="synthetic"),
            machine_id="test-machine",
            resume_session_url=URL,
            progress_callback=progress,
        )
    assert observed == [confirmed]


def test_failed_initial_probe_retains_saved_checkpoint_without_claiming_zero(local_lane, monkeypatch):
    prior = saved(local_lane)
    network(monkeypatch)
    monkeypatch.setattr(backup, "get_drive_credentials", lambda: SimpleNamespace(token="synthetic"))
    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS", "1")

    class Session:
        def put(self, *a, **kw):
            raise backup.requests.ConnectionError("synthetic failed probe")

        def close(self):
            pass

    monkeypatch.setattr(backup.requests, "Session", Session)
    with pytest.raises(backup.DriveUploadStalledError):
        backup.run_backup(**local_lane, phase="upload")
    row = latest(local_lane)
    assert row["bytes_confirmed"] == 25
    assert row["bytes_confirmed_at"] == prior["bytes_confirmed_at"]


def test_expired_session_replacement_has_no_checkpoint_before_it_starts(local_lane, monkeypatch):
    saved(local_lane)
    network(monkeypatch)

    def upload(path, *a, **kw):
        if kw.get("resume_session_url"):
            raise backup.DriveSessionExpiredError("synthetic expired session")
        raise RuntimeError("synthetic replacement not started")

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    with pytest.raises(RuntimeError, match="replacement not started"):
        backup.run_backup(**local_lane, phase="upload")
    row = latest(local_lane)
    assert "bytes_confirmed" not in row
    assert "bytes_confirmed_at" not in row


def test_sleep_observation_survives_upload_session_replacement(local_lane, monkeypatch):
    from brainlayer import backup_sleep

    saved(local_lane)
    network(monkeypatch)
    values = {"inclusive": 0, "awake": 0}
    tracker = backup_sleep.SleepTracker(inclusive=lambda: values["inclusive"], awake=lambda: values["awake"])
    monkeypatch.setattr(backup_sleep, "SleepTracker", lambda: tracker)

    def upload(path, *args, **kwargs):
        if kwargs.get("resume_session_url"):
            values["inclusive"] = 21600
            kwargs["telemetry_callback"]({"upload_elapsed_seconds": 0})
            raise backup.DriveSessionExpiredError("synthetic expiry after sleep")
        values.update(inclusive=21630, awake=30)
        kwargs["telemetry_callback"]({"upload_elapsed_seconds": 30, "sleep_intervals": []})
        raise RuntimeError("synthetic stop after replacement telemetry")

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    with pytest.raises(RuntimeError, match="after replacement telemetry"):
        backup.run_backup(**local_lane, phase="upload")
    row = latest(local_lane)
    assert row["upload_wall_seconds"] == 21630
    assert row["upload_slept_seconds"] == 21600
    assert len(row["sleep_intervals"]) == 1


def test_upload_failure_after_sleep_records_pause_without_another_progress_tick(local_lane, monkeypatch):
    from brainlayer import backup_sleep

    saved(local_lane)
    network(monkeypatch)
    values = {"inclusive": 0, "awake": 0}
    tracker = backup_sleep.SleepTracker(inclusive=lambda: values["inclusive"], awake=lambda: values["awake"])
    monkeypatch.setattr(backup_sleep, "SleepTracker", lambda: tracker)

    def upload(*args, **kwargs):
        values["inclusive"] = 21600
        raise RuntimeError("synthetic permanent upload failure after wake")

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    with pytest.raises(RuntimeError, match="permanent upload failure"):
        backup.run_backup(**local_lane, phase="upload")
    row = latest(local_lane)
    assert row["upload_wall_seconds"] == row["upload_slept_seconds"] == 21600
    assert len(row["sleep_intervals"]) == 1
    assert "uploaded the rest" not in row["sleep_note"]


def test_verification_recovery_retains_completed_upload_sleep(local_lane, monkeypatch):
    prior = saved(local_lane, uploaded=True)
    prior.update(
        upload_elapsed_seconds=30,
        upload_wall_seconds=21630,
        upload_slept_seconds=21600,
        sleep_intervals=[{"from": "2026-11-22T07:09:00+02:00", "to": "2026-11-22T13:09:00+02:00", "seconds": 21600}],
        sleep_note="paused 6 h while the Mac slept; uploaded the rest after wake",
    )
    backup._append_json_log(local_lane["log_path"], prior)
    network(monkeypatch)
    monkeypatch.setattr(
        backup, "build_drive_service", lambda: (_ for _ in ()).throw(RuntimeError("synthetic verification stop"))
    )
    with pytest.raises(RuntimeError, match="verification stop"):
        backup.run_backup(**local_lane, phase="upload")
    row = latest(local_lane)
    for field in (
        "upload_elapsed_seconds",
        "upload_wall_seconds",
        "upload_slept_seconds",
        "sleep_intervals",
        "sleep_note",
    ):
        assert row[field] == prior[field]
