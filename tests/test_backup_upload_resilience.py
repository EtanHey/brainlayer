"""Drive recovery with fake transport, isolated receipts, and tiny local artifacts."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup
from tests.test_backup_daily import _DriveResponse, _stub_backup_for_drive_upload

URL = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=test"


@pytest.fixture
def lane(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS", "2")
    snapshot = tmp_path / "2026-10-17.db.gz"
    snapshot.write_bytes(b"abcdef")
    _stub_backup_for_drive_upload(backup, monkeypatch, snapshot)
    monkeypatch.setattr(backup, "verify_drive_upload", lambda *a, **kw: None)
    monkeypatch.setattr(backup, "_sleep", lambda seconds: None)
    posts, puts, outcomes, creations = [], [], [], []

    def post(*args, **kwargs):
        posts.append(kwargs)
        return _DriveResponse(200, headers={"Location": URL})

    def put(session, url, *, headers, **kwargs):
        puts.append((url, headers["Content-Range"]))
        assert outcomes, "unexpected upload request"
        response = outcomes.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def create(*args, **kwargs):
        path = tmp_path / f"{kwargs['date_stamp']}.db.gz"
        creations.append(path)
        path.write_bytes(b"abcdef")
        return SimpleNamespace(gzip_path=path, uncompressed_path=None, sentinel_chunks=1, local_retention_deleted=[])

    monkeypatch.setattr(backup.requests, "post", post)
    monkeypatch.setattr(backup, "_drive_put_with_deadline", put)
    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", create)
    log = tmp_path / "receipt.jsonl"

    def run(day="2026-10-17", **kwargs):
        return backup.run_backup(
            db_path=tmp_path / "unused.db",
            staging_dir=tmp_path,
            date_stamp=day,
            log_path=log,
            remove_local_after_upload=False,
            **kwargs,
        )

    return SimpleNamespace(
        run=run,
        posts=posts,
        puts=puts,
        outcomes=outcomes,
        creations=creations,
        snapshot=snapshot,
        log=log,
        root=tmp_path,
    )


def stalled(lane):
    lane.outcomes.extend(
        [ConnectionError("offline"), _DriveResponse(308), ConnectionError("offline"), _DriveResponse(308)]
    )
    with pytest.raises(backup.DriveUploadStalledError):
        lane.run()
    return json.loads(lane.log.read_text().splitlines()[-1])


def test_six_minute_dns_outage_recovers_within_default_budget(lane, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS")
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(backup.time, "monotonic", lambda: clock[0])

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    def put(*args, **kwargs):
        if clock[0] < 360:
            raise backup.requests.ConnectionError("DNS lookup failed")
        if kwargs["headers"]["Content-Range"].startswith("bytes */"):
            return _DriveResponse(308)
        return _DriveResponse(200, payload={"id": "done"})

    monkeypatch.setattr(backup, "_sleep", sleep)
    monkeypatch.setattr(backup, "_drive_put_with_deadline", put)
    assert lane.run()["verified"] is True
    assert 360 <= clock[0] <= 1200
    assert sleeps == sorted(sleeps)


@pytest.mark.parametrize("where", ["chunk", "offset"])
@pytest.mark.parametrize("code", [400, 401, 403])
def test_http_auth_and_client_errors_fail_without_backoff(lane, monkeypatch, where, code):
    lane.outcomes.extend(([ConnectionError("offline")] if where == "offset" else []) + [_DriveResponse(code)])
    monkeypatch.setattr(backup, "_sleep", lambda seconds: pytest.fail("HTTP client error retried"))
    with pytest.raises(RuntimeError, match=f"HTTP {code}"):
        lane.run()


def test_stalled_receipt_resumes_same_session_without_new_snapshot(lane):
    receipt = stalled(lane)
    state = receipt["resumable_upload"]
    assert state == {
        "url": URL,
        "folder_id": "folder-id",
        "path": str(lane.snapshot.resolve()),
        "size": 6,
        "sha256": hashlib.sha256(b"abcdef").hexdigest(),
    }
    assert lane.log.stat().st_mode & 0o077 == 0
    lane.outcomes.extend(
        [_DriveResponse(308, headers={"Range": "bytes=0-2"}), _DriveResponse(200, payload={"id": "done"})]
    )
    result = lane.run("2026-10-18")
    assert result["uploaded"] and result["verified"]
    assert len(lane.creations) == len(lane.posts) == 1
    assert lane.puts[-2:] == [(URL, "bytes */6"), (URL, "bytes 3-5/6")]
    assert result["snapshot"] == str(lane.snapshot)


@pytest.mark.parametrize("change", ["digest", "missing", "expired", "wrong_owner", "unsafe_url"])
def test_unusable_resume_starts_fresh_and_only_removes_proven_stale(lane, change):
    receipt = stalled(lane)
    if change == "digest":
        lane.snapshot.write_bytes(b"ghijkl")  # same length, different artifact
    elif change == "missing":
        lane.snapshot.unlink()
    elif change == "wrong_owner":
        receipt["machine_id"] = "another-machine"
        lane.log.write_text(json.dumps(receipt) + "\n")
    elif change == "unsafe_url":
        receipt["resumable_upload"]["url"] = "https://evil.test/steal"
        lane.log.write_text(json.dumps(receipt) + "\n")
    else:
        lane.outcomes.append(_DriveResponse(404))
    lane.outcomes.append(_DriveResponse(200, payload={"id": "new"}))

    def verify(*args, **kwargs):
        if change in {"expired", "unsafe_url"}:
            assert lane.snapshot.exists()
        return {"verified": True}

    from unittest.mock import patch

    with patch.object(backup, "verify_sqlite_backup_artifact", verify):
        result = lane.run("2026-10-18")
    assert result["uploaded"]
    assert len(lane.posts) == len(lane.creations) == 2
    assert lane.snapshot.exists() is (change in {"digest", "wrong_owner"})
    if change in {"expired", "unsafe_url"}:
        assert result["superseded_stalled_deleted"] == [lane.snapshot.name]


def test_verified_copy_and_unrecorded_artifact_survive_stale_cleanup(lane):
    receipt = stalled(lane)
    verified = lane.root / "2026-10-16.db.gz"
    verified.write_bytes(b"verified")
    unknown = lane.root / "2026-10-15.db.gz"
    unknown.write_bytes(b"unknown")
    protected = {**receipt, "snapshot": str(verified), "uploaded": True, "verified": True}
    lane.log.write_text(lane.log.read_text() + json.dumps(protected) + "\n")
    lane.outcomes.extend([_DriveResponse(410), _DriveResponse(200, payload={"id": "new"})])
    lane.run("2026-10-18")
    assert not lane.snapshot.exists()
    assert verified.read_bytes() == b"verified"
    assert unknown.read_bytes() == b"unknown"


def test_resume_outage_keeps_artifact_instead_of_creating_or_deleting(lane):
    stalled(lane)
    lane.outcomes.extend([ConnectionError("offline"), ConnectionError("offline")])
    with pytest.raises(backup.DriveUploadStalledError):
        lane.run("2026-10-18")
    assert lane.snapshot.exists()
    assert len(lane.creations) == len(lane.posts) == 1


def test_configured_network_budget_expires_and_saves_resume_state(lane, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS")
    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_RETRY_BUDGET_SECONDS", "90")
    sleeps = []
    monkeypatch.setattr(backup, "_sleep", sleeps.append)
    lane.outcomes.extend([ConnectionError("offline")] * 6)
    with pytest.raises(backup.DriveUploadStalledError):
        lane.run()
    assert sum(sleeps) == pytest.approx(90, abs=0.01)
    assert json.loads(lane.log.read_text().splitlines()[-1])["resumable_upload"]["url"] == URL


def test_mutation_during_stall_cannot_rebind_session_to_different_bytes(lane):
    identity = hashlib.sha256(b"abcdef").hexdigest()
    original = backup._drive_put_with_deadline

    def mutate(*args, **kwargs):
        lane.snapshot.write_bytes(b"ghijkl")
        return original(*args, **kwargs)

    from unittest.mock import patch

    with patch.object(backup, "_drive_put_with_deadline", mutate):
        receipt = stalled(lane)
    assert receipt["resumable_upload"]["sha256"] == identity


def test_previously_verified_path_cannot_be_reclaimed_by_a_later_stall(lane):
    receipt = stalled(lane)
    verified = {**receipt, "uploaded": False, "verified": True}
    lane.log.write_text(json.dumps(verified) + "\n" + json.dumps(receipt) + "\n")
    lane.outcomes.append(_DriveResponse(200, payload={"id": "new"}))
    lane.run("2026-10-18")
    assert lane.snapshot.exists()
    assert len(lane.posts) == 2


def test_real_requests_http_error_from_offset_query_is_not_retried(lane, monkeypatch):
    response = backup.requests.Response()
    response.status_code = 401
    lane.outcomes.extend([ConnectionError("offline"), response])
    monkeypatch.setattr(backup, "_sleep", lambda seconds: pytest.fail("auth error retried"))
    with pytest.raises(backup.requests.HTTPError):
        lane.run()


def test_changed_destination_id_never_reuses_saved_session(lane, monkeypatch):
    stalled(lane)
    monkeypatch.setattr(backup, "ensure_drive_folder_chain", lambda *a, **kw: "replacement-folder")
    lane.outcomes.append(_DriveResponse(200, payload={"id": "new"}))
    result = lane.run("2026-10-18")
    assert result["verified"]
    assert len(lane.posts) == 2
    assert lane.puts[-1] == (URL, "bytes 0-5/6")


def test_corrupt_receipt_disables_reclamation(lane):
    stalled(lane)
    lane.log.write_text(lane.log.read_text() + "truncated-json\n")
    lane.outcomes.append(_DriveResponse(200, payload={"id": "new"}))
    lane.run("2026-10-18")
    assert lane.snapshot.exists()
    assert len(lane.posts) == 2


def test_new_process_resumes_using_only_persisted_receipt(lane):
    import subprocess
    import sys

    stalled(lane)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
import pytest
from tests.test_backup_upload_resilience import lane, _DriveResponse
with pytest.MonkeyPatch.context() as patch:
    state = lane.__wrapped__(Path(sys.argv[1]), patch)
    state.outcomes.extend([_DriveResponse(308, headers={"Range": "bytes=0-2"}),
                           _DriveResponse(200, payload={"id": "resumed"})])
    result = state.run("2026-10-18")
    assert result["uploaded"] and result["verified"]
    assert state.posts == [] and state.creations == []
    assert state.puts[-1][1] == "bytes 3-5/6"
""",
            str(lane.root),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert child.returncode == 0, child.stderr


def test_artifact_rechecked_after_destination_lookup_before_resumed_put(lane, monkeypatch):
    stalled(lane)

    def lookup(*args, **kwargs):
        lane.snapshot.write_bytes(b"ghijkl")
        return "folder-id"

    monkeypatch.setattr(backup, "ensure_drive_folder_chain", lookup)
    lane.outcomes.append(_DriveResponse(200, payload={"id": "new"}))
    lane.run("2026-10-18")
    assert len(lane.posts) == 2
    assert lane.puts[-1][1] == "bytes 0-5/6"


@pytest.mark.parametrize("failure", ["capacity", "create", "upload", "verify"])
def test_r1_stalled_copy_survives_until_newer_drive_copy_verifies(lane, monkeypatch, failure):
    stalled(lane)
    lane.outcomes.append(_DriveResponse(404))
    if failure in {"capacity", "create"}:

        def fail_create(*args, **kwargs):
            assert lane.snapshot.exists(), "only retained copy was deleted before create"
            raise RuntimeError("Insufficient free space for backup" if failure == "capacity" else "VACUUM failed")

        monkeypatch.setattr(backup, "create_sqlite_backup_artifact", fail_create)
    elif failure == "upload":
        lane.outcomes.extend([ConnectionError("offline"), _DriveResponse(308)] * 2)
    else:
        lane.outcomes.append(_DriveResponse(200, payload={"id": "new"}))
        monkeypatch.setattr(backup, "verify_sqlite_backup_artifact", lambda *a, **kw: {"verified": False})
    if failure == "verify":
        assert lane.run("2026-10-18")["verified"] is False
    else:
        with pytest.raises(RuntimeError):
            lane.run("2026-10-18")
    assert lane.snapshot.read_bytes() == b"abcdef"
    receipt = json.loads(lane.log.read_text().splitlines()[-1])
    assert not receipt.get("superseded_stalled_deleted")
    if failure == "capacity":
        assert lane.snapshot.name in receipt["error_type"]


@pytest.mark.parametrize("abandon", ["expired", "unsafe_url"])
def test_r1_reclamation_follows_new_upload_and_verification(lane, monkeypatch, abandon):
    receipt = stalled(lane)
    if abandon == "unsafe_url":
        receipt["resumable_upload"]["url"] = "https://evil.test/steal"
        lane.log.write_text(json.dumps(receipt) + "\n")
    else:
        lane.outcomes.append(_DriveResponse(404))
    lane.outcomes.append(_DriveResponse(200, payload={"id": "new"}))

    def verify(*args, **kwargs):
        assert lane.snapshot.exists(), "stale backup deleted before Drive verification"
        return {"verified": True}

    monkeypatch.setattr(backup, "verify_sqlite_backup_artifact", verify)
    result = lane.run("2026-10-18")
    assert result["superseded_stalled_deleted"] == [lane.snapshot.name]
    assert not lane.snapshot.exists()


@pytest.mark.parametrize("hours", [36, 96])
def test_r1_old_snapshot_not_relabelled_by_recent_attempt(lane, hours):
    import datetime as dt

    receipt = stalled(lane)
    receipt["snapshot_created_at"] = (dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours)).isoformat()
    lane.log.write_text(json.dumps(receipt) + "\n")
    lane.outcomes.append(_DriveResponse(200, payload={"id": "fresh"}))
    result = lane.run("2026-10-18")
    assert len(lane.posts) == 2
    assert result["snapshot_created_at"] != receipt["snapshot_created_at"]


def test_r1_weekly_resume_obeys_snapshot_freshness_window(lane, monkeypatch):
    import datetime as dt

    receipt = stalled(lane)
    receipt["snapshot_created_at"] = (dt.datetime.now(dt.UTC) - dt.timedelta(hours=8)).isoformat()
    lane.log.write_text(json.dumps(receipt) + "\n")
    monkeypatch.setenv("BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS", "6")
    lane.outcomes.append(_DriveResponse(200, payload={"id": "fresh"}))
    assert lane.run("2026-10-18")["verified"]
    assert len(lane.posts) == 2


def test_r1_reuse_and_observability_age_snapshot_not_verification(lane):
    import datetime as dt

    from brainlayer.observability_backup import _daily_snapshot

    now = dt.datetime.now(dt.UTC)
    old = (now - dt.timedelta(days=4)).isoformat()
    receipt = {
        "attempted_at": now.isoformat(),
        "snapshot_created_at": old,
        "db": str(lane.root / "unused.db"),
        "snapshot": str(lane.snapshot),
        "uploaded": True,
        "verified": True,
        "drive_file": {"id": "old", "name": lane.snapshot.name},
        "backup_log_provenance": "real",
    }
    lane.log.write_text(json.dumps(receipt) + "\n")
    assert backup._recent_verified_backup_for_reuse(lane.log, lane.root / "unused.db", now=now, max_age_hours=6) is None
    assert _daily_snapshot([receipt])[0]["last_at"] == old.replace("+00:00", "Z")


def test_r1_resumed_receipt_preserves_original_snapshot_time(lane):
    receipt = stalled(lane)
    assert receipt["snapshot_created_at"]
    lane.outcomes.extend([_DriveResponse(308), _DriveResponse(200, payload={"id": "done"})])
    result = lane.run("2026-10-18")
    assert result["snapshot_created_at"] == receipt["snapshot_created_at"]


@pytest.mark.parametrize("interruption", ["timeout", "stop", "auth"])
def test_r1_every_prefix_interruption_retains_private_resume_state(lane, interruption):
    import signal

    lane.outcomes.append(_DriveResponse(308, headers={"Range": "bytes=0-2"}))
    error = {
        "timeout": backup.BackupTimeoutError("wall deadline"),
        "stop": backup.BackupStoppedError(signal.SIGTERM),
        "auth": RuntimeError("HTTP 401"),
    }[interruption]
    lane.outcomes.append(error)
    with pytest.raises(type(error)):
        lane.run()
    receipt = json.loads(lane.log.read_text().splitlines()[-1])
    assert receipt["resumable_upload"]["url"] == URL
    assert receipt["bytes_confirmed"] == 3
    assert lane.log.stat().st_mode & 0o077 == 0
    lane.outcomes.extend(
        [_DriveResponse(308, headers={"Range": "bytes=0-2"}), _DriveResponse(200, payload={"id": "done"})]
    )
    assert lane.run("2026-10-18")["verified"]
    assert len(lane.posts) == 1


def test_r1_session_is_durable_before_first_chunk_for_supervisor_exit(lane, monkeypatch):
    def interrupted(*args, **kwargs):
        receipt = json.loads(lane.log.read_text().splitlines()[-1])
        assert receipt["resumable_upload"]["url"] == URL
        raise backup.BackupTimeoutError("supervisor exit")

    monkeypatch.setattr(backup, "_drive_put_with_deadline", interrupted)
    with pytest.raises(backup.BackupTimeoutError):
        lane.run()


def test_r1_session_init_dns_outage_uses_network_backoff(lane, monkeypatch):
    original = backup.requests.post
    attempts = []

    def post(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise backup.requests.ConnectionError("DNS offline")
        return original(*args, **kwargs)

    monkeypatch.delenv("BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS")
    monkeypatch.setattr(backup.requests, "post", post)
    sleeps = []
    monkeypatch.setattr(backup, "_sleep", sleeps.append)
    lane.outcomes.append(_DriveResponse(200, payload={"id": "done"}))
    assert lane.run()["verified"]
    assert sleeps == [30, 60]


def test_r1_recovery_budget_reserves_receipt_time_before_run_deadline(lane, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS")
    monkeypatch.setenv("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", "20")
    clock = [100.0]
    monkeypatch.setattr(backup.time, "monotonic", lambda: clock[0])
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(backup, "_sleep", sleep)
    lane.outcomes.extend([ConnectionError("offline")] * 50)
    with pytest.raises((backup.BackupTimeoutError, backup.DriveUploadStalledError)):
        lane.run()
    assert sum(sleeps) <= 15
    assert json.loads(lane.log.read_text().splitlines()[-1])["resumable_upload"]["url"] == URL


def test_r1_http_error_and_receipt_redact_session_credentials(lane):
    response = backup.requests.Response()
    response.status_code = 401
    response.url = URL
    lane.outcomes.extend([_DriveResponse(308, headers={"Range": "bytes=0-2"}), response])
    with pytest.raises(backup.requests.HTTPError) as error:
        lane.run()
    assert URL not in str(error.value)
    assert "upload_id=test" not in str(error.value)
    receipt = json.loads(lane.log.read_text().splitlines()[-1])
    assert "upload_id=test" not in receipt["error"]
    assert receipt["resumable_upload"]["url"] == URL


def test_r1_failure_traceback_and_retry_log_redact_session_url(lane, monkeypatch, capsys):
    def fail(**kwargs):
        raise RuntimeError(f"for url: {URL}; upload_id=test")

    monkeypatch.setattr(backup, "run_backup", fail)
    monkeypatch.setattr(backup, "get_db_path", lambda: lane.root / "unused.db")
    assert backup._run_backup_process(20) == 1
    output = capsys.readouterr().out
    assert URL not in output and "upload_id=test" not in output
    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS", "2")

    def network_fail():
        raise ConnectionError(f"for url: {URL}")

    with pytest.raises(RuntimeError):
        backup._retry_transient_drive_call(network_fail, what="Drive probe")
    assert "upload_id=test" not in capsys.readouterr().out


def test_r1_receipt_append_writes_whole_line_flushes_and_fsyncs(tmp_path, monkeypatch):
    from pathlib import Path

    events = []
    real_open = Path.open

    class TrackedHandle:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __getattr__(self, key):
            return getattr(self.handle, key)

        def write(self, line):
            events.append(("write", line))
            return self.handle.write(line)

        def flush(self):
            events.append(("flush", None))
            return self.handle.flush()

    monkeypatch.setattr(Path, "open", lambda path, *args, **kwargs: TrackedHandle(real_open(path, *args, **kwargs)))
    real_fsync = backup.os.fsync

    def fsync(fd):
        events.append(("fsync", None))
        real_fsync(fd)

    monkeypatch.setattr(backup.os, "fsync", fsync)
    backup._append_json_log(tmp_path / "receipt.jsonl", {"error": "example"})
    assert [event[0] for event in events] == ["write", "flush", "fsync"]
    assert events[0][1].endswith("\n") and json.loads(events[0][1])["error"] == "example"


def test_r1_truncated_final_receipt_preserves_last_durable_session(lane):
    stalled(lane)
    lane.log.write_text(lane.log.read_text() + '{"resumable_upload":')
    lane.outcomes.extend([_DriveResponse(308), _DriveResponse(200, payload={"id": "done"})])
    assert lane.run("2026-10-18")["verified"]
    assert len(lane.posts) == 1
    # Appending after the crash tail must preserve independently parseable JSONL.
    receipts = [json.loads(line) for line in lane.log.read_text().splitlines()]
    assert receipts[-1]["verified"]


def test_r1_missing_range_after_confirmed_prefix_probes_without_rewinding(lane):
    lane.outcomes.extend(
        [
            _DriveResponse(308, headers={"Range": "bytes=0-2"}),
            _DriveResponse(308),
            _DriveResponse(308, headers={"Range": "bytes=0-2"}),
            _DriveResponse(200, payload={"id": "done"}),
        ]
    )
    assert lane.run()["verified"]
    assert [item[1] for item in lane.puts] == ["bytes 0-5/6", "bytes 3-5/6", "bytes */6", "bytes 3-5/6"]


def test_r1_newest_stalled_candidate_wins_after_older_key_is_updated(lane):
    import datetime as dt

    receipt = stalled(lane)
    now = dt.datetime.now(dt.UTC)
    receipt["snapshot_created_at"] = (now - dt.timedelta(hours=1)).isoformat()
    receipt["attempted_at"] = (now - dt.timedelta(hours=3)).isoformat()
    older_path = lane.root / "2026-10-16.db.gz"
    older_path.write_bytes(b"abcdef")
    older = {
        **receipt,
        "snapshot": str(older_path),
        "snapshot_created_at": (now - dt.timedelta(hours=2)).isoformat(),
        "attempted_at": (now - dt.timedelta(hours=2)).isoformat(),
        "resumable_upload": {**receipt["resumable_upload"], "path": str(older_path), "url": URL + "-older"},
    }
    updated = {**receipt, "attempted_at": (now - dt.timedelta(minutes=30)).isoformat()}
    lane.log.write_text("".join(json.dumps(r) + "\n" for r in [receipt, older, updated]))
    lane.outcomes.extend([_DriveResponse(308), _DriveResponse(200, payload={"id": "done"})])
    result = lane.run("2026-10-18")
    assert result["snapshot"] == str(lane.snapshot)
    assert lane.puts[-2][0] == URL


def test_r1_same_day_fresh_snapshot_cannot_overwrite_retained_stalled_copy(lane, monkeypatch):
    stalled(lane)
    lane.outcomes.extend([_DriveResponse(404), _DriveResponse(200, payload={"id": "new"})])

    def overwrite(*args, **kwargs):
        lane.snapshot.write_bytes(b"fresh!")
        return SimpleNamespace(
            gzip_path=lane.snapshot, uncompressed_path=None, sentinel_chunks=1, local_retention_deleted=[]
        )

    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", overwrite)
    monkeypatch.setattr(backup, "verify_sqlite_backup_artifact", lambda *a, **kw: {"verified": False})
    with pytest.raises(RuntimeError, match="retaining.*2026-10-17.db.gz"):
        lane.run("2026-10-17")
    assert lane.snapshot.read_bytes() == b"abcdef"


def test_r1_weekly_snapshot_expiring_during_upload_cannot_authorize_vacuum(lane, monkeypatch):
    import datetime as dt

    receipt = stalled(lane)
    clock = [dt.datetime.now(dt.UTC)]
    created = (clock[0] - dt.timedelta(hours=5, minutes=59)).isoformat()
    receipt["snapshot_created_at"] = created
    lane.log.write_text(json.dumps(receipt) + "\n")
    real_datetime = dt.datetime

    class Clock(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz)

    monkeypatch.setattr(backup.dt, "datetime", Clock)
    monkeypatch.setenv("BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS", "6")
    lane.outcomes.extend([_DriveResponse(308), _DriveResponse(200, payload={"id": "done"})])

    def verify(*args, **kwargs):
        clock[0] += dt.timedelta(minutes=2)
        return {"verified": True}

    monkeypatch.setattr(backup, "verify_sqlite_backup_artifact", verify)
    result = lane.run("2026-10-18")
    assert result["uploaded"]
    assert result["verified"] is False
    assert "snapshot" in result["verification_error"]
    assert lane.snapshot.exists()


def _r2_completed_with_failed_verification(lane, monkeypatch, failure):
    """Replay Cursor's _r2_repro{,2}: completed upload, unsuccessful verification."""
    if failure == "weekly_age":
        import datetime as dt

        receipt = stalled(lane)
        clock = [dt.datetime.now(dt.UTC)]
        receipt["snapshot_created_at"] = (clock[0] - dt.timedelta(hours=5, minutes=59)).isoformat()
        lane.log.write_text(json.dumps(receipt) + "\n")
        real_datetime = dt.datetime

        class Clock(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return clock[0].astimezone(tz)

        monkeypatch.setattr(backup.dt, "datetime", Clock)
        monkeypatch.setenv("BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS", "6")
        lane.outcomes.extend(
            [_DriveResponse(308, headers={"Range": "bytes=0-2"}), _DriveResponse(200, payload={"id": "done"})]
        )

        def verify(*args, **kwargs):
            clock[0] += dt.timedelta(minutes=2)
            return {"verified": True}
    else:
        lane.outcomes.extend(
            [_DriveResponse(308, headers={"Range": "bytes=0-2"}), _DriveResponse(200, payload={"id": "done"})]
        )

        def verify(*args, **kwargs):
            if failure == "bare_local_verify":
                return {"verified": False}
            return {"verified": False, "verification_error": "local gzip verification failed"}

    monkeypatch.setattr(backup, "verify_sqlite_backup_artifact", verify)
    return lane.run("2026-10-18" if failure == "weekly_age" else "2026-10-17")


@pytest.mark.parametrize("failure", ["weekly_age", "local_verify", "bare_local_verify"])
def test_r2_completed_upload_is_not_an_interrupted_session(lane, monkeypatch, failure):
    result = _r2_completed_with_failed_verification(lane, monkeypatch, failure)
    assert result["uploaded"] is True and result["drive_file"]["id"] == "done"
    assert result["verified"] is False
    assert "resumable_upload" not in result
    assert result.get("error_code") != "drive_upload_interrupted"
    reason = {
        "weekly_age": "weekly backup safety window",
        "local_verify": "local gzip verification failed",
        "bare_local_verify": "local backup verification failed",
    }[failure]
    assert reason in result["verification_error"]
    receipt = json.loads(lane.log.read_text().splitlines()[-1])
    assert "resumable_upload" not in receipt and URL not in json.dumps(receipt)
    assert receipt.get("error_code") != "drive_upload_interrupted"
    assert reason in receipt["verification_error"]
    monkeypatch.delenv("BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS", raising=False)
    monkeypatch.setattr(backup, "verify_sqlite_backup_artifact", lambda *a, **kw: {"verified": True})
    lane.outcomes.append(_DriveResponse(200, payload={"id": "new"}))
    assert lane.run("2026-10-19")["verified"]
    assert len(lane.posts) == 2
    assert lane.puts[-1][1] == "bytes 0-5/6"


def test_r2_truncated_verification_receipt_cannot_revive_completed_session(lane, monkeypatch):
    _r2_completed_with_failed_verification(lane, monkeypatch, "local_verify")
    lines = lane.log.read_text().splitlines()
    lane.log.write_text("\n".join(lines[:-1]) + '\n{"uploaded":')
    monkeypatch.setattr(backup, "verify_sqlite_backup_artifact", lambda *a, **kw: {"verified": True})
    lane.outcomes.append(_DriveResponse(200, payload={"id": "new"}))
    assert lane.run("2026-10-18")["verified"]
    assert len(lane.posts) == 2, "crash tail resurrected an already completed upload session"
    assert lane.puts[-1][1] == "bytes 0-5/6"


@pytest.mark.parametrize("bad_age", ["invalid", "nan", "inf", "-1"])
def test_r2_malformed_weekly_age_preserves_upload_and_denies_vacuum(lane, monkeypatch, bad_age):
    monkeypatch.setenv("BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS", "6")
    lane.outcomes.append(_DriveResponse(200, payload={"id": "completed"}))

    def verified_then_bad_config(*args, **kwargs):
        # The outer reuse gate already validated the initial configuration.
        monkeypatch.setenv("BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS", bad_age)
        return {"verified": True}

    monkeypatch.setattr(backup, "verify_sqlite_backup_artifact", verified_then_bad_config)
    result = lane.run()
    assert result["uploaded"] is True and result["drive_file"]["id"] == "completed"
    assert result["verified"] is False
    assert "invalid BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS" in result["verification_error"]
    assert "resumable_upload" not in result and result.get("error_code") != "drive_upload_interrupted"
    receipt = json.loads(lane.log.read_text().splitlines()[-1])
    assert receipt["uploaded"] is True and receipt["verified"] is False
    assert receipt["drive_file"]["id"] == "completed"
    assert "invalid BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS" in receipt["verification_error"]
    assert lane.snapshot.exists()


@pytest.mark.parametrize("forthcoming", [0, backup.COPY_CHUNK_BYTES])
def test_r2_raw_floor_failure_names_retained_stalled_backup(lane, monkeypatch, forthcoming):
    stalled(lane)
    lane.outcomes.append(_DriveResponse(404))
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda path: SimpleNamespace(free=backup.MIN_RAW_FREE_BYTES - 1))

    def floor_during_snapshot(*args, **kwargs):
        backup._require_raw_floor(lane.root, forthcoming=forthcoming)

    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", floor_during_snapshot)
    with pytest.raises(backup.RetainedStalledBackupCapacityError):
        lane.run("2026-10-18")
    receipt = json.loads(lane.log.read_text().splitlines()[-1])
    assert lane.snapshot.name in receipt["error_type"]
    assert lane.snapshot.name in receipt["error"]
    assert "raw free space floor breached" in receipt["error"]
    assert lane.snapshot.read_bytes() == b"abcdef"
    assert not receipt.get("superseded_stalled_deleted")
