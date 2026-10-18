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
        if isinstance(response, Exception):
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
    assert json.loads(lane.log.read_text())["resumable_upload"]["url"] == URL


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
