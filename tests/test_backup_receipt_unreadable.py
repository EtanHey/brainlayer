"""The receipt journal tolerates foreign plain-text lines and fails loudly only when it holds no receipt.

Production, MBP, 2026-10-20: the weekly ``backup-daily`` step (``BRAINLAYER_BACKUP_PHASE=upload``)
failed with ``error_type: JSONDecodeError`` / ``Expecting value: line 1 column 1 (char 0)``. The
journal is ``backup-daily.log`` itself. Its first 2,396 lines hold 2,393 plain-text
``drive upload progress: N/M bytes`` lines. The 2026-05-13 plist (ec4a544a) redirected the job's
stdout into the journal until 3dca26a2 moved stdout to ``~/Library/Logs/brainlayer`` on 2026-05-18.
The fixtures below copy that shape: plain lines first, then JSON receipts.
"""

import inspect
import json
import plistlib
from pathlib import Path

import pytest

from brainlayer import backup_daily as backup
from brainlayer import maintenance_calendar
from tests.test_weekly_local_gate import local_lane  # noqa: F401  (pytest fixture)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _legacy_progress(count: int, total: int = 5242019358) -> bytes:
    return b"".join(f"drive upload progress: {8388608 * (i + 1)}/{total} bytes\n".encode() for i in range(count))


def _valid_receipt_line(**fields) -> bytes:
    return json.dumps({"attempted_at": "2026-10-19T01:00:17+00:00", "uploaded": False, **fields}).encode() + b"\n"


def _retained_raw(local_lane) -> Path:  # noqa: F811
    staging = local_lane["staging_dir"]
    staging.mkdir(parents=True, exist_ok=True)
    raw = staging / f"{backup._today()}.db"
    raw.write_bytes(b"SQLite format 3\x00 raw snapshot left by a timed-out snapshot step")
    return raw


def test_production_shaped_journal_skips_and_counts_plain_lines(tmp_path):
    journal = tmp_path / "backup-daily.log"
    receipts = [_valid_receipt_line(n=1), _valid_receipt_line(n=2)]
    journal.write_bytes(_legacy_progress(623) + receipts[0] + _legacy_progress(5) + receipts[1])

    found, skipped = backup._scan_backup_receipts(journal)

    assert [r["n"] for r in found] == [1, 2]
    assert skipped == 628
    assert backup._read_backup_receipts(journal) == found


def test_upload_phase_resumes_through_legacy_plain_lines(local_lane, monkeypatch):  # noqa: F811
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    log_path = local_lane["log_path"]
    log_path.write_bytes(_legacy_progress(40) + log_path.read_bytes())
    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", lambda *a, **kw: pytest.fail("second snapshot"))
    monkeypatch.setattr(backup, "get_drive_credentials", lambda: object())
    monkeypatch.setattr(backup, "build_drive_service", lambda: object())
    monkeypatch.setattr(backup, "ensure_drive_folder_chain", lambda *a: "folder")
    monkeypatch.setattr(backup, "assert_drive_folder_owned_by_machine", lambda *a, **kw: None)
    monkeypatch.setattr(backup, "verify_drive_upload", lambda *a, **kw: None)
    uploads = []

    def upload(path, *args, **kwargs):
        uploads.append(path)
        return {"id": "uploaded", "name": path.name}

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    result = backup.run_backup(**local_lane, phase="upload")

    assert result["verified"] is True
    assert result["receipt_journal_skipped_lines"] == 40
    assert uploads == [Path(receipt["snapshot"])]
    assert log_path.read_bytes().startswith(_legacy_progress(40))  # never repaired


@pytest.mark.parametrize(
    ("journal", "error_code", "detail"),
    [
        (b"", "backup_receipt_empty", "is empty (0 bytes)"),
        (_legacy_progress(3), "backup_receipt_unreadable", "holds no JSON receipt object (3 non-receipt lines"),
        (b'{"snapshot": "/x/2026-10-20.db.gz", "upl\n', "backup_receipt_unreadable", "(1 non-receipt lines"),
    ],
    ids=["empty", "plain-lines-only", "truncated-record-only"],
)
def test_upload_phase_fails_with_named_receipt_error(local_lane, journal, error_code, detail):  # noqa: F811
    log_path = local_lane["log_path"]
    log_path.write_bytes(journal)
    raw = _retained_raw(local_lane)
    raw_bytes = raw.read_bytes()

    with pytest.raises(backup.BackupReceiptUnreadableError) as excinfo:
        backup.run_backup(**local_lane, phase="upload")

    assert excinfo.value.error_code == error_code
    assert str(log_path) in str(excinfo.value)
    assert detail in str(excinfo.value)

    receipt = json.loads(log_path.read_bytes().splitlines()[-1])
    assert receipt["error_code"] == error_code
    assert receipt["error_type"] == "BackupReceiptUnreadableError"
    assert str(log_path) in receipt["error"]
    assert receipt["upload_state"] == "failed"
    assert receipt["retained_raw_snapshots"] == [raw.name]
    assert receipt["local_uncompressed_snapshot"] == str(raw)
    assert receipt["raw_snapshot_retained_reason"] == "No verified gzip replaced this raw snapshot"

    # Fail closed: nothing is deleted or "repaired".
    assert raw.read_bytes() == raw_bytes
    assert log_path.read_bytes().startswith(journal)


def test_receipt_error_never_quotes_the_unreadable_line(tmp_path):
    secret_line = b'{"resumable_upload": {"url": "https://www.googleapis.com/upload/drive/v3/files?upload_id=SECRET"\n'
    journal = tmp_path / "backup-daily.log"
    journal.write_bytes(secret_line)
    with pytest.raises(backup.BackupReceiptUnreadableError) as excinfo:
        backup._read_backup_receipts(journal)
    assert "SECRET" not in str(excinfo.value)
    assert excinfo.value.error_code == "backup_receipt_unreadable"


def test_receipt_error_is_a_value_error_so_degrading_readers_keep_degrading(tmp_path):
    journal = tmp_path / "backup-daily.log"
    journal.write_bytes(_legacy_progress(2))
    assert issubclass(backup.BackupReceiptUnreadableError, ValueError)
    assert backup._stalled_uploads(journal, {"attempted_at": "2026-10-20T00:00:00+00:00"}, tmp_path) == []


def test_torn_final_record_is_tolerated_and_not_counted(tmp_path):
    journal = tmp_path / "backup-daily.log"
    journal.write_bytes(_valid_receipt_line() + b'{"snapshot": "/x/2026-10-20.db.gz", "upl')
    assert backup._scan_backup_receipts(journal) == ([json.loads(_valid_receipt_line())], 0)


def test_missing_journal_still_means_nothing_pending(local_lane):  # noqa: F811
    assert not local_lane["log_path"].exists()
    assert backup.run_backup(**local_lane, phase="upload")["upload_state"] == "nothing_pending"


def test_job_stdout_never_lands_in_the_receipt_journal():
    """The writer of the plain-text lines was stdout. Its routes must not reach the journal."""
    with (REPO_ROOT / "scripts/launchd/com.brainlayer.backup-daily.plist").open("rb") as handle:
        plist = plistlib.load(handle)
    journal_suffix = "/.local/share/brainlayer/logs/backup-daily.log"
    for key in ("StandardOutPath", "StandardErrorPath"):
        assert not plist[key].endswith(journal_suffix), key
        assert plist[key].startswith("__HOME__/Library/Logs/brainlayer/"), key
    # The calendar routes each step's stdout to <logs-dir>/<step>.out.log, never to the journal name.
    source = inspect.getsource(maintenance_calendar._execute_step)
    assert '(logs / f"{step.name}.out.log").open("ab") as out' in source
    assert backup.DEFAULT_LOG_PATH.name not in {f"{step}.out.log" for step in ("backup-daily", "backup-snapshot")}


def test_skipped_lines_keep_deleting_readers_fail_closed(tmp_path):
    """A skipped line may have been the receipt that superseded what remains (#1384 contract)."""
    journal = tmp_path / "backup-daily.log"
    journal.write_bytes(_legacy_progress(2) + _valid_receipt_line())
    target = tmp_path / "2026-10-20.db.gz"
    target.write_bytes(b"existing local artifact")

    with pytest.raises(backup.BackupReceiptUnreadableError) as excinfo:
        backup._read_backup_receipts(journal, strict=True)
    assert excinfo.value.error_code == "backup_receipt_unreadable"
    assert "2 non-receipt lines" in str(excinfo.value)
    assert backup._retained_target_collision(journal, target) is True
    assert backup._stalled_uploads(journal, {"attempted_at": "2026-10-20T00:00:00+00:00"}, tmp_path) == []
