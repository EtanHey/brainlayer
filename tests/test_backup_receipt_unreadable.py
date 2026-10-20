"""An unreadable backup receipt journal fails the upload phase loudly, never as a bare JSONDecodeError.

Production, MBP, 2026-10-20: the weekly ``backup-daily`` step (``BRAINLAYER_BACKUP_PHASE=upload``)
failed with ``error_type: JSONDecodeError`` / ``Expecting value: line 1 column 1 (char 0)``. The
journal it read is ``backup-daily.log`` itself, and its first 2,396 lines are legacy plain-text
``drive upload progress: N/M bytes`` lines, which ``_read_backup_receipts`` refuses mid-file.
"""

import json
from pathlib import Path

import pytest

from brainlayer import backup_daily as backup
from tests.test_weekly_local_gate import local_lane  # noqa: F401  (pytest fixture)

LEGACY_PROGRESS_LINE = b"drive upload progress: 8388608/5242019358 bytes\n"


def _valid_receipt_line() -> bytes:
    return json.dumps({"attempted_at": "2026-10-19T01:00:17+00:00", "uploaded": False}).encode() + b"\n"


def _retained_raw(local_lane) -> Path:  # noqa: F811
    staging = local_lane["staging_dir"]
    staging.mkdir(parents=True, exist_ok=True)
    raw = staging / f"{backup._today()}.db"
    raw.write_bytes(b"SQLite format 3\x00 raw snapshot left by a timed-out snapshot step")
    return raw


@pytest.mark.parametrize(
    ("journal", "error_code", "detail"),
    [
        (b"", "backup_receipt_empty", "is empty (0 bytes)"),
        (b'{"snapshot": "/x/2026-10-20.db.gz", "upl\n' + _valid_receipt_line(), "backup_receipt_unreadable", "line 1"),
        (LEGACY_PROGRESS_LINE + _valid_receipt_line(), "backup_receipt_unreadable", "line 1"),
    ],
    ids=["empty", "truncated-record", "legacy-plain-text-line"],
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
    assert "JSONDecodeError" not in receipt["error_type"]
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
    journal.write_bytes(secret_line + _valid_receipt_line())
    with pytest.raises(backup.BackupReceiptUnreadableError) as excinfo:
        backup._read_backup_receipts(journal)
    assert "SECRET" not in str(excinfo.value)
    assert excinfo.value.error_code == "backup_receipt_unreadable"


def test_receipt_error_is_a_value_error_so_degrading_readers_keep_degrading(tmp_path):
    journal = tmp_path / "backup-daily.log"
    journal.write_bytes(LEGACY_PROGRESS_LINE + _valid_receipt_line())
    assert backup._stalled_uploads(journal, {"attempted_at": "2026-10-20T00:00:00+00:00"}, tmp_path) == []
    assert issubclass(backup.BackupReceiptUnreadableError, ValueError)


def test_torn_final_record_is_still_tolerated(tmp_path):
    journal = tmp_path / "backup-daily.log"
    journal.write_bytes(_valid_receipt_line() + b'{"snapshot": "/x/2026-10-20.db.gz", "upl')
    assert backup._read_backup_receipts(journal) == [json.loads(_valid_receipt_line())]


def test_missing_journal_still_means_nothing_pending(local_lane):  # noqa: F811
    assert not local_lane["log_path"].exists()
    assert backup.run_backup(**local_lane, phase="upload")["upload_state"] == "nothing_pending"
