"""Receipt reconciliation for local JSONL archives made before #1776 wrote receipts."""

import hashlib
import inspect
import json
import os
import time

import pytest

from brainlayer import jsonl_backup as b

NOW = 2000000000
DAY = 86400
OLD_BYTES = b"old archive bytes"


class _FakeDrive:
    """Answers files().list().execute() page by page and records each request.

    A page that is an exception is raised when requested, like a listing that fails mid-pagination.
    """

    def __init__(self, items, pages=None):
        self.pages = pages if pages is not None else [items]
        self.requests = []

    def files(self):
        return self

    def list(self, **kwargs):
        self.requests.append(kwargs)
        return self

    def execute(self):
        index = int(self.requests[-1].get("pageToken") or 0)
        page = self.pages[index]
        if isinstance(page, Exception):
            raise page
        response = {"files": page}
        if index + 1 < len(self.pages):
            response["nextPageToken"] = str(index + 1)
        return response


def _remote(**overrides):
    item = {
        "id": "drive-old",
        "name": "claude-jsonl-2026-01-01.tar.gz",
        "size": str(len(OLD_BYTES)),
        "md5Checksum": hashlib.md5(OLD_BYTES).hexdigest(),
        "trashed": False,
    }
    item.update(overrides)
    return item


def _scratch(tmp_path, monkeypatch, items, state=None, pages=None):
    monkeypatch.delenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION", raising=False)
    monkeypatch.delenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION_DAYS", raising=False)
    staging = tmp_path / "jsonl-backups"
    staging.mkdir()
    old = staging / "claude-jsonl-2026-01-01.tar.gz"
    old.write_bytes(OLD_BYTES)
    os.utime(old, (0, 0))
    newest = staging / "claude-jsonl-2026-02-01.tar.gz"
    newest.write_bytes(b"newest")
    os.utime(newest, (NOW - DAY, NOW - DAY))
    state_path = tmp_path / "state.json"
    # A legacy file entry claims archive-backed coverage, so run_backup lists Drive.
    state = state if state is not None else {"files": {"legacy.jsonl": {"archive_id": "legacy-id"}}}
    state_path.write_text(json.dumps(state))
    drive = _FakeDrive(items, pages)
    monkeypatch.setattr(b.backup_daily, "get_drive_credentials", lambda: None)
    monkeypatch.setattr(b.backup_daily, "build_drive_service", lambda: drive)
    monkeypatch.setattr(b.backup_daily, "ensure_drive_folder_chain", lambda *a: "folder")
    source = tmp_path / "source"
    source.mkdir()
    args = dict(
        source_roots=[source],
        staging_dir=staging,
        state_path=state_path,
        log_path=tmp_path / "log.jsonl",
        queue_dir=tmp_path / "queue",
    )
    return old, newest, state_path, drive, args


def _receipts(state_path):
    return json.loads(state_path.read_text()).get("drive_archives", {})


def test_matching_drive_object_yields_receipt_and_deletes_nothing_that_run(tmp_path, monkeypatch):
    old, newest, state_path, drive, args = _scratch(tmp_path, monkeypatch, [_remote()])
    result = b.run_backup(now=NOW, **args)

    assert result["status"] == "no-op"
    assert "files(id,name,md5Checksum,size,trashed)" in drive.requests[0]["fields"]
    assert result["local_retention_reconciled"] == [old.name]
    assert result["local_retention_deleted"] == []
    assert old.exists() and newest.exists()
    assert _receipts(state_path)[old.name] == {
        "name": old.name,
        "size": len(OLD_BYTES),
        "sha256": hashlib.sha256(OLD_BYTES).hexdigest(),
        "archive_md5": hashlib.md5(OLD_BYTES).hexdigest(),
        "verified": True,
        "verified_at": NOW,
        "archive_id": "drive-old",
        "source": "reconciled",
    }
    assert "local_retention_reconciled" in (tmp_path / "log.jsonl").read_text()


def test_reconciled_receipt_retires_archive_only_after_fourteen_days(tmp_path, monkeypatch):
    old, newest, state_path, _drive, args = _scratch(tmp_path, monkeypatch, [_remote()])
    b.run_backup(now=NOW, **args)

    assert b.run_backup(now=NOW + 14 * DAY - 1, **args)["local_retention_deleted"] == []
    assert old.exists()
    assert b.run_backup(now=NOW + 14 * DAY, **args)["local_retention_deleted"] == [old.name]
    assert not old.exists() and newest.exists()


@pytest.mark.parametrize(
    "items",
    [
        pytest.param([_remote(md5Checksum=hashlib.md5(b"other").hexdigest())], id="different-md5"),
        pytest.param([_remote(md5Checksum=None)], id="no-md5"),
        pytest.param([_remote(size=str(len(OLD_BYTES) + 1))], id="different-size"),
        pytest.param([_remote(size=None)], id="no-size"),
        pytest.param([_remote(trashed=True)], id="trashed"),
        pytest.param([_remote(trashed=None)], id="trashed-unknown"),
        pytest.param([], id="missing"),
        pytest.param([_remote(name="claude-jsonl-2026-01-02.tar.gz")], id="other-name"),
        pytest.param([_remote(), _remote(id="drive-dup")], id="ambiguous-duplicate-name"),
    ],
)
def test_non_matching_drive_object_yields_no_receipt(tmp_path, monkeypatch, items):
    old, _newest, state_path, _drive, args = _scratch(tmp_path, monkeypatch, items)
    result = b.run_backup(now=NOW, **args)

    assert result["local_retention_reconciled"] == []
    assert old.name not in _receipts(state_path)
    assert old.exists()


@pytest.mark.parametrize(
    "existing",
    [
        {"archive_id": "legacy-id", "verified": False},
        {"archive_id": "drive-old", "verified": True, "verified_at": NOW - 1},
    ],
)
def test_existing_receipt_is_never_overwritten(tmp_path, monkeypatch, existing):
    state = {
        "files": {"legacy.jsonl": {"archive_id": "legacy-id"}},
        "drive_archives": {"claude-jsonl-2026-01-01.tar.gz": existing},
    }
    old, _newest, state_path, _drive, args = _scratch(tmp_path, monkeypatch, [_remote()], state=state)
    result = b.run_backup(now=NOW, **args)

    assert result["local_retention_reconciled"] == []
    assert _receipts(state_path)[old.name] == existing
    assert old.exists()


def test_verified_at_is_never_earlier_than_the_wall_clock(tmp_path, monkeypatch):
    old, _newest, state_path, _drive, args = _scratch(tmp_path, monkeypatch, [_remote()])
    before = time.time()
    b.run_backup(now=0, **args)

    assert _receipts(state_path)[old.name]["verified_at"] >= before


def test_no_reconciliation_while_a_drive_prune_revocation_is_pending(tmp_path, monkeypatch):
    state = {"files": {"legacy.jsonl": {"archive_id": "legacy-id"}}, "local_retention_revoke_pending": True}
    old, _newest, state_path, _drive, args = _scratch(tmp_path, monkeypatch, [_remote()], state=state)
    result = b.run_backup(now=NOW, **args)

    assert result["local_retention_reconciled"] == []
    assert old.name not in _receipts(state_path)


def test_match_on_a_later_listing_page_yields_receipt(tmp_path, monkeypatch):
    other = _remote(id="drive-other", name="claude-jsonl-2025-12-01.tar.gz")
    old, _newest, state_path, drive, args = _scratch(tmp_path, monkeypatch, None, pages=[[other], [_remote()]])
    result = b.run_backup(now=NOW, **args)

    assert [request.get("pageToken") for request in drive.requests] == [None, "1"]
    assert result["local_retention_reconciled"] == [old.name]
    assert _receipts(state_path)[old.name]["archive_id"] == "drive-old"


def test_duplicate_name_on_a_later_listing_page_yields_no_receipt(tmp_path, monkeypatch):
    pages = [[_remote()], [_remote(id="drive-dup")]]
    old, _newest, state_path, _drive, args = _scratch(tmp_path, monkeypatch, None, pages=pages)

    assert b.run_backup(now=NOW, **args)["local_retention_reconciled"] == []
    assert old.name not in _receipts(state_path)


def test_listing_that_fails_mid_pagination_never_reconciles(tmp_path, monkeypatch):
    """Page 1 holds a unique exact match; page 2 raises. A partial inventory writes nothing."""
    pages = [[_remote()], OSError("page 2 failed")]
    old, newest, state_path, _drive, args = _scratch(tmp_path, monkeypatch, None, pages=pages)
    result = b.run_backup(now=NOW, **args)

    assert "Drive inventory unavailable: page 2 failed" in result["local_retention_error"]
    assert result["local_retention_reconciled"] == []
    assert result["local_retention_deleted"] == []
    assert old.name not in _receipts(state_path)
    assert old.exists() and newest.exists()


def test_backdated_reconcile_mutation_deletes_in_the_writing_run(tmp_path, monkeypatch):
    """The same-run test bites: a receipt written 15 days in the past retires the archive at once."""
    source = inspect.getsource(b.reconcile_drive_receipts)
    honest = "verified_at = max(now, time.time())"
    assert honest in source
    namespace = dict(b.__dict__)
    exec(compile(source.replace(honest, "verified_at = now - 15 * 86400"), "<backdated>", "exec"), namespace)
    monkeypatch.setattr(b, "reconcile_drive_receipts", namespace["reconcile_drive_receipts"])
    old, _newest, _state_path, _drive, args = _scratch(tmp_path, monkeypatch, [_remote()])

    assert b.run_backup(now=NOW, **args)["local_retention_deleted"] == [old.name]
