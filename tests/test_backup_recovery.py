import gzip
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as daily
from brainlayer import backup_recovery as recovery


class Drive:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.queries = []

    def files(self):
        return self

    def list(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(execute=lambda: next(self.pages))


def candidate(**changes):
    return {
        "id": "file",
        "name": "2026-10-09.db.gz",
        "size": "42",
        "parents": ["folder"],
        "trashed": False,
        "md5Checksum": "a" * 32,
        "appProperties": {daily.DRIVE_MACHINE_PROPERTY: "test-machine"},
        **changes,
    }


def find(service):
    return recovery.find_verified_drive_match(
        service,
        folder_id="folder",
        name="2026-10-09.db.gz",
        size=42,
        md5="a" * 32,
        machine_id="test-machine",
    )


def test_candidate_matching_reads_all_pages_and_preserves_incomplete_objects():
    service = Drive(
        [
            {"files": [candidate(size="3", md5Checksum=None)], "nextPageToken": "next"},
            {"files": [candidate()]},
        ]
    )
    assert find(service)["id"] == "file"
    assert service.queries[1]["pageToken"] == "next"


@pytest.mark.parametrize(
    "item", [candidate(parents=["other"]), candidate(trashed=True), candidate(appProperties={}), candidate(size=None)]
)
def test_unknown_candidate_is_rejected(item):
    with pytest.raises(RuntimeError, match="unknown"):
        find(Drive([{"files": [item]}]))


def test_multiple_complete_matches_fail_instead_of_picking_one():
    with pytest.raises(RuntimeError, match="ambiguous"):
        find(Drive([{"files": [candidate(), candidate(id="second")]}]))
    assert find(Drive([{"files": [candidate(md5Checksum="b" * 32)]}])) is None


def test_folder_resolution_never_creates_missing_or_ambiguous_folders():
    assert (
        recovery.resolve_existing_folder(Drive([{"files": [{"id": "folder", "name": "archive"}]}]), ["archive"])
        == "folder"
    )
    for items in ([], [{"id": "one", "name": "archive"}, {"id": "two", "name": "archive"}]):
        with pytest.raises(RuntimeError, match="ambiguous"):
            recovery.resolve_existing_folder(Drive([{"files": items}]), ["archive"])


def source_pair(tmp_path, monkeypatch):
    monkeypatch.setenv(daily.BACKUP_LOG_PROVENANCE_ENV, "pytest")
    raw = tmp_path / "2026-10-09.db"
    with sqlite3.connect(raw) as db:
        db.execute("CREATE TABLE chunks (id TEXT)")
        db.execute("INSERT INTO chunks VALUES ('one')")
    compressed = tmp_path / "2026-10-09.db.gz"
    compressed.write_bytes(gzip.compress(raw.read_bytes()))
    receipt = {
        "attempted_at": "2026-10-09T00:17:07Z",
        "snapshot": str(compressed),
        "machine_id": "test-machine",
        "drive_folder": "archive",
        "bytes": compressed.stat().st_size,
        "sentinel_snapshot_chunks": 1,
        "uploaded": False,
        "verified": False,
        "backup_log_provenance": "pytest",
    }
    log = tmp_path / "backup.log"
    log.write_text(json.dumps(receipt) + "\n")
    return raw, compressed, receipt, log


def read_source(tmp_path, raw, log):
    return recovery.read_source_receipt(
        log,
        line_number=1,
        snapshot_date="2026-10-09",
        staging_dir=tmp_path,
        machine_id="test-machine",
        folder_parts=["archive"],
        expected_raw_bytes=raw.stat().st_size,
    )


def test_receipt_provenance_preserves_original_time_and_line_hash(tmp_path, monkeypatch):
    raw, compressed, receipt, log = source_pair(tmp_path, monkeypatch)
    original = log.read_bytes()
    result, facts = read_source(tmp_path, raw, log)
    assert result == receipt
    assert facts["source_attempted_at"] == receipt["attempted_at"]
    assert facts["source_receipt_sha256"] == hashlib.sha256(original).hexdigest()
    assert log.read_bytes() == original
    assert facts["bytes"] == compressed.stat().st_size


@pytest.mark.parametrize(
    "changes",
    [
        {"machine_id": "other"},
        {"snapshot": "/elsewhere/2026-10-09.db.gz"},
        {"bytes": True},
        {"sentinel_snapshot_chunks": 0},
        {"uploaded": True},
        {"attempted_at": "2026-10-09T00:17:07"},
        {"backup_log_provenance": "real"},
    ],
)
def test_receipt_rejects_unknown_or_mismatched_provenance(tmp_path, monkeypatch, changes):
    raw, _, receipt, log = source_pair(tmp_path, monkeypatch)
    log.write_text(json.dumps({**receipt, **changes}) + "\n")
    with pytest.raises(ValueError):
        read_source(tmp_path, raw, log)


def test_remote_verification_requires_checksum_parent_and_exact_identity():
    class Service:
        def files(self):
            return self

        def get(self, **kwargs):
            return SimpleNamespace(execute=lambda: self.metadata)

    service = Service()
    for changes in ({}, {"md5Checksum": None}, {"parents": ["wrong"]}, {"id": "wrong"}):
        service.metadata = candidate(**changes)
        kwargs = dict(
            file_id="file",
            expected_name="2026-10-09.db.gz",
            expected_size=42,
            expected_machine_id="test-machine",
            expected_md5="a" * 32,
            expected_parent="folder",
        )
        if changes:
            with pytest.raises(RuntimeError):
                daily.verify_drive_upload(service, **kwargs)
        else:
            assert daily.verify_drive_upload(service, **kwargs) == candidate()
