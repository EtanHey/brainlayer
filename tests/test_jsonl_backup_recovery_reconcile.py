import hashlib
import json
import os

import pytest

from brainlayer import jsonl_backup as b

NOW = 2_000_000_000
DAY = 86400


def remote(object_id, content=b"local", **overrides):
    return (
        dict(
            id=object_id,
            size=str(len(content)),
            md5Checksum=hashlib.md5(content).hexdigest(),
            trashed=False,
        )
        | overrides
    )


def test_recovery_receipt_then_retention(tmp_path, monkeypatch):
    staging = tmp_path / "jsonl-backups" / "release-current"
    recovery = staging.parent / "recovery-2026-10-23"
    staging.mkdir(parents=True)
    recovery.mkdir()
    archive = recovery / "claude-jsonl-2026-10-24.tar.gz"
    archive.write_bytes(b"local")
    os.utime(archive, (0, 0))
    newest = staging / "claude-jsonl-2026-11-17.tar.gz"
    newest.write_bytes(b"new")
    state = {}
    monkeypatch.setattr(b.time, "time", lambda: NOW)
    assert b.reconcile_drive_receipts(
        staging, state=state, now=NOW, drive_objects={archive.name: [remote("exact")]}
    ) == [archive.name]
    receipt = state["drive_archives"][archive.name]
    assert receipt["archive_id"] == "exact"
    assert receipt["verified_at"] == NOW
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION", "1")
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION_DAYS", "14")
    args = dict(state_path=state_path, log_path=tmp_path / "log", surviving_archives={"exact": receipt["archive_md5"]})
    assert b.prune_local_archives(staging, now=NOW + 14 * DAY - 1, **args) == []
    assert b.prune_local_archives(staging, now=NOW + 15 * DAY, **args) == [archive.name]
    assert not archive.exists()
    assert newest.exists()


@pytest.mark.parametrize(
    "objects,expected",
    [
        ([remote("exact"), remote("different", b"other")], "exact"),
        ([remote("different", b"other"), remote("exact")], "exact"),
        ([remote("other-a", b"other"), remote("other-b", b"other")], None),
        ([remote("exact-a"), remote("exact-b")], None),
        ([remote("trashed", trashed=True), remote("exact")], "exact"),
        ([remote("bad-size", size="invalid"), remote("exact")], "exact"),
        ([remote("wrong-size", size="99"), remote("exact")], "exact"),
        ([], None),
    ],
    ids=[
        "exact-different",
        "different-exact",
        "no-identical",
        "ambiguous",
        "trashed",
        "bad-size",
        "wrong-size",
        "absent",
    ],
)
def test_unique_identical_object_only(tmp_path, monkeypatch, objects, expected):
    archive = tmp_path / "claude-jsonl-2026-10-16.tar.gz"
    archive.write_bytes(b"local")
    os.utime(archive, (0, 0))
    (tmp_path / "claude-jsonl-2026-11-17.tar.gz").write_bytes(b"new")
    state = {"files": {"unchanged": True}}
    monkeypatch.setattr(b.time, "time", lambda: NOW + 1)
    result = b.reconcile_drive_receipts(tmp_path, state=state, now=NOW, drive_objects={archive.name: objects})
    assert result == ([archive.name] if expected else [])
    assert state["files"] == {"unchanged": True}
    if expected:
        assert state["drive_archives"][archive.name]["archive_id"] == expected
        assert state["drive_archives"][archive.name]["verified_at"] == NOW + 1
        original = json.loads(json.dumps(state))
        b.reconcile_drive_receipts(
            tmp_path, state=state, now=NOW + DAY, drive_objects={archive.name: [remote("replacement")]}
        )
        assert state == original
    else:
        assert state["drive_archives"] == {}
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps(state))
        monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION", "1")
        monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION_DAYS", "14")
        for age_days in (15, 365, 36500):
            assert (
                b.prune_local_archives(
                    tmp_path,
                    state_path=state_path,
                    log_path=tmp_path / "log",
                    now=NOW + age_days * DAY,
                    surviving_archives={obj["id"]: obj["md5Checksum"] for obj in objects},
                )
                == []
            )
            assert archive.read_bytes() == b"local"


def test_recovery_scan_excludes_symlinks_and_non_directories(tmp_path):
    staging = tmp_path / "jsonl-backups"
    staging.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "claude-jsonl-2026-10-24.tar.gz").write_bytes(b"local")
    (staging / "recovery-link").symlink_to(outside, target_is_directory=True)
    (staging / "recovery-file").write_bytes(b"not a directory")
    assert b._local_archive_paths(staging) == []


def test_no_size_match_never_hashes_the_archive(tmp_path, monkeypatch):
    archive = tmp_path / "claude-jsonl-2026-09-14.tar.gz"
    archive.write_bytes(b"local")
    (tmp_path / "claude-jsonl-2026-11-17.tar.gz").write_bytes(b"new")
    state = {}

    def hashed(path):
        raise AssertionError(f"hashed {path.name} although no Drive copy has its size")

    monkeypatch.setattr(b, "_md5_file", hashed)
    objects = [remote("longer", b"local-but-longer"), remote("trashed", trashed=True), remote("bad", size="x")]
    assert b.reconcile_drive_receipts(tmp_path, state=state, now=NOW, drive_objects={archive.name: objects}) == []
    assert state["drive_archives"] == {}
