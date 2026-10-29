"""Disposable-source regressions for rotation during transcript bundling."""

import errno
import hashlib
import json
import tarfile
import time
from pathlib import Path

import pytest

from brainlayer import jsonl_backup
from tests.test_jsonl_backup import _covered_state_kwargs, _install_drive, _write_jsonl


def test_vanish_before_tarinfo_excludes_digest_and_size(tmp_path):
    root = tmp_path / "sessions"
    doomed = _write_jsonl(root / "doomed.jsonl", mtime=time.time() - 3600)
    keeper = _write_jsonl(root / "keeper.jsonl", mtime=time.time() - 3600)
    candidates = jsonl_backup._discover_jsonl_candidates([root])
    doomed.unlink()

    archive, digests, sizes = jsonl_backup.create_jsonl_bundle_with_digests(
        candidates, tmp_path / "staging", date_stamp="2026-10-29"
    )

    assert digests == {keeper.as_posix(): hashlib.sha256(keeper.read_bytes()).hexdigest()}
    assert sizes == {keeper.as_posix(): keeper.stat().st_size}
    with tarfile.open(archive, "r:gz") as bundle:
        assert bundle.getnames() == ["source-0/keeper.jsonl"]


def test_unlink_after_open_finishes_from_handle(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    source = _write_jsonl(root / "session.jsonl", mtime=time.time() - 3600)
    content = source.read_bytes()
    candidates = jsonl_backup._discover_jsonl_candidates([root])
    real_open = Path.open

    def open_then_unlink(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        if path == source:
            source.unlink()
        return handle

    monkeypatch.setattr(Path, "open", open_then_unlink)
    archive, digests, sizes = jsonl_backup.create_jsonl_bundle_with_digests(
        candidates, tmp_path / "staging", date_stamp="2026-10-29"
    )
    assert digests == {source.as_posix(): hashlib.sha256(content).hexdigest()}
    assert sizes == {source.as_posix(): len(content)}
    with tarfile.open(archive, "r:gz") as bundle:
        assert bundle.extractfile("source-0/session.jsonl").read() == content


@pytest.mark.parametrize("phase", ["open", "tarinfo", "read", "mid-read"])
def test_one_of_many_vanishes_without_coverage_and_retries(tmp_path, monkeypatch, phase):
    now = time.time()
    root = tmp_path / "sessions"
    doomed = _write_jsonl(root / "a-doomed.jsonl", line="x" * 65536, mtime=now - 3600)
    keeper = _write_jsonl(root / "z-keeper.jsonl", mtime=now - 3600)
    uploads, surviving = [], []
    _install_drive(monkeypatch, jsonl_backup, uploads, surviving)
    monkeypatch.setattr(jsonl_backup, "_forever_enabled", lambda: False)
    kwargs = _covered_state_kwargs(tmp_path, root, now)
    real_open = Path.open
    real_info = tarfile.TarFile.gettarinfo

    class VanishingReader:
        def __init__(self, handle):
            self.handle = handle
            self.reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size=-1):
            self.reads += 1
            if phase == "read" or self.reads > 1:
                doomed.unlink()
                raise OSError(errno.ENOENT, "source rotated")
            return self.handle.read(size)

    def vanish_on_open(path, *args, **kwargs):
        if path == doomed:
            if phase == "open":
                doomed.unlink()
            handle = real_open(path, *args, **kwargs)
            return VanishingReader(handle) if "read" in phase else handle
        return real_open(path, *args, **kwargs)

    def vanish_on_tarinfo(tar, name=None, arcname=None, fileobj=None):
        if phase == "tarinfo" and arcname == "source-0/a-doomed.jsonl":
            doomed.unlink()
            raise FileNotFoundError(errno.ENOENT, "source rotated")
        return real_info(tar, name=name, arcname=arcname, fileobj=fileobj)

    with monkeypatch.context() as race:
        race.setattr(Path, "open", vanish_on_open)
        race.setattr(tarfile.TarFile, "gettarinfo", vanish_on_tarinfo)
        result = jsonl_backup.run_backup(date_stamp="2026-10-29", **kwargs)

    assert result["status"] == "uploaded"
    assert result["verified"] is True
    assert result["vanished_source_count"] == 1
    assert result["bundled_file_count"] == result["archive_listing_count"] == 1
    with tarfile.open(result["archive"], "r:gz") as bundle:
        assert bundle.getnames() == ["source-0/z-keeper.jsonl"]
        assert bundle.extractfile("source-0/z-keeper.jsonl").read() == keeper.read_bytes()
    state = json.loads(kwargs["state_path"].read_text())
    assert set(state["files"]) == {keeper.as_posix()}
    receipt = json.loads(kwargs["log_path"].read_text())
    assert receipt["vanished_source_count"] == 1
    assert doomed.as_posix() not in json.dumps(receipt)
    assert not list(kwargs["staging_dir"].glob(".*.tmp"))

    _write_jsonl(doomed, mtime=now - 3600)
    retried = jsonl_backup.run_backup(date_stamp="2026-10-30", **kwargs)
    assert retried["bundled_file_count"] == 1
    assert retried["already_covered_files"] == 1
    assert set(json.loads(kwargs["state_path"].read_text())["files"]) == {
        keeper.as_posix(),
        doomed.as_posix(),
    }


def test_all_vanish_returns_noop_without_upload_or_state(tmp_path, monkeypatch):
    now = time.time()
    root = tmp_path / "sessions"
    source = _write_jsonl(root / "session.jsonl", mtime=now - 3600)
    kwargs = _covered_state_kwargs(tmp_path, root, now)
    real_bundle = jsonl_backup.create_jsonl_bundle_with_digests

    def vanish_before_bundle(*args, **kwargs):
        source.unlink()
        return real_bundle(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("all-vanished run must not authenticate, upload, or advance state")

    monkeypatch.setattr(jsonl_backup, "create_jsonl_bundle_with_digests", vanish_before_bundle)
    for name in ("get_drive_credentials", "build_drive_service", "upload_file_to_drive_raw"):
        monkeypatch.setattr(jsonl_backup.backup_daily, name, forbidden)
    monkeypatch.setattr(jsonl_backup, "_atomic_write_json", forbidden)
    result = jsonl_backup.run_backup(date_stamp="2026-10-29", **kwargs)

    assert result["status"] == "no-op"
    assert result["verified"] is True
    assert result["uploaded"] is False
    assert result["vanished_source_count"] == 1
    assert json.loads(kwargs["log_path"].read_text())["vanished_source_count"] == 1
    assert not kwargs["state_path"].exists()
    assert {path.name for path in kwargs["staging_dir"].iterdir()} == {".jsonl-backup.lock"}


@pytest.mark.parametrize("error", [errno.EACCES, errno.EIO])
@pytest.mark.parametrize("phase", ["open", "read"])
def test_other_source_errors_still_fail(tmp_path, monkeypatch, error, phase):
    root = tmp_path / "sessions"
    source = _write_jsonl(root / "session.jsonl", mtime=time.time() - 3600)
    candidates = jsonl_backup._discover_jsonl_candidates([root])
    real_open = Path.open
    real_read = jsonl_backup._HashingReader.read

    def fail_open(path, *args, **kwargs):
        if path == source and phase == "open":
            raise OSError(error, "source failed")
        return real_open(path, *args, **kwargs)

    def fail_read(reader, size=-1):
        if phase == "read":
            raise OSError(error, "source failed")
        return real_read(reader, size)

    monkeypatch.setattr(Path, "open", fail_open)
    monkeypatch.setattr(jsonl_backup._HashingReader, "read", fail_read)
    staging = tmp_path / "staging"
    with pytest.raises(OSError) as failure:
        jsonl_backup.create_jsonl_bundle_with_digests(candidates, staging, date_stamp="2026-10-29")
    assert failure.value.errno == error
    assert list(staging.iterdir()) == []
