"""Queue portability simulations; these do not establish native Windows behavior."""

import builtins
import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest

from brainlayer import queue_io


def test_queue_module_import_does_not_require_posix_fcntl(monkeypatch):
    real_import = builtins.__import__

    def without_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ModuleNotFoundError("No module named 'fcntl'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_fcntl)
    spec = importlib.util.spec_from_file_location("queue_io_without_fcntl", queue_io.__file__)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_windows_without_directory_handles_warns_and_syncs_file_before_rename(tmp_path, monkeypatch):
    calls = []
    real_fsync, real_replace = os.fsync, Path.replace
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delattr(os, "O_DIRECTORY")

    def sync(fd):
        assert stat.S_ISREG(os.fstat(fd).st_mode), "Unsupported directory handles must never be opened"
        calls.append("file sync")
        real_fsync(fd)

    def replace(path, target):
        assert calls == ["file sync"]
        calls.append("rename")
        return real_replace(path, target)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(Path, "replace", replace)
    with pytest.warns(RuntimeWarning, match="directory-entry durability"):
        path = queue_io.enqueue_jsonl({"content": "platform simulation"}, source="watcher", queue_dir=tmp_path)
    assert path.exists()
    assert calls == ["file sync", "rename"]


def test_missing_directory_support_on_posix_is_not_silently_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delattr(os, "O_DIRECTORY")
    with pytest.raises((AttributeError, OSError)):
        queue_io._sync_directory(tmp_path)


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_opened_directory_sync_failure_is_fatal_and_handle_is_closed(tmp_path, monkeypatch, platform):
    real_close = os.close
    closed = []
    monkeypatch.setattr(sys, "platform", platform)

    def fail(_fd):
        raise OSError("opened handle sync failure")

    def close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(os, "fsync", fail)
    monkeypatch.setattr(os, "close", close)
    with pytest.raises(OSError, match="opened handle sync failure"):
        queue_io._sync_directory(tmp_path)
    assert len(closed) == 1
