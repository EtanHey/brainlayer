import plistlib
from types import SimpleNamespace

import pytest

from brainlayer import brainbar_identity as identity


@pytest.mark.parametrize("stale", [False, True])
def test_running_identity_binds_mapped_vnode_not_only_path(tmp_path, monkeypatch, stale):
    app = tmp_path / "BrainBar.app"
    binaries = app / "Contents/MacOS"
    binaries.mkdir(parents=True)
    (app / "Contents/Info.plist").write_bytes(
        plistlib.dumps({"CFBundleShortVersionString": "1.5.99", "GitCommit": "abc"})
    )
    for _, binary in identity.JOBS:
        (binaries / binary).touch()

    def loaded(label, timeout):
        binary = dict(identity.JOBS)[label]
        return SimpleNamespace(returncode=0, stdout=f"program = {binaries / binary}\npid = 123\n")

    monkeypatch.setattr(identity, "loaded", loaded)
    count = []

    def run(args, timeout):
        binary = identity.JOBS[len(count)][1]
        count.append(binary)
        path = binaries / binary
        st = path.stat()
        return SimpleNamespace(returncode=0, stdout=f"p123\nftxt\nD{st.st_dev:x}\ni{st.st_ino + int(stale)}\nn{path}\n")

    monkeypatch.setattr(identity, "run", run)
    assert identity.running_bundle_matches(app) is (not stale)
    st = (binaries / "BrainBar").stat()
    assert identity.app_identity(app) == {
        "version": "1.5.99",
        "git_commit": "abc",
        "executable_dev": st.st_dev,
        "executable_ino": st.st_ino,
    }


def test_identity_refuses_pid_change_during_inspection(tmp_path, monkeypatch):
    app = tmp_path / "BrainBar.app"
    (app / "Contents/MacOS").mkdir(parents=True)
    binary = app / "Contents/MacOS/BrainBarDaemon"
    binary.touch()
    count = []

    def loaded(label, timeout):
        count.append(1)
        return SimpleNamespace(returncode=0, stdout=f"program = {binary}\npid = {len(count)}\n")

    monkeypatch.setattr(identity, "loaded", loaded)
    st = binary.stat()
    monkeypatch.setattr(
        identity, "run", lambda *a: SimpleNamespace(returncode=0, stdout=f"D{st.st_dev:x}\ni{st.st_ino}\nn{binary}\n")
    )
    assert not identity.running_bundle_matches(app)


@pytest.mark.parametrize("changed", ["atime", "mtime", "size", "inode", "device"])
def test_running_identity_ignores_only_access_time(tmp_path, monkeypatch, changed):
    app = tmp_path / "BrainBar.app"
    binaries = app / "Contents/MacOS"
    binaries.mkdir(parents=True)
    for _, name in identity.JOBS:
        (binaries / name).touch()
    original_stat = type(binaries).stat
    count = {}

    def stat(path, *args, **kwargs):
        st = original_stat(path, *args, **kwargs)
        if path.parent != binaries:
            return st
        count[path] = count.get(path, 0) + 1
        # First read is the identity snapshot; second is its final validation.
        if count[path] == 2:
            fields = dict(st_dev=st.st_dev, st_ino=st.st_ino, st_size=st.st_size, st_mtime_ns=st.st_mtime_ns)
            field = {"mtime": "st_mtime_ns", "size": "st_size", "inode": "st_ino", "device": "st_dev"}.get(changed)
            if field:
                fields[field] += 1
            return SimpleNamespace(**fields, st_atime_ns=st.st_atime_ns + 100)
        return st

    def loaded(label, timeout):
        binary = binaries / dict(identity.JOBS)[label]
        return SimpleNamespace(returncode=0, stdout=f"program = {binary}\npid = 123\n")

    def run(args, timeout):
        binary = binaries / identity.JOBS[len(count)][1]
        st = original_stat(binary)
        return SimpleNamespace(returncode=0, stdout=f"D{st.st_dev:x}\ni{st.st_ino}\nn{binary}\n")

    monkeypatch.setattr(type(binaries), "stat", stat)
    monkeypatch.setattr(identity, "loaded", loaded)
    monkeypatch.setattr(identity, "run", run)
    assert identity.running_bundle_matches(app) is (changed == "atime")
