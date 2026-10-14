"""Actual direct-publication SIGKILLs retain bytes and recover only their staging link."""

import os
import signal
import subprocess
import sys

import pytest

PROBE = r"""
import os, signal, sys
from pathlib import Path
from brainlayer.migration_custody import migration_lock, write_new_file, write_launch_agent
home, path, launcher = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
original = os.link
def crash(*args, **kwargs):
    original(*args, **kwargs)
    os.kill(os.getpid(), signal.SIGKILL)
os.link = crash
with migration_lock(home):
    (write_launch_agent if launcher else write_new_file)(path, b"complete bytes")
"""


@pytest.mark.parametrize("launcher", [False, True])
def test_direct_publisher_link_crash_recovers_live_destination(tmp_path, launcher):
    from brainlayer.migration_custody import migration_lock
    from brainlayer.migration_journal import recover_staging

    home = tmp_path / "home"
    agents = home / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    control = home / ".local/share/brainlayer/supervisor"
    control.mkdir(mode=0o700, parents=True)
    archive = agents / ".archive"
    archive.mkdir(mode=0o700)
    path = agents / "com.brainlayer.daemon.plist" if launcher else control / "migration.json"
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(home), str(path), str(int(launcher))],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == -signal.SIGKILL, result.stderr
    (stage,) = path.parent.glob(".migration-*")
    assert stage.stat().st_ino == path.stat().st_ino and path.stat().st_nlink == 2
    assert path.read_bytes() == b"complete bytes"
    with migration_lock(home):
        recover_staging(path, archive)
    assert not stage.exists() and path.stat().st_nlink == 1
    assert path.read_bytes() == b"complete bytes" and not list(archive.iterdir())


def test_recovery_refuses_a_third_link_even_for_known_destination(tmp_path):
    from brainlayer.migration_journal import recover_staging

    archive = tmp_path / "archive"
    archive.mkdir(mode=0o700)
    path = tmp_path / "journal"
    path.write_bytes(b"personal bytes")
    path.chmod(0o600)
    stage = tmp_path / (".migration-" + "a" * 32)
    personal = tmp_path / "personal"
    os.link(path, stage)
    os.link(path, personal)
    with pytest.raises(ValueError, match="custody"):
        recover_staging(path, archive)
    assert stage.exists() and personal.read_bytes() == path.read_bytes() == b"personal bytes"
