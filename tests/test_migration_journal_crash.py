"""SIGKILL every update syscall boundary in an owned scratch directory."""

import json
import signal
import subprocess
import sys

PROBE = r"""
import json, os, signal, sys
from pathlib import Path
from brainlayer.migration_custody import migration_lock, write_new_file
from brainlayer.migration_journal import archive_replace
path, archive, cut = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
events = []
def boundary(name):
    events.append(name)
    if cut == len(events):
        os.kill(os.getpid(), signal.SIGKILL)
def wrap(name, original):
    def call(*args, **kwargs):
        boundary("before:" + name)
        result = original(*args, **kwargs)
        boundary("after:" + name)
        return result
    return call
for name in ("open", "close", "fchmod", "fsync", "link", "unlink", "rename", "replace"):
    setattr(os, name, wrap(name, getattr(os, name)))
fdopen = os.fdopen
class Stream:
    def __init__(self, stream): self.stream = stream
    def __getattr__(self, name): return getattr(self.stream, name)
    def __enter__(self): return self
    def __exit__(self, *args): return wrap("stream_close", self.stream.__exit__)(*args)
    def fileno(self): return self.stream.fileno()
    def write(self, content): return wrap("write", self.stream.write)(content)
    def flush(self): return wrap("flush", self.stream.flush)()
os.fdopen = lambda *args, **kwargs: Stream(fdopen(*args, **kwargs))
with migration_lock(path.parents[4]):
    archive_replace(path, b"old", b"new", archive, write_new_file)
print(json.dumps(events))
"""


def run_probe(root, cut):
    root.mkdir(mode=0o700)
    agents = root / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    archive = agents / ".archive"
    archive.mkdir(mode=0o700)
    control = root / ".local/share/brainlayer/supervisor"
    control.mkdir(mode=0o700, parents=True)
    source = control / "journal"
    source.write_bytes(b"old")
    source.chmod(0o600)
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(source), str(archive), str(cut)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return source, archive, result


def test_live_journal_is_old_or_new_at_every_syscall_boundary(tmp_path):
    from brainlayer import migration_journal
    from brainlayer.migration_custody import migration_lock

    _, _, baseline = run_probe(tmp_path / "baseline", 0)
    assert baseline.returncode == 0, baseline.stderr
    events = json.loads(baseline.stdout)
    assert len(events) >= 20
    failures, leftovers = [], []
    for cut, event in enumerate(events, 1):
        source, archive, result = run_probe(tmp_path / str(cut), cut)
        assert result.returncode == -signal.SIGKILL, (cut, event, result.stderr)
        value = source.read_bytes() if source.exists() else None
        if value not in (b"old", b"new"):
            failures.append((cut, event, value))
            continue
        official = [p for p in archive.iterdir() if not p.name.startswith(".migration-")]
        assert len(official) <= 1 and all(p.read_bytes() == b"old" for p in official), (cut, event)
        if recover := getattr(migration_journal, "recover_staging", None):
            with migration_lock(source.parents[4]):
                recover(source, archive)
            assert source.read_bytes() == value
        copies = list(archive.iterdir())
        if (
            list(source.parent.glob(".migration-*"))
            or list(archive.glob(".migration-*"))
            or len(copies) > 1
            or any(p.read_bytes() != b"old" or p.stat().st_nlink != 1 for p in copies)
        ):
            leftovers.append((cut, event))
    assert not failures, failures
    assert not leftovers, leftovers
