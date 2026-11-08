"""Isolation and cleanup contracts for the copy-only writer rehearsal."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "rehearse_1661", Path(__file__).resolve().parents[1] / "scripts/rehearse_1661.py"
)
rehearsal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rehearsal)


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "account"
    monkeypatch.setattr(rehearsal.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(home)))
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.mark.parametrize("shape", ["direct", "symlink", "relative", "home_override"])
def test_refuse_production(tmp_path, home, monkeypatch, shape):
    prod = home / ".local/share/brainlayer"
    prod.mkdir(parents=True)
    db = prod / "brainlayer.db"
    if shape == "symlink":
        link = tmp_path / "link"
        link.symlink_to(prod, target_is_directory=True)
        db = link / db.name
    elif shape == "relative":
        monkeypatch.chdir(tmp_path)
        db = Path(os.path.relpath(db, tmp_path))
    elif shape == "home_override":
        monkeypatch.setenv("HOME", str(tmp_path / "decoy"))
    with pytest.raises(ValueError, match="production or temporary"):
        rehearsal.safe(db)
    assert list(prod.iterdir()) == []


@pytest.mark.parametrize("root", ["/tmp", "/private/tmp", "/var/folders"])
def test_refuse_temporary(root, home):
    with pytest.raises(ValueError, match="temporary"):
        rehearsal.safe(Path(root) / "rehearsal.db")


def test_refuse_tmpdir(tmp_path, home, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "temporary"))
    with pytest.raises(ValueError, match="temporary"):
        rehearsal.safe(tmp_path / "temporary/copy.db")


@pytest.mark.parametrize("inside", ["copy", "production"])
def test_refuse_source_dir(tmp_path, home, inside):
    db = tmp_path / "copy/db"
    db.parent.mkdir()
    db.touch()
    source = db.parent if inside == "copy" else home / ".local/share/brainlayer/source"
    with pytest.raises(ValueError):
        rehearsal.rehearse(db, source)
    assert not source.exists() or source == db.parent


@pytest.mark.parametrize("fail", [False, True])
def test_real_flush_is_isolated_and_cleaned(tmp_path, home, monkeypatch, capsys, fail):
    # Fail before a default can touch the real home or system temporary directory.
    import brainlayer.watcher_bridge as bridge
    from brainlayer import writer_telemetry
    from brainlayer.vector_store import VectorStore

    db = tmp_path / "copy/db"
    db.parent.mkdir()
    # Schema initialization also owns a writer pidfile.
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(tmp_path / "setup"))
    VectorStore(db).close()
    monkeypatch.delenv("BRAINLAYER_WRITER_PIDFILE_DIR")
    source = tmp_path / "source"
    observed = []
    closed = []
    heartbeat_files = []
    previous_heartbeat = os.environ.get("BRAINLAYER_WRITER_HEARTBEAT_DIR")
    real_factory = bridge.create_flush_callback
    real_append = writer_telemetry.append_event
    real_heartbeat = writer_telemetry._flush_heartbeat

    def check(path):
        path = Path(path)
        assert path.is_relative_to(db.parent)
        assert path.parent.name.startswith("rehearse-1661-")
        assert path.parent.stat().st_mode & 0o777 == 0o700
        observed.append(path)

    def append(event):
        check(writer_telemetry.telemetry_path())
        return real_append(event)

    def heartbeat(path):
        check(path)
        result = real_heartbeat(path)
        if Path(path).is_file():
            heartbeat_files.append(Path(path))
        return result

    def factory(*args, **kwargs):
        for key in (
            "BRAINLAYER_WRITER_TELEMETRY_PATH",
            "BRAINLAYER_WRITER_HEARTBEAT_DIR",
            "BRAINLAYER_WRITER_PIDFILE_DIR",
            "BRAINLAYER_QUEUE_DIR",
        ):
            assert Path(os.environ[key]).is_relative_to(db.parent)
        callback = real_factory(*args, **kwargs)
        # This harness branch predates #1661. Distinct stems let isolation and
        # success cleanup run against its real writer without requiring that fix.
        for index, path in enumerate(sorted(source.rglob("*.jsonl"))):
            path.rename(path.with_name(f"session{index}-fixture.jsonl"))

        def flush(entries):
            for index, entry in enumerate(entries):
                entry["_source_file"] = str(Path(entry["_source_file"]).with_name(f"session{index}-fixture.jsonl"))
            result = callback(entries)
            scratch = Path(os.environ["BRAINLAYER_WRITER_HEARTBEAT_DIR"])
            assert (scratch / "writer-telemetry.jsonl").is_file()
            if fail:
                raise RuntimeError("injected after write")
            return result

        def close():
            callback.close()
            closed.append(True)

        flush.close = close
        return flush

    monkeypatch.setattr(writer_telemetry, "append_event", append)
    monkeypatch.setattr(writer_telemetry, "_flush_heartbeat", heartbeat)
    monkeypatch.setattr(bridge, "create_flush_callback", factory)
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            rehearsal.rehearse(db, source)
    else:
        rehearsal.rehearse(db, source)
        report = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert report["result"] == "PASS"
        assert report["counts"] == [2, 2]
    assert observed and heartbeat_files and closed == [True]
    assert not source.exists()
    assert not list(db.parent.glob("rehearse-1661-*"))
    assert not (home / ".local/share/brainlayer").exists()
    assert os.environ.get("BRAINLAYER_WRITER_HEARTBEAT_DIR") == previous_heartbeat
