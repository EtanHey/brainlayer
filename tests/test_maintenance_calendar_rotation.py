"""Journal rotation preserves attempt boundaries and reader snapshots."""

import json

import pytest

from brainlayer import maintenance_calendar as calendar


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def seed(directory):
    events = []
    for night in range(4):
        events.extend(
            [
                dict(event="runner", kind="nightly", scheduled_date=f"2026-11-{night + 1:02}", steps=["index"]),
                dict(event="finished", step="index", status="ok", padding="x" * 200),
                dict(event="summary", complete=True),
            ]
        )
    path = directory / "runs.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return path


def test_rotation_keeps_complete_nights_and_open_reader(tmp_path, monkeypatch):
    path = seed(tmp_path)
    monkeypatch.setattr(calendar, "RUNS_ROTATE_BYTES", 1200, raising=False)
    with path.open("rb") as reader:
        original = reader.read()
        calendar.record(tmp_path, event="transcript-recovery", kind="recovery", scheduled_date="2026-11-04")
        reader.seek(0)
        assert reader.read().startswith(original)
    assert (tmp_path / "runs.jsonl.1").exists()
    current = rows(path)
    assert current[0]["event"] == "runner"
    assert current[-1]["event"] == "transcript-recovery"
    assert current[-4]["scheduled_date"] == "2026-11-04"
    assert path.stat().st_size < 1200
    archived = rows(tmp_path / "runs.jsonl.1")
    assert archived[-1]["event"] == "summary"
    assert archived + current[:-1] == [json.loads(line) for line in original.splitlines()]


@pytest.mark.parametrize("boundary", range(1, 8))
def test_rotation_crash_keeps_current_night(tmp_path, monkeypatch, boundary):
    path = seed(tmp_path)
    monkeypatch.setattr(calendar, "RUNS_ROTATE_BYTES", 1200, raising=False)
    calls = 0
    real_fsync, real_replace = calendar.os.fsync, calendar.os.replace

    def interrupt(fn, *args):
        nonlocal calls
        result = fn(*args)
        calls += 1
        if calls == boundary:
            raise OSError("injected crash")
        return result

    monkeypatch.setattr(calendar.os, "fsync", lambda *args: interrupt(real_fsync, *args))
    monkeypatch.setattr(calendar.os, "replace", lambda *args: interrupt(real_replace, *args))
    try:
        calendar.record(tmp_path, event="finished", step="newest", status="ok")
    except OSError:
        pass
    current = rows(path)
    newest = next(i for i, row in enumerate(current) if row.get("scheduled_date") == "2026-11-04")
    assert [row["event"] for row in current[newest:]] == ["runner", "finished", "summary", "finished"]
    assert current[-1]["step"] == "newest"


def test_rotation_holds_stable_writer_lock(tmp_path, monkeypatch):
    import fcntl

    path = seed(tmp_path)
    monkeypatch.setattr(calendar, "RUNS_ROTATE_BYTES", 1200)
    real_replace = calendar.os.replace
    publications = []

    def replace(source, destination):
        with (tmp_path / "runs.lock").open("a+b") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A reader opening at either publication point still has a complete night.
        assert rows(path)[0]["event"] == "runner"
        publications.append(destination.name)
        return real_replace(source, destination)

    monkeypatch.setattr(calendar.os, "replace", replace)
    calendar.record(tmp_path, event="summary", complete=True)
    assert publications == ["runs.jsonl.1", "runs.jsonl"]


def test_oversized_single_attempt_is_not_split(tmp_path, monkeypatch):
    monkeypatch.setattr(calendar, "RUNS_ROTATE_BYTES", 100)
    calendar.record(tmp_path, event="runner", kind="nightly")
    calendar.record(tmp_path, event="finished", padding="x" * 200)
    assert [row["event"] for row in rows(tmp_path / "runs.jsonl")] == ["runner", "finished"]
    assert not (tmp_path / "runs.jsonl.1").exists()


def test_production_threshold_rotates_growing_journal(tmp_path):
    for night in range(4):
        calendar.record(tmp_path, event="runner", kind="nightly", scheduled_date=f"2026-11-{night + 1:02}")
        calendar.record(tmp_path, event="finished", padding="x" * 600_000)
        calendar.record(tmp_path, event="summary", complete=True)
    path = tmp_path / "runs.jsonl"
    assert path.stat().st_size < calendar.RUNS_ROTATE_BYTES
    assert (tmp_path / "runs.jsonl.1").stat().st_size > 0
    assert rows(path)[0]["event"] == "runner"
    assert [row["event"] for row in rows(path)[-3:]] == ["runner", "finished", "summary"]
