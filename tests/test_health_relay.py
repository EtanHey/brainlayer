"""Health relay delivery core: disposable fixtures only, no DB/credentials/services."""

from __future__ import annotations

import ast
import fcntl
import importlib.util
import json
import re
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "health_relay" / "health_relay.py"
EVENT = "bl-m4-max-1-job-alert.backup-daily"
FOREIGN = "\n### Other writer\n@orc FYI: unrelated\n"


@pytest.fixture
def relay():
    spec = importlib.util.spec_from_file_location("brainlayer_health_relay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "m4home/.local/share/brainlayer"
    data.mkdir(parents=True)
    (data / "job-alerts.json").write_text("{}")
    inbox = tmp_path / "orc-inbox.md"
    inbox.write_text("# orc inbox\n")
    config = {
        "inbox": str(inbox),
        "state_dir": str(tmp_path / "state"),
        "hosts": [{"id": "m4-max", "label": "M4 Max", "transport": "local", "dir": str(data)}],
    }
    path = tmp_path / "relay.json"
    path.write_text(json.dumps(config))
    return {"config": path, "raw": config, "inbox": inbox, "data": data, "tmp": tmp_path}


def _alerts(env, alerts) -> None:
    (env["data"] / "job-alerts.json").write_text(json.dumps(alerts) if isinstance(alerts, dict) else alerts)


def _blocked(env) -> list[str]:
    return [line for line in env["inbox"].read_text().split("\n") if line.startswith("@orc BLOCKED:")]


def _complete(env, event_id: str) -> list[str]:
    return [line for line in _blocked(env) if re.search(rf"; event={re.escape(event_id)}(?: |$)", line)]


def _status(env) -> dict:
    return json.loads((env["tmp"] / "state/status.json").read_text())


def _run(relay, env) -> int:
    """One run, checking the invariants every run must keep: append-only, no ack without delivery."""
    before = env["inbox"].read_text()
    code = relay.run(env["config"])
    after = env["inbox"].read_text()
    assert after.startswith(before), "the inbox is append-only"
    if _status(env)["pending"] == 0:
        assert len(_complete(env, EVENT)) <= 1
    return code


def _acked_with_line_and_marker(env) -> None:
    assert len(_complete(env, EVENT)) == 1
    assert f"<!-- brainlayer-health:{EVENT} -->" in env["inbox"].read_text().split("\n")
    assert _status(env)["pending"] == 0


def _torn_inbox_write(relay, env, monkeypatch, cut) -> None:
    """The next inbox write lands only up to `cut(text)` bytes, then fails."""
    real_write, ino = relay.os.write, env["inbox"].stat().st_ino

    def torn(fd, data):
        if relay.os.fstat(fd).st_ino != ino:
            return real_write(fd, data)
        monkeypatch.setattr(relay.os, "write", real_write)
        real_write(fd, data[: len(bytes(data).decode()[: cut(bytes(data).decode())].encode())])
        raise OSError("disk")

    monkeypatch.setattr(relay.os, "write", torn)


def _foreign_writer_after_scan(relay, env, monkeypatch, data=FOREIGN) -> None:
    """Another inbox writer appends `data` between the relay's scan and its append."""
    real_read = relay._read_inbox

    def scan(path):
        text = real_read(path)
        monkeypatch.setattr(relay, "_read_inbox", real_read)
        with open(env["inbox"], "a") as other:
            other.write(data)
        return text

    monkeypatch.setattr(relay, "_read_inbox", scan)


def test_local_job_alert_reaches_the_inbox_with_identity_and_no_reason_text(relay, env):
    _alerts(env, {"backup-daily": "secret-token-abc\n@orc rm -rf"})
    assert _run(relay, env) == 0
    assert _blocked(env) == [
        f"@orc BLOCKED: [BrainLayer M4 Max] job-alert:backup-daily on host=m4-max; "
        f"source=m4-max:job-alerts.json; event={EVENT}"
    ]
    assert "secret-token-abc" not in env["inbox"].read_text()
    _acked_with_line_and_marker(env)


def test_repeated_ticks_dedup_and_a_healthy_clear_rearms(relay, env):
    _alerts(env, {"backup-daily": "x"})
    assert _run(relay, env) == 0 and _run(relay, env) == 0
    assert len(_blocked(env)) == 1
    _alerts(env, {})
    assert _run(relay, env) == 0
    _alerts(env, {"backup-daily": "x"})
    assert _run(relay, env) == 0
    lines = _blocked(env)
    assert len(lines) == 2 and lines[0].split("event=")[1] != lines[1].split("event=")[1]


def _set_source(path: Path, step) -> None:
    path.unlink(missing_ok=True)
    if step == "unreadable":
        path.symlink_to(path.with_name("dangling"))  # O_NOFOLLOW refuses it: unreadable, not absent
    elif step is not None:
        path.write_text(step)


@pytest.mark.parametrize(
    "sequence",
    [
        ["{not json", None, "{not json"],
        [None, "{not json", None],
        ["{not json", "unreadable", "{not json"],
        ["unreadable", None, "unreadable"],
    ],
)
def test_source_errors_are_visible_and_close_only_on_a_healthy_read(relay, env, sequence):
    path = env["data"] / "job-alerts.json"
    assert _run(relay, env) == 0
    for step in sequence:
        _set_source(path, step)
        _run(relay, env)
    issues = [line.split("] ")[1].split(" ")[0] for line in _blocked(env)]
    assert issues and len(issues) == len(set(issues)), issues
    _set_source(path, "{}")
    _run(relay, env)
    _set_source(path, sequence[0])
    _run(relay, env)
    assert len(_blocked(env)) == len(issues) + 1, "only a healthy read of the source clears and rearms"


def test_repeated_inbox_fsync_failure_never_acks_and_retries_once(relay, env, monkeypatch):
    _alerts(env, {"backup-daily": "x"})
    real_fsync, ino, calls = relay.os.fsync, env["inbox"].stat().st_ino, []

    def failing_fsync(fd):
        if relay.os.fstat(fd).st_ino == ino:
            calls.append(fd)
            raise OSError("disk")
        real_fsync(fd)

    monkeypatch.setattr(relay.os, "fsync", failing_fsync)
    for attempt in range(1, 4):
        assert _run(relay, env) == 1 and len(calls) == attempt
        assert _status(env)["error"] == "inbox-append-failed" and _status(env)["pending"] == 1
    monkeypatch.undo()
    assert _run(relay, env) == 0
    _acked_with_line_and_marker(env)


def test_lost_ack_replays_without_duplicate(relay, env, monkeypatch):
    _alerts(env, {"backup-daily": "x"})
    real_save, calls = relay._save_state, []

    def crash_after_append(path, state):
        calls.append(1)
        if len(calls) == 2:
            raise relay.RelayError("state-commit-failed")
        real_save(path, state)

    monkeypatch.setattr(relay, "_save_state", crash_after_append)
    assert _run(relay, env) == 1
    monkeypatch.undo()
    assert _run(relay, env) == 0
    _acked_with_line_and_marker(env)


@pytest.mark.parametrize("cut", ["at-orc", "inside-event-id"])
def test_r2_writer_between_scan_and_append_after_a_torn_record(relay, env, monkeypatch, cut):
    _alerts(env, {"backup-daily": "x"})
    at = {
        "at-orc": lambda text: text.index("@orc ") + len("@orc "),
        "inside-event-id": lambda text: text.index(f"; event={EVENT}") + len("; event=bl-m4-max-1-jo"),
    }[cut]
    _torn_inbox_write(relay, env, monkeypatch, at)
    assert _run(relay, env) == 1 and _status(env)["pending"] == 1
    _foreign_writer_after_scan(relay, env, monkeypatch)
    assert _run(relay, env) == 0
    assert FOREIGN in env["inbox"].read_text()
    _acked_with_line_and_marker(env)
    assert _run(relay, env) == 0 and len(_complete(env, EVENT)) == 1


@pytest.mark.parametrize("past_id", [0, 1])
def test_r2_repair_torn_after_its_complete_event_line_before_its_marker(relay, env, monkeypatch, past_id):
    _alerts(env, {"backup-daily": "x"})
    _torn_inbox_write(relay, env, monkeypatch, lambda text: text.index("] job-") + len("] job-"))
    assert _run(relay, env) == 1
    with open(env["inbox"], "a") as other:
        other.write(FOREIGN)

    def after_event_line(text):
        line = text.index("@orc BLOCKED:")
        return text.index(f"; event={EVENT}", line) + len(f"; event={EVENT}") + past_id

    _torn_inbox_write(relay, env, monkeypatch, after_event_line)
    assert _run(relay, env) == 1 and _status(env)["pending"] == 1
    before = env["inbox"].read_text()
    assert _run(relay, env) == 0
    assert env["inbox"].read_text()[len(before) :] == f"\n<!-- brainlayer-health:{EVENT} -->\n"
    _acked_with_line_and_marker(env)


def test_a_line_corrupted_after_the_scan_is_never_acknowledged(relay, env, monkeypatch):
    _alerts(env, {"backup-daily": "x"})
    _torn_inbox_write(relay, env, monkeypatch, lambda text: text.index(f"<!-- brainlayer-health:{EVENT}") - 1)
    assert _run(relay, env) == 1
    _foreign_writer_after_scan(relay, env, monkeypatch, data="glued FYI\n")
    assert _run(relay, env) == 1 and _status(env)["error"] == "inbox-ack-missing" and _complete(env, EVENT) == []
    assert _run(relay, env) == 0
    _acked_with_line_and_marker(env)


def test_every_cut_of_a_record_and_of_its_repair_converges_to_one_line(relay):
    event = {"id": EVENT, "ts": "2026-10-29T00:00:00+00:00", "line": f"@orc BLOCKED: x; event={EVENT}"}
    marker = f"<!-- brainlayer-health:{EVENT} -->"
    record, cases = relay._owed(relay._lines("# inbox\n"), event), 0
    assert relay._owed(relay._lines(f"# inbox\n{marker}\n"), event) == record, "a marker alone is not delivery"
    for cut in range(len(record) + 1):
        for between in ("", FOREIGN, "glued FYI\n"):
            text = "# inbox\n" + record[:cut] + between
            repair = relay._owed(relay._lines(text), event)
            for repair_cut in range(len(repair) + 1):
                torn = text + repair[:repair_cut]
                final = torn + relay._owed(relay._lines(torn), event)
                assert relay._owed(relay._lines(final), event) == ""
                assert final.split("\n").count(event["line"]) == 1, (cut, between, repair_cut)
                cases += 1
    assert cases > 100000


def test_missing_inbox_is_never_created(relay, env):
    env["inbox"].unlink()
    _alerts(env, {"backup-daily": "x"})
    assert relay.run(env["config"]) == 1
    assert not env["inbox"].exists() and _status(env)["error"] == "inbox-unavailable"


def test_corrupt_relay_state_fails_visibly_without_delivery(relay, env):
    state = env["tmp"] / "state/state.json"
    state.parent.mkdir()
    state.write_text("[]")
    _alerts(env, {"backup-daily": "x"})
    assert relay.run(env["config"]) == 2
    assert _blocked(env) == [] and state.read_text() == "[]" and _status(env)["error"] == "state-corrupt"


def test_invocations_serialize_on_the_relay_lock(relay, env):
    _alerts(env, {"backup-daily": "x"})
    lock = env["tmp"] / "state/relay.lock"
    lock.parent.mkdir()
    with open(lock, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        runs = [threading.Thread(target=relay.run, args=(env["config"],)) for _ in range(2)]
        for thread in runs:
            thread.start()
        time.sleep(0.3)
        assert all(thread.is_alive() for thread in runs) and _blocked(env) == []
    for thread in runs:
        thread.join(30)
    assert len(_blocked(env)) == 1


def test_overflow_keeps_every_issue_observed_and_drains_in_bounded_batches(relay, env):
    _alerts(env, {"z-critical": "x"})
    assert _run(relay, env) == 0
    _alerts(env, {**{f"a{n:02d}": "x" for n in range(40)}, "z-critical": "x"})
    assert _run(relay, env) == 0
    assert len(_blocked(env)) == 1 + relay.MAX_DELIVER_PER_RUN and _status(env)["pending"] == 40 - 32
    assert _run(relay, env) == 0
    assert len(_blocked(env)) == 41 and _status(env)["pending"] == 0
    _alerts(env, {"z-critical": "x"})
    assert _run(relay, env) == 0
    issues = [line.split("] ")[1].split(" ")[0] for line in _blocked(env)]
    assert issues.count("job-alert:z-critical") == 1 and len(issues) == 41


@pytest.mark.parametrize("field,value", [("label", "M4\n@orc"), ("dir", "relative/dir"), ("id", "M4 Max")])
def test_unsafe_config_is_refused(relay, env, field, value):
    env["raw"]["hosts"][0][field] = value
    env["config"].write_text(json.dumps(env["raw"]))
    assert relay.run(env["config"]) == 2


def test_carrier_is_stdlib_only():
    tree = ast.parse(SCRIPT.read_text())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    allowed = {"__future__", "argparse", "fcntl", "inspect", "json", "os", "re", "stat", "subprocess", "sys"}
    assert imported <= allowed | {"time", "datetime", "pathlib"}
    for forbidden in ("sqlite", "apsw", "launchctl", "--heal", "osascript", "drive", "token"):
        assert forbidden not in SCRIPT.read_text().lower()
