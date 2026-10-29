"""Health relay delivery core: disposable fixtures only, no DB/credentials/services."""

from __future__ import annotations

import ast
import fcntl
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "health_relay" / "health_relay.py"
REAL_URANDOM = os.urandom
EPOCH = "01" * 16
EVENT = f"bl-m4-max-{EPOCH}-1-job-alert.backup-daily"
FOREIGN = "\n### Other writer\n@orc FYI: unrelated\n"


@pytest.fixture
def relay(monkeypatch):
    spec = importlib.util.spec_from_file_location("brainlayer_health_relay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.os, "urandom", lambda n: b"\x01" * n)
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
    state_path = env["tmp"] / "state/state.json"
    if _status(env)["pending"] is not None:
        state = json.loads(state_path.read_text())
        pending = {event["id"] for event in state["outbox"]}
        whole, _ = relay._lines(after)
        for host_id, host in state["hosts"].items():
            for issue, episode in host.get("episodes", {}).items():
                event_id = f"bl-{host_id}-{state['epoch']}-{episode['seq']}-{issue.replace(':', '.')}"
                if event_id in pending:
                    continue
                complete = _complete(env, event_id)
                assert len(complete) == 1, "an ack requires exactly one complete alert line"
                assert complete[0] in whole and relay._marker(event_id) in whole
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
    real_read, real_deliver = relay._read_inbox, relay._deliver

    def scan(path):
        text = real_read(path)
        monkeypatch.setattr(relay, "_read_inbox", real_read)
        with open(env["inbox"], "a") as other:
            other.write(data)
        return text

    def deliver(*args):
        monkeypatch.setattr(relay, "_deliver", real_deliver)
        monkeypatch.setattr(relay, "_read_inbox", scan)
        return real_deliver(*args)

    monkeypatch.setattr(relay, "_deliver", deliver)


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
    assert not env["inbox"].exists() and _status(env)["error"] == "state-missing"
    assert not (env["tmp"] / "state/state.json").exists()


@pytest.mark.parametrize(
    "document",
    [
        [],
        {"version": 2, "hosts": {"m4-max": None}, "outbox": []},
        {"version": 2, "hosts": {}, "outbox": [{}]},
        {"version": 2, "hosts": {"m4-max": {"seq": "broken"}}, "outbox": []},
        *[{"version": v, "hosts": {}, "outbox": []} for v in (None, 1, 2.0, True)],
        *[
            {"version": 2, "hosts": {"m4-max": {field: value}}, "outbox": []}
            for field, value in (
                ("seq", True),
                ("seq", -1),
                ("seen", None),
                ("seen", [1]),
                ("episodes", []),
                ("episodes", {"job-alert:x": None}),
                ("episodes", {"job-alert:x": {"open": "yes", "seq": 1}}),
                ("episodes", {"job-alert:x": {"open": False}}),
                ("episodes", {"job-alert:x": {"open": True, "seq": True}}),
            )
        ],
        *[
            {"version": 2, "hosts": {}, "outbox": [{"id": EVENT, "ts": "now", "line": "alert", field: value}]}
            for field, value in (("id", None), ("ts", 1), ("line", []), ("line", "alert\n"))
        ],
    ],
)
def test_corrupt_relay_state_fails_visibly_without_delivery(relay, env, document):
    state = env["tmp"] / "state/state.json"
    state.parent.mkdir()
    original = json.dumps(document).encode()
    state.write_bytes(original)
    _alerts(env, {"backup-daily": "x"})
    assert relay.run(env["config"]) == 2
    assert _blocked(env) == [] and state.read_bytes() == original
    assert _status(env)["error"] == "state-corrupt" and _status(env)["pending"] is None


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
    assert imported <= allowed | {"time", "datetime", "pathlib", "selectors", "shlex"}
    for forbidden in ("sqlite", "apsw", "launchctl", "--heal", "osascript", "drive", "token"):
        assert forbidden not in SCRIPT.read_text().lower()


@pytest.mark.parametrize(
    "destination", ["state.json", "status.json", "relay.lock", ".state.json.tmp", ".status.json.tmp"]
)
@pytest.mark.parametrize("alias", ["direct", "resolved", "symlink", "hardlink"])
def test_inbox_write_destination_collision_is_refused_before_any_write(relay, env, destination, alias):
    state_dir = env["tmp"] / "state"
    state_dir.mkdir()
    target = state_dir / destination
    if alias == "direct":
        env["raw"]["inbox"] = str(target)
        env["inbox"] = target
        target.write_text("# shared inbox\n@orc FYI: preserve me\n")
    elif alias == "resolved":
        link = env["tmp"] / "state-alias"
        link.symlink_to(state_dir, target_is_directory=True)
        env["raw"]["inbox"] = str(link / destination)
        env["inbox"] = target
        target.write_text("# shared inbox\n@orc FYI: preserve me\n")
    elif alias == "symlink":
        target.symlink_to(env["inbox"])
    else:
        os.link(env["inbox"], target)
    env["config"].write_text(json.dumps(env["raw"]))
    _alerts(env, {"backup-daily": "x"})
    before = env["inbox"].read_bytes()
    names = sorted(p.name for p in state_dir.iterdir())
    code = relay.run(env["config"])
    assert (code, env["inbox"].read_bytes(), sorted(p.name for p in state_dir.iterdir())) == (2, before, names)
    with pytest.raises(relay.RelayError, match="config-invalid"):
        relay.load_config(env["config"])


@pytest.mark.parametrize("field,value", [("id", "m4-max\n"), ("label", "M4 Max\n")])
def test_final_newline_config_identifiers_are_refused_before_writes(relay, env, field, value):
    env["raw"]["hosts"][0][field] = value
    env["config"].write_text(json.dumps(env["raw"]))
    before = env["inbox"].read_bytes()
    assert relay.run(env["config"]) == 2
    assert env["inbox"].read_bytes() == before and not (env["tmp"] / "state").exists()


def test_final_newline_source_keys_do_not_starve_a_valid_alert(relay, env):
    _alerts(env, {**{f"a{n:02d}\n": "x" for n in range(32)}, "z-critical": "x"})
    for _ in range(3):
        assert _run(relay, env) == 0
        assert _status(env)["pending"] == 0
    assert len(_blocked(env)) == 2
    assert len(_complete(env, f"bl-m4-max-{EPOCH}-2-job-alert.z-critical")) == 1
    assert len(_complete(env, f"bl-m4-max-{EPOCH}-1-job-alert.invalid-key")) == 1


@pytest.mark.parametrize(
    "python",
    [
        None,
        pytest.param(
            "/usr/bin/python3",
            marks=pytest.mark.skipif(not os.access("/usr/bin/python3", os.X_OK), reason="no /usr/bin/python3"),
        ),
    ],
    ids=["in-process", "python39-isolated"],
)
def test_deep_source_is_corrupt_preserves_incidents_and_delivers_pending(relay, env, monkeypatch, python):
    _alerts(env, {"backup-daily": "x"})
    with monkeypatch.context() as patch:
        patch.setattr(relay, "_append", lambda *args: (_ for _ in ()).throw(relay.RelayError("inbox-append-failed")))
        assert _run(relay, env) == 1
    deep = "[" * 1500 + "0" + "]" * 1500
    assert len(deep.encode()) < relay.MAX_SOURCE_BYTES
    _alerts(env, deep)
    if python:
        result = subprocess.run(
            [python, "-I", str(SCRIPT), "--config", str(env["config"])], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    else:
        assert _run(relay, env) == 0
    state = json.loads((env["tmp"] / "state/state.json").read_text())
    episodes = state["hosts"]["m4-max"]["episodes"]
    assert episodes["job-alert:backup-daily"]["open"] is True
    assert episodes["source-corrupt:job-alerts"]["open"] is True
    assert not state["outbox"] and _status(env)["pending"] == 0
    _acked_with_line_and_marker(env)
    assert len(_complete(env, f"bl-m4-max-{EPOCH}-2-source-corrupt.job-alerts")) == 1


def test_run_helper_rejects_ack_without_a_complete_alert(relay, env, monkeypatch):
    _alerts(env, {"backup-daily": "x"})

    def drop_events(cfg, state, path):
        state["outbox"] = []
        relay._save_state(path, state)

    monkeypatch.setattr(relay, "_deliver", drop_events)
    with pytest.raises(AssertionError, match="an ack requires exactly one complete alert line"):
        _run(relay, env)


@pytest.mark.parametrize("bad", ["unreadable", "binary-corrupt", "oversize", "{not json"])
def test_first_bad_observation_then_absence_keeps_the_original_episode(relay, env, bad):
    source = env["data"] / "job-alerts.json"

    def bad_source():
        _set_source(source, "unreadable" if bad == "unreadable" else "x")
        if bad == "binary-corrupt":
            source.write_bytes(b"\xff")
        elif bad == "oversize":
            source.write_bytes(b"x" * (relay.MAX_SOURCE_BYTES + 1))
        elif bad == "{not json":
            source.write_text(bad)

    bad_source()
    assert _run(relay, env) == 0
    state_path = env["tmp"] / "state/state.json"
    original = json.loads(state_path.read_text())["hosts"]["m4-max"]["episodes"]
    issue, episode = next(iter(original.items()))
    source.unlink()
    assert _run(relay, env) == 0
    host = json.loads(state_path.read_text())["hosts"]["m4-max"]
    assert "job-alerts" in host["seen"]
    assert host["episodes"][issue] == episode and episode["open"] is True
    assert host["episodes"]["source-missing:job-alerts"]["open"] is True
    bad_source()
    assert _run(relay, env) == 0
    assert sum(f"] {issue} " in line for line in _blocked(env)) == 1


def test_state_loss_refuses_bootstrap_until_explicit_new_epoch(relay, env, monkeypatch):
    monkeypatch.setattr(relay.os, "urandom", REAL_URANDOM)
    _alerts(env, {"backup-daily": "x"})
    assert relay.run(env["config"]) == 0
    first = _blocked(env)[0].split("event=")[1]
    _alerts(env, {})
    assert relay.run(env["config"]) == 0
    state_path = env["tmp"] / "state/state.json"
    state_path.unlink()
    _alerts(env, {"backup-daily": "x"})
    before = env["inbox"].read_bytes()
    assert relay.run(env["config"]) == 1
    assert _status(env) == {"ts": _status(env)["ts"], "error": "state-missing", "pending": None, "hosts": {}}
    assert env["inbox"].read_bytes() == before and not state_path.exists()
    assert relay.main(["--config", str(env["config"]), "--accept-new-epoch"]) == 0
    second = _blocked(env)[1].split("event=")[1]
    assert first != second
    state = json.loads(state_path.read_text())
    assert re.fullmatch(r"[0-9a-f]{32}", state["epoch"])
    assert f"-{state['epoch']}-" in second


@pytest.mark.parametrize("suffix,refused", [("\n", True), ("", True), (" partial\n", False)])
def test_bootstrap_checks_complete_legacy_markers_including_tail(relay, env, suffix, refused):
    env["inbox"].write_text("# inbox\n<!-- brainlayer-health:bl-old-1-job-alert.x -->" + suffix)
    before = env["inbox"].read_bytes()
    assert relay.run(env["config"]) == (1 if refused else 0)
    assert env["inbox"].read_bytes() == before
    assert _status(env)["error"] == ("state-missing" if refused else None)


def test_markerless_bootstraps_generate_distinct_persisted_epochs(relay, env, monkeypatch):
    monkeypatch.setattr(relay.os, "urandom", REAL_URANDOM)
    state_path = env["tmp"] / "state/state.json"
    assert relay.run(env["config"]) == 0
    first = json.loads(state_path.read_text())["epoch"]
    assert relay.run(env["config"]) == 0
    assert json.loads(state_path.read_text())["epoch"] == first
    state_path.unlink()
    assert relay.run(env["config"]) == 0
    assert json.loads(state_path.read_text())["epoch"] != first


@pytest.mark.parametrize("alias", ["direct", "resolved", "symlink", "hardlink"])
@pytest.mark.parametrize("override", [False, True])
def test_inbox_source_alias_is_refused_before_any_write(relay, env, alias, override):
    source = env["data"] / "job-alerts.json"
    if override:
        source = env["tmp"] / "other-job-alerts.json"
        source.write_text('{"backup-daily":"x"}')
        env["raw"]["hosts"].append(
            {
                "id": "other",
                "label": "Other",
                "transport": "local",
                "dir": str(env["tmp"]),
                "sources": {"job-alerts": str(source)},
            }
        )
    if alias == "direct":
        inbox = source
    elif alias == "resolved":
        directory = env["tmp"] / "source-alias"
        directory.symlink_to(source.parent, target_is_directory=True)
        inbox = directory / source.name
    else:
        inbox = env["inbox"]
        inbox.unlink()
        if alias == "symlink":
            inbox.symlink_to(source)
        else:
            os.link(source, inbox)
    env["raw"]["inbox"] = str(inbox)
    env["config"].write_text(json.dumps(env["raw"]))
    before = source.read_bytes()
    assert relay.run(env["config"]) == 2
    assert source.read_bytes() == before and not (env["tmp"] / "state").exists()


@pytest.mark.parametrize("name", ["state.json", "status.json"])
@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_atomic_temp_links_never_truncate_their_target(relay, env, name, link):
    target = env["tmp"] / "preserve.txt"
    target.write_bytes(b"preserve these bytes")
    destination = env["tmp"] / name
    tmp = relay._atomic_temp(destination)
    if link == "symlink":
        tmp.symlink_to(target)
    else:
        os.link(target, tmp)
    relay._write_atomic(destination, {"x": 1})
    assert target.read_bytes() == b"preserve these bytes"
    assert json.loads(destination.read_text()) == {"x": 1}
    assert destination.stat().st_mode & 0o777 == 0o600 and not tmp.exists()


def test_atomic_files_are_created_private_even_under_permissive_umask(relay, env):
    destination = env["tmp"] / "private.json"
    previous = os.umask(0)
    try:
        relay._write_atomic(destination, {"private": True})
    finally:
        os.umask(previous)
    assert destination.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("status_fails", [False, True])
def test_lock_open_failure_returns_one_and_attempts_failure_status(relay, env, status_fails):
    state_dir = env["tmp"] / "state"
    state_dir.mkdir()
    (state_dir / "relay.lock").symlink_to(env["tmp"] / "missing")
    if status_fails:
        (state_dir / ".status.json.tmp").symlink_to(env["tmp"] / "missing-status")
    before = env["inbox"].read_bytes()
    assert relay.run(env["config"]) == 1
    assert env["inbox"].read_bytes() == before and not (state_dir / "state.json").exists()
    if not status_fails:
        assert _status(env)["error"] == "lock-open-failed" and _status(env)["pending"] is None


def test_version_two_migrates_pending_without_rewriting_legacy_event_ids(relay, env):
    state_path = env["tmp"] / "state/state.json"
    state_path.parent.mkdir()
    legacy_id = "bl-m4-max-1-job-alert.backup-daily"
    legacy_line = f"@orc BLOCKED: legacy alert; event={legacy_id}"
    host = {"seq": 1, "seen": ["job-alerts"], "episodes": {"job-alert:backup-daily": {"open": True, "seq": 1}}}
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "hosts": {"m4-max": host},
                "outbox": [{"id": legacy_id, "ts": "now", "line": legacy_line}],
            }
        )
    )
    _alerts(env, {"backup-daily": "x"})
    assert relay.run(env["config"]) == 0
    state = json.loads(state_path.read_text())
    assert state["version"] == 3 and state["epoch"] == EPOCH
    migrated = state["hosts"]["m4-max"]
    assert {key: migrated[key] for key in host} == host and state["outbox"] == []
    assert migrated["unreach"] == 0 and isinstance(migrated["first_seen"], float)
    assert _blocked(env) == [legacy_line]
    _alerts(env, {})
    assert relay.run(env["config"]) == 0
    _alerts(env, {"backup-daily": "x"})
    assert relay.run(env["config"]) == 0
    assert f"bl-m4-max-{EPOCH}-2-job-alert.backup-daily" in _blocked(env)[1]


@pytest.mark.parametrize("epoch", [None, "", "bad", 42, "a" * 33])
def test_version_three_requires_a_valid_epoch(relay, env, epoch):
    state_path = env["tmp"] / "state/state.json"
    state_path.parent.mkdir()
    original = json.dumps({"version": 3, "epoch": epoch, "hosts": {}, "outbox": []})
    state_path.write_text(original)
    assert relay.run(env["config"]) == 2
    assert _status(env)["error"] == "state-corrupt"
    assert state_path.read_text() == original and _blocked(env) == []


PRE_HARDENING_RELAY = "9c6910f1657e50d7b5efb86a075e885cf414dd5f"


def _clean_git_env() -> dict[str, str]:
    # An inherited GIT_DIR (a git hook runs the suite) would point `git show` at the wrong repo.
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def test_rollback_rejects_new_state_explicitly(relay, env):
    # A depth-1 CI checkout lacks the pre-hardening commit; local and pre-push runs have it.
    has_commit = subprocess.run(
        ["git", "cat-file", "-e", f"{PRE_HARDENING_RELAY}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        env=_clean_git_env(),
    )
    if has_commit.returncode != 0:
        pytest.skip(f"pre-hardening relay commit {PRE_HARDENING_RELAY[:8]} not in this checkout")
    assert relay.run(env["config"]) == 0
    state_path = env["tmp"] / "state/state.json"
    result = subprocess.run(
        ["git", "show", f"{PRE_HARDENING_RELAY}:scripts/health_relay/health_relay.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=_clean_git_env(),
    )
    old_script = env["tmp"] / "old-relay.py"
    old_script.write_text(result.stdout)
    original = state_path.read_bytes()
    result = subprocess.run(
        [sys.executable, "-I", str(old_script), "--config", str(env["config"])], capture_output=True
    )
    assert result.returncode == 2
    assert _status(env)["error"] == "state-corrupt" and _status(env)["pending"] is None
    assert state_path.read_bytes() == original and _blocked(env) == []


@pytest.mark.parametrize("fault", ["permission", "loop", "not-directory"])
def test_inaccessible_source_does_not_block_other_hosts_or_pending(relay, env, monkeypatch, fault):
    _alerts(env, {"backup-daily": "x"})
    with monkeypatch.context() as patch:
        patch.setattr(relay, "_append", lambda *args: (_ for _ in ()).throw(relay.RelayError("inbox-append-failed")))
        assert _run(relay, env) == 1
    other = env["tmp"] / "other"
    other.mkdir()
    (other / "job-alerts.json").write_text('{"restic":"x"}')
    env["raw"]["hosts"].append({"id": "m1-pro", "label": "M1 Pro", "transport": "local", "dir": str(other)})
    source = env["data"] / "job-alerts.json"
    previous = env["data"].stat().st_mode & 0o777
    if fault == "permission":
        env["data"].chmod(0)
    elif fault == "loop":
        source.unlink()
        source.symlink_to(source)
    else:
        source = source / "child.json"
        env["raw"]["hosts"][0]["sources"] = {"job-alerts": str(source)}
    env["config"].write_text(json.dumps(env["raw"]))
    try:
        if fault == "permission" and os.access(source, os.R_OK):
            pytest.skip("user can bypass chmod 000")
        assert relay.run(env["config"]) == 0
        assert _status(env)["error"] is None and _status(env)["pending"] == 0
        assert len(_complete(env, EVENT)) == 1
        state = json.loads((env["tmp"] / "state/state.json").read_text())
        episode = state["hosts"]["m4-max"]["episodes"]["source-unreadable:job-alerts"]
        assert episode["open"] is True
        assert len(_complete(env, f"bl-m4-max-{EPOCH}-{episode['seq']}-source-unreadable.job-alerts")) == 1
        assert len(_complete(env, f"bl-m1-pro-{EPOCH}-1-job-alert.restic")) == 1
    finally:
        env["data"].chmod(previous)


@pytest.mark.parametrize("name", ["state.json", "status.json", "relay.lock", ".state.json.tmp", ".status.json.tmp"])
def test_owned_destination_samefile_errors_fail_closed(relay, env, monkeypatch, name):
    real_samefile = relay.os.path.samefile

    def inaccessible(inbox, destination):
        if destination == env["tmp"] / "state" / name:
            raise PermissionError("denied")
        return real_samefile(inbox, destination)

    monkeypatch.setattr(relay.os.path, "samefile", inaccessible)
    before = env["inbox"].read_bytes()
    assert relay.run(env["config"]) == 2
    assert env["inbox"].read_bytes() == before and not (env["tmp"] / "state").exists()


@pytest.mark.parametrize("accept", [False, True])
def test_restored_older_state_refuses_recurrence_and_preserves_bytes(relay, env, monkeypatch, accept):
    monkeypatch.setattr(relay.os, "urandom", REAL_URANDOM)
    assert _run(relay, env) == 0
    path = env["tmp"] / "state/state.json"
    snapshot = path.read_bytes()
    _alerts(env, {"backup-daily": "x"})
    assert _run(relay, env) == 0
    _alerts(env, {})
    assert _run(relay, env) == 0
    path.write_bytes(snapshot)
    _alerts(env, {"backup-daily": "x"})
    before = env["inbox"].read_bytes()
    assert relay.run(env["config"], accept_new_epoch=accept) == 1
    assert _status(env)["error"] == "state-stale" and _status(env)["pending"] is None
    assert path.read_bytes() == snapshot and env["inbox"].read_bytes() == before
    path.rename(path.with_name("state-stale.json"))
    assert relay.main(["--config", str(env["config"]), "--accept-new-epoch"]) == 0
    events = [line.split("event=")[1] for line in _blocked(env)]
    assert len(events) == len(set(events)) == 2


@pytest.mark.parametrize(
    "host,epoch,seq,prefix,suffix,refused",
    [
        ("m4-max", EPOCH, 1, "", "\n", True),
        ("m4-max", EPOCH, 1, "", "", True),
        ("m4-max", EPOCH, 0, "", "\n", False),
        ("m4-max", "02" * 16, 1, "", "\n", False),
        ("m1-pro", EPOCH, 1, "", "\n", True),
        ("m4-max", EPOCH, 1, "> ", "\n", False),
        ("m4-max", EPOCH, 1, "", " partial\n", False),
    ],
)
def test_stale_state_scan_matches_only_complete_markers_for_its_epoch(
    relay, env, host, epoch, seq, prefix, suffix, refused
):
    assert _run(relay, env) == 0
    path = env["tmp"] / "state/state.json"
    before = path.read_bytes()
    with env["inbox"].open("a") as inbox:
        inbox.write(f"{prefix}<!-- brainlayer-health:bl-{host}-{epoch}-{seq}-job-alert.x -->{suffix}")
    inbox_before = env["inbox"].read_bytes()
    assert relay.run(env["config"]) == (1 if refused else 0)
    assert _status(env)["error"] == ("state-stale" if refused else None)
    assert env["inbox"].read_bytes() == inbox_before
    if refused:
        assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["state.json", "status.json"])
def test_atomic_temp_link_planted_after_unlink_is_not_opened(relay, env, monkeypatch, name):
    target = env["tmp"] / "preserve.txt"
    target.write_bytes(b"preserve these bytes")
    destination = env["tmp"] / name
    tmp = relay._atomic_temp(destination)
    real_open = relay.os.open

    def plant(path, flags, *args, **kwargs):
        if path == tmp:
            os.link(target, tmp)
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(relay.os, "open", plant)
        with pytest.raises(FileExistsError):
            relay._write_atomic(destination, {"x": 1})
    assert target.read_bytes() == b"preserve these bytes" and not destination.exists()


@pytest.mark.parametrize("fault", ["permission", "missing"])
@pytest.mark.parametrize("markers", [False, True])
def test_bootstrap_requires_a_readable_markerless_inbox(relay, env, fault, markers):
    if markers:
        env["inbox"].write_text(f"# inbox\n<!-- brainlayer-health:{EVENT} -->\n")
    before = env["inbox"].read_bytes()
    state = env["tmp"] / "state/state.json"
    parked = env["tmp"] / "parked-inbox.md"
    previous = env["inbox"].stat().st_mode & 0o777
    if fault == "permission":
        env["inbox"].chmod(0)
    else:
        env["inbox"].rename(parked)
    try:
        if fault == "permission" and os.access(env["inbox"], os.R_OK):
            pytest.skip("user can bypass chmod 000")
        assert relay.run(env["config"]) == 1
        assert not state.exists()
        assert _status(env)["error"] == "state-missing" and _status(env)["pending"] is None
        if fault == "missing":
            assert not env["inbox"].exists() and parked.read_bytes() == before
    finally:
        if fault == "permission":
            env["inbox"].chmod(previous)
        else:
            parked.rename(env["inbox"])
    assert env["inbox"].read_bytes() == before
    assert relay.run(env["config"]) == (1 if markers else 0)
    assert state.exists() is not markers
    assert env["inbox"].read_bytes() == before
    assert _status(env)["error"] == ("state-missing" if markers else None)


@pytest.mark.parametrize("fault", ["permission", "missing"])
def test_explicit_new_epoch_can_recover_with_an_unreadable_inbox(relay, env, fault):
    _alerts(env, {"backup-daily": "x"})
    previous = env["inbox"].stat().st_mode & 0o777
    if fault == "permission":
        env["inbox"].chmod(0)
    else:
        env["inbox"].unlink()
    try:
        if fault == "permission" and os.access(env["inbox"], os.R_OK):
            pytest.skip("user can bypass chmod 000")
        assert relay.run(env["config"], accept_new_epoch=True) == 1
        state = json.loads((env["tmp"] / "state/state.json").read_text())
        assert len(state["outbox"]) == 1
        assert _status(env)["error"] == ("inbox-unreadable" if fault == "permission" else "inbox-unavailable")
    finally:
        if fault == "permission":
            env["inbox"].chmod(previous)
        else:
            env["inbox"].write_text("# inbox\n")
    assert _run(relay, env) == 0
    _acked_with_line_and_marker(env)


@pytest.mark.parametrize("error", [PermissionError, NotADirectoryError, OSError])
def test_host_source_resolve_errors_do_not_abort_other_alerts(relay, env, monkeypatch, error):
    source = env["data"] / "job-alerts.json"
    _alerts(env, {"backup-daily": "x"})
    real_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == source:
            raise error("source resolution failed")
        return real_resolve(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", resolve)
        assert relay.run(env["config"]) == 0
    _acked_with_line_and_marker(env)
