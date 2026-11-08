"""S22: CPU-only startup, backlog pause/resume, and installed budget evidence."""

import importlib
import json
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import ci_ratchet_table
from scripts import hotlane_cpu_ratchet as ratchet


@pytest.fixture
def hotlane():
    sys.modules.pop("scripts.hotlane_brainbar_daemon", None)
    return importlib.import_module("scripts.hotlane_brainbar_daemon")


@pytest.mark.parametrize(
    "chip,override,threads", [("Apple M1 Pro", None, 1), ("Apple M4 Max", None, 2), ("Apple M4", "3", 3)]
)
def test_startup_cpu_only_policy_and_threads_precede_embedder(hotlane, tmp_path, monkeypatch, chip, override, threads):
    setpriority = Mock()
    monkeypatch.setattr(hotlane.sys, "platform", "darwin")
    monkeypatch.setattr(hotlane.os, "setpriority", setpriority)
    monkeypatch.setattr(hotlane.os, "getpriority", lambda *_: 20)
    monkeypatch.setattr(hotlane, "_sysctl", lambda name: chip if "brand" in name else "10")
    if override:
        monkeypatch.setenv("BRAINLAYER_HOTLANE_THREADS", override)
    else:
        monkeypatch.delenv("BRAINLAYER_HOTLANE_THREADS", raising=False)
    events = []

    def set_threads(n):
        assert {os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")} == {str(n)}
        events.append("threads")

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(set_num_threads=set_threads, get_num_threads=lambda: threads)
    )
    samplers = []

    def run(**kwargs):
        samplers.append(kwargs["cpu_sampler"])
        events.append("embedder")

    monkeypatch.setattr(hotlane, "run", run)
    monkeypatch.setattr(sys, "argv", ["hotlane", "--db", str(tmp_path / "private.db")])
    hotlane.main()
    assert events == ["threads", "embedder"]
    setpriority.assert_called_once_with(os.PRIO_PROCESS, 0, 20)
    receipt = json.loads((tmp_path / "hotlane-budget.json").read_text())
    assert receipt["nice"] == 20 and receipt["threads"] == threads
    assert (tmp_path / "hotlane-cpu.json").is_file()
    assert not samplers[0].worker.is_alive()


def test_backlog_drains_without_duplicate_encoding_across_pause_and_load(hotlane, tmp_path, monkeypatch, caplog):
    from brainlayer.vector_store import VectorStore

    db = tmp_path / "private.db"
    store = VectorStore(db)
    for i in range(3):
        store.conn.execute(
            "INSERT INTO chunks (id,content,metadata,source_file,source) VALUES (?,?,'{}', 'fixture','claude_code')",
            (f"old{i}", f"old{i}"),
        )
    store.close()
    encoded, counts = [], []
    marker = tmp_path / "hotlane.pause"
    monkeypatch.setenv("BRAINLAYER_HOTLANE_BACKLOG_MAX_LOAD", "2")
    load = [0.0]
    monkeypatch.setattr(hotlane.os, "getloadavg", lambda: (load[0], 0, 0))
    monkeypatch.setattr("brainlayer.search_repo.clear_hybrid_search_cache", lambda _: None)
    caplog.set_level("INFO", logger=hotlane.LOGGER.name)
    from brainlayer.hotlane_budget import HotlaneCpuSampler

    monkeypatch.setattr(sys, "platform", "darwin")
    clock, cpu_clock = [0.0], [0.0]
    sampler = HotlaneCpuSampler(db, clock=lambda: clock[0], cpu_clock=lambda: cpu_clock[0])

    def add_hot(i):
        store = VectorStore(db)
        store.conn.execute(
            "INSERT INTO chunks (id,content,metadata,source_file,source) VALUES (?,?,'{}','brainbar-store','mcp')",
            (f"hot{i}", f"hot{i}"),
        )
        store.close()

    def after_cycle(_):
        clock[0] += 120
        cpu_clock[0] += 6
        counts.append(ratchet.backlog(db))
        i = len(counts)
        if i == 1:
            marker.touch()
        if i == 3:
            marker.unlink()
            load[0] = 3.0
        if i == 4:
            load[0] = 0.0
        if i < 6:
            add_hot(i)

    def embed(text):
        encoded.append(text)
        clock[0] += 10
        cpu_clock[0] += 0.5
        return [0.25] * 1024

    add_hot(0)
    hotlane.run(
        db_path=db,
        interval=0.25,
        recent_limit=5,
        backlog_interval=0,
        backlog_batch=1,
        enrich_interval=0,
        enrich_limit=0,
        enrich_since_hours=0,
        model_factory=lambda: SimpleNamespace(embed_query=embed),
        queue_depth_fn=lambda _: 0,
        high_priority_queue_depth_fn=lambda _: 0,
        time_fn=lambda: 100.0,
        sleep_fn=after_cycle,
        max_cycles=6,
        cpu_sampler=sampler,
    )
    assert counts == [2, 2, 2, 2, 1, 0]
    assert sorted(encoded) == sorted([f"old{i}" for i in range(3)] + [f"hot{i}" for i in range(6)])
    store = VectorStore(db, readonly=True)
    assert store.conn.execute("SELECT count(*), count(DISTINCT chunk_id) FROM chunk_vectors").fetchone() == (9, 9)
    store.close()
    transitions = [r.message for r in caplog.records if r.message.startswith("backlog embedding")]
    assert transitions == [
        "backlog embedding yielding: pause marker",
        "backlog embedding yielding: high load",
        "backlog embedding resumed",
    ]
    payload = json.loads((tmp_path / "hotlane-cpu.json").read_text())
    assert payload["window_seconds"] >= 600
    assert payload["backlog_wall_seconds"] == 40 and payload["backlog_cpu_seconds"] == 2
    assert payload["backlog_encoded"] > 0
    monkeypatch.setattr(ratchet, "ps_sample", lambda _: (("start", "installed"), 30))
    probe = SimpleNamespace(hotlane_cpu_report=None, db_path=db, os_name="Darwin")
    assert ci_ratchet_table.row_hotlane_cpu(probe, {}).status == ci_ratchet_table.GREEN
    marker.touch()
    assert ci_ratchet_table.row_hotlane_cpu(probe, {}).status == ci_ratchet_table.NA


def test_ratchet_ps_audit_cannot_certify_an_active_backlog_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(ratchet.sys, "platform", "darwin")
    ticks = iter([0, 600])
    samples = iter(range(61))
    monkeypatch.setattr(ratchet.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(ratchet.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        ratchet,
        "ps_sample",
        lambda _: (("start", "/opt/homebrew/Cellar/brainlayer/1/bin/hotlane_brainbar_daemon.py"), next(samples) * 7),
    )
    pending = iter([5] * 60 + [4])
    monkeypatch.setattr(ratchet, "backlog", lambda _: next(pending))
    report = tmp_path / "cpu.json"
    report.write_text(json.dumps(ratchet.measure(123, tmp_path / "private.db")))
    status, value = ratchet.read_report(report)
    assert status == ci_ratchet_table.NA and "70.0%" in value
    assert "lacks active backlog CPU/wall timing" in value
    assert (
        ci_ratchet_table.row_hotlane_cpu(SimpleNamespace(hotlane_cpu_report=report), {}).status == ci_ratchet_table.NA
    )
    monkeypatch.setattr(ratchet, "measure", lambda *_: json.loads(report.read_text()))
    output = tmp_path / "cli.json"
    monkeypatch.setattr(sys, "argv", ["ratchet", "--pid", "123", "--db", "private.db", "--out", str(output)])
    assert ratchet.main() == 0
    monkeypatch.setenv("BRAINLAYER_HOTLANE_CPU_BUDGET_PERCENT", "80")
    assert ratchet.read_report(report)[0] == ci_ratchet_table.NA
    payload = json.loads(report.read_text())
    payload["window_seconds"] = 599
    report.write_text(json.dumps(payload))
    assert ratchet.read_report(report)[0] == ci_ratchet_table.RED


def test_ratchet_off_mac_and_bad_report_are_honest(tmp_path, monkeypatch):
    monkeypatch.setattr(ratchet.sys, "platform", "linux")
    report = tmp_path / "cpu.json"
    report.write_text(json.dumps(ratchet.measure(1, tmp_path / "private.db")))
    assert ratchet.read_report(report) == (
        ci_ratchet_table.NA,
        "n/a — requires an installed Mac; this host is not Darwin",
    )
    assert ratchet.read_report(None)[0] == ci_ratchet_table.NA
    report.write_text("{}")
    assert ratchet.read_report(report)[0] == ci_ratchet_table.RED


@pytest.mark.parametrize("cpu", [5, 70])
def test_ratchet_does_not_pass_a_yielded_backlog(tmp_path, cpu):
    report = tmp_path / "paused.json"
    report.write_text(
        json.dumps(
            dict(
                status="measured",
                platform="darwin",
                pid=123,
                command="/opt/homebrew/opt/brainlayer/hotlane_brainbar_daemon.py",
                measured_at=datetime.now(timezone.utc).isoformat(),
                window_seconds=600,
                cpu_percent=cpu,
                backlog=[500] * 61,
            )
        )
    )
    status, value = ratchet.read_report(report)
    assert status == ci_ratchet_table.NA
    assert "no observed backlog drain" in value


@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("queue_pressure", [False, True])
@pytest.mark.parametrize("max_wait", [300, 60])
def test_high_load_minimum_duty_never_overrides_pause_or_queue_backpressure(
    hotlane, tmp_path, monkeypatch, paused, queue_pressure, max_wait
):
    monkeypatch.setenv("BRAINLAYER_HOTLANE_BACKLOG_MAX_LOAD", "2")
    monkeypatch.delenv("BRAINLAYER_HOTLANE_BACKLOG_MAX_WAIT_SECONDS", raising=False)
    if max_wait != 300:
        monkeypatch.setenv("BRAINLAYER_HOTLANE_BACKLOG_MAX_WAIT_SECONDS", str(max_wait))
    monkeypatch.setattr(hotlane.os, "getloadavg", lambda: (25, 0, 0))
    if paused:
        (tmp_path / "hotlane.pause").touch()
    calls = []
    hotlane.run(
        db_path=tmp_path / "private.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=7,
        backlog_batch=4,
        enrich_interval=0,
        enrich_limit=0,
        enrich_since_hours=0,
        model_factory=lambda: SimpleNamespace(embed_query=lambda _: [0.25] * 1024),
        vector_store_cls=lambda _: SimpleNamespace(close=lambda: None),
        cycle_fn=lambda **kw: calls.append(kw) or hotlane.CycleResult(),
        queue_depth_fn=lambda _: int(queue_pressure),
        high_priority_queue_depth_fn=lambda _: 1,
        time_fn=iter([0, 0, max_wait - 1, max_wait, max_wait + 1, max_wait * 2]).__next__,
        sleep_fn=lambda _: None,
        max_cycles=5,
    )
    assert [c["backlog_batch"] for c in calls if c["backlog_batch"]] == ([] if paused else [4, 4])
    if queue_pressure:
        assert all(c["recent_limit"] == 0 for c in calls)


def test_daemon_sampler_rolls_active_work_without_diluting_it_during_yields(tmp_path, monkeypatch):
    from brainlayer.hotlane_budget import HotlaneCpuSampler

    monkeypatch.setattr(sys, "platform", "darwin")
    clock, cpu = [0.0], [0.0]
    db = tmp_path / "private.db"
    sampler = HotlaneCpuSampler(db, clock=lambda: clock[0], cpu_clock=lambda: cpu[0])
    assert "warming" in ratchet.read_report(sampler.path)[1]
    clock[0], cpu[0] = 600, 300
    sampler.observe("running", 2)
    assert ratchet.read_report(sampler.path)[0] == ci_ratchet_table.GREEN
    monkeypatch.setenv("BRAINLAYER_HOTLANE_CPU_BUDGET_PERCENT", "40")
    assert ratchet.read_report(sampler.path)[0] == ci_ratchet_table.RED
    monkeypatch.setenv("BRAINLAYER_HOTLANE_CPU_BUDGET_PERCENT", "60")
    for state in ("pause marker", "high load", "load unavailable", "store queue", "error"):
        sampler.observe(state)
        # Instantaneous state cannot change the measured active-cycle ratio.
        assert ratchet.read_report(sampler.path)[0] == ci_ratchet_table.GREEN
    clock[0], cpu[0] = 1200, 600
    sampler.observe("waiting")
    assert "backlog ran 0s, not enough to measure" in ratchet.read_report(sampler.path)[1]
    assert len(sampler.samples) == 2


def test_ps_parser_checks_installed_identity(monkeypatch):
    monkeypatch.setattr(
        ratchet.subprocess,
        "check_output",
        lambda *a, **kw: (
            "Mon Nov 2 12:00:00 2026 10:03.50 /opt/homebrew/Cellar/brainlayer/1/bin/python hotlane_brainbar_daemon.py"
        ),
    )
    assert ratchet.ps_sample(123)[1] == 603.5
    monkeypatch.setattr(ratchet.subprocess, "check_output", lambda *a, **kw: "unrelated process")
    with pytest.raises(ValueError, match="installed"):
        ratchet.ps_sample(123)


def test_cpu_receipt_keeps_last_batch_time_across_load_pause(tmp_path):
    from brainlayer.hotlane_budget import HotlaneCpuSampler

    tick = [0.0]
    sampler = HotlaneCpuSampler(tmp_path / "fixture.db", clock=lambda: tick[0], cpu_clock=lambda: tick[0])
    tick[0] = 20
    sampler.observe("running", backlog_encoded=16, cycle_start=(10, 10))
    last = json.loads(sampler.path.read_text())["backlog_last_batch_at"]
    assert datetime.fromisoformat(last).tzinfo is not None
    tick[0] = 700
    sampler.observe("high load")
    receipt = json.loads(sampler.path.read_text())
    assert receipt["backlog_encoded"] == 0
    assert receipt["backlog_last_batch_at"] == last
