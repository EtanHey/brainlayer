"""S22: CPU-only startup, backlog pause/resume, and installed budget evidence."""

import importlib
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

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
    library = Mock()
    library.setpriority.return_value = 0
    monkeypatch.setattr(hotlane.sys, "platform", "darwin")
    monkeypatch.setattr(hotlane.ctypes, "CDLL", lambda *_a, **_kw: library)
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
    monkeypatch.setattr(hotlane, "run", lambda **_: events.append("embedder"))
    monkeypatch.setattr(sys, "argv", ["hotlane", "--db", str(tmp_path / "private.db")])
    hotlane.main()
    assert events == ["threads", "embedder"]
    assert library.mock_calls == [call.setpriority(os.PRIO_PROCESS, 0, 20)]
    receipt = json.loads((tmp_path / "hotlane-budget.json").read_text())
    assert receipt["nice"] == 20 and receipt["threads"] == threads


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

    def add_hot(i):
        store = VectorStore(db)
        store.conn.execute(
            "INSERT INTO chunks (id,content,metadata,source_file,source) VALUES (?,?,'{}','brainbar-store','mcp')",
            (f"hot{i}", f"hot{i}"),
        )
        store.close()

    def after_cycle(_):
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
        model_factory=lambda: SimpleNamespace(embed_query=lambda text: encoded.append(text) or [0.25] * 1024),
        queue_depth_fn=lambda _: int(marker.exists() or load[0] > 2),
        high_priority_queue_depth_fn=lambda _: 1,
        time_fn=lambda: 100.0,
        sleep_fn=after_cycle,
        max_cycles=6,
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


def test_ratchet_collects_full_ps_window_and_fails_build_row(tmp_path, monkeypatch):
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
    monkeypatch.setattr(ratchet, "backlog", lambda _: 5)
    report = tmp_path / "cpu.json"
    report.write_text(json.dumps(ratchet.measure(123, tmp_path / "private.db")))
    status, value = ratchet.read_report(report)
    assert status == ci_ratchet_table.RED and "70.0%" in value
    assert (
        ci_ratchet_table.row_hotlane_cpu(SimpleNamespace(hotlane_cpu_report=report), {}).status == ci_ratchet_table.RED
    )
    monkeypatch.setattr(ratchet, "measure", lambda *_: json.loads(report.read_text()))
    output = tmp_path / "cli.json"
    monkeypatch.setattr(sys, "argv", ["ratchet", "--pid", "123", "--db", "private.db", "--out", str(output)])
    assert ratchet.main() == 1
    monkeypatch.setenv("BRAINLAYER_HOTLANE_CPU_BUDGET_PERCENT", "80")
    assert ratchet.read_report(report)[0] == ci_ratchet_table.GREEN
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
