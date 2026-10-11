#!/usr/bin/env python3
"""Private Python producer -> compiled production Swift consumer freshness contract.

This measures liveness interpretation, not a load budget, DB persistence, or installed UI.
Missing compiler, malformed evidence, or a missing case fails; there is no mock fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from scripts.watcher_heartbeat_contract import (
        CASES,
        ROOT,
        SOURCES,
        digest,
        process_declaration,
        store_content,
        validate_artifact,
        validate_completion,
        validate_stores,
    )
except ModuleNotFoundError:
    from watcher_heartbeat_contract import (
        CASES,
        ROOT,
        SOURCES,
        digest,
        process_declaration,
        store_content,
        validate_artifact,
        validate_completion,
        validate_stores,
    )

__all__ = [
    "CASES",
    "ROOT",
    "SOURCES",
    "digest",
    "process_declaration",
    "store_content",
    "validate_artifact",
    "validate_completion",
    "validate_stores",
    "measure",
]


def measure(directory: Path, evidence_root: Path | None = None) -> dict:
    # Isolate before importing BrainLayer: no env-file secrets, real socket, DB, model or telemetry.
    os.environ.update(
        BRAINLAYER_ENV_FILE=str(directory / "absent.env"),
        BRAINLAYER_DB=str(directory / "private.db"),
        BRAINLAYER_QUEUE_DIR=str(directory / "queue"),
        BRAINLAYER_T3_STATE_DB=str(directory / "absent-t3.db"),
        BRAINLAYER_FORBID_EMBEDDING_MODEL="1",
        BRAINLAYER_FORBID_BRAINBAR_SOCKET="1",
    )
    for key in ("AXIOM_TOKEN", "BRAINLAYER_INGEST_DENYLIST"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(ROOT / "src"))
    from brainlayer.queue_io import enqueue_store
    from brainlayer.watcher import JSONLWatcher, WatchRoot
    from brainlayer.watcher_bridge import create_flush_callback

    dashboard = ROOT / "brain-bar/Sources/BrainBar/Dashboard"
    # The reader's process enum lives in a large UI file. Compile its exact declaration,
    # rather than substituting a mock type or importing the AppKit database dashboard.
    enum_file = directory / "ProcessEvidence.swift"
    enum_file.write_text(process_declaration())
    binary = directory / "heartbeat-consumer"
    subprocess.run(
        [
            "swiftc",
            str(enum_file),
            str(dashboard / "WatcherHealthStatus.swift"),
            str(dashboard / "DashboardMetricFormatter.swift"),
            str(ROOT / "scripts/watcher_heartbeat_probe.swift"),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=180,
    )
    retained = (evidence_root or directory) / "heartbeat-evidence"
    retained.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, retained / binary.name)
    shutil.copy2(enum_file, retained / enum_file.name)
    compile_sources = {
        p: digest(ROOT / p) for p in SOURCES if p.endswith(".swift") and not p.endswith("PipelineState.swift")
    }
    compile_sources["ProcessEvidence.swift"] = digest(enum_file)
    source = directory / "transcripts"
    source.mkdir()
    transcript = source / "fixture.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {
                    "content": "Synthetic heartbeat fixture: preserve this user message during watcher polling."
                },
            }
        )
        + "\n"
    )
    health = directory / "watcher-health.json"
    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("custom", source)],
        registry_path=directory / "offsets.json",
        health_path=health,
        batch_size=1,
        on_flush=create_flush_callback(directory / "private.db", arbitrated=True),
    )
    watcher.poll_once()
    payload = json.loads(health.read_text())
    queued = list((directory / "queue").glob("watcher-*.jsonl"))
    if not queued or not all(json.loads(p.read_text())["content"] for p in queued):
        raise RuntimeError("real watcher bridge did not produce queued ingest")
    heartbeat = datetime.fromisoformat(payload["updated_at"])
    results = []

    def check(name, expected, now=None, process="running"):
        result = json.loads(
            subprocess.check_output(
                [
                    str(binary),
                    str(health),
                    (now or datetime.now(timezone.utc)).isoformat(),
                    process,
                ],
                text=True,
                timeout=10,
            )
        )
        results.append({"name": name, "expected": expected, "observed": result["state"]})
        if result["state"] != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {result}")

    check("fresh", "running")
    check("boundary_300", "running", heartbeat + timedelta(seconds=300))
    check("boundary_over_300", "degraded", heartbeat + timedelta(seconds=301))
    check("future", "running", heartbeat - timedelta(seconds=1))
    payload["updated_at"] = (heartbeat - timedelta(seconds=600)).isoformat()
    health.write_text(json.dumps(payload))
    check("stale", "degraded")
    stores = []

    def store(index):
        queue = directory / "queue"
        before = set(queue.glob("*.jsonl"))
        path = enqueue_store(content=store_content(index))
        if not isinstance(path, Path) or path.resolve().parent != queue.resolve():
            raise RuntimeError("independent store did not return a private persisted path")
        if set(queue.glob("*.jsonl")) - before != {path}:
            raise RuntimeError("independent store did not create exactly one new queue file")
        lines = path.read_text().splitlines()
        if len(lines) != 1:
            raise RuntimeError("independent store requires exactly one decoded event")
        item = {"path": path.name, "event": json.loads(lines[0])}
        stores.append(item)
        validate_stores(stores, complete=False)

    before_store = health.read_bytes()
    store(0)
    if health.read_bytes() != before_store:
        raise RuntimeError("agent store altered watcher health")
    check("unrelated_store", "degraded")
    watcher.poll_once()
    refreshed = json.loads(health.read_text())
    if refreshed["poll_count"] != payload["poll_count"] + 1:
        raise RuntimeError("private producer poll did not advance")
    check("refresh_recovery", "running")
    # Delay a real discovery stat at a barrier. This is controlled filesystem latency,
    # not a claim that this tiny fixture recreates the host's measured 42-minute delay.
    import brainlayer.watcher as producer

    original_walk = producer._iter_jsonl_files
    entered, release = threading.Event(), threading.Event()
    progress, errors = [], []

    def delayed_walk(*args, **kwargs):
        for path, read_stat in original_walk(*args, **kwargs):

            def delayed_stat(read_stat=read_stat):
                value = read_stat()
                progress.append(1)
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("private filesystem barrier timed out")
                return value

            yield path, delayed_stat

    def poll():
        try:
            watcher.poll_once()
        except Exception as error:
            errors.append(str(error))

    health.write_text(json.dumps(payload))  # old completed poll, deliberately aged by fixture
    before_scan = health.read_bytes()
    producer._iter_jsonl_files = delayed_walk
    worker = threading.Thread(target=poll)
    try:
        worker.start()
        if not entered.wait(10):
            raise RuntimeError("private discovery did not reach a filesystem stat")
        scan_started = time.monotonic()
        time.sleep(1)  # real bounded latency; the fixture's 600s age is separately synthetic
        for index in range(1, 4):
            if release.is_set() or not worker.is_alive():
                raise RuntimeError("store stimulus escaped the held filesystem barrier")
            store(index)
        validate_stores(stores)
        if health.read_bytes() != before_scan or not progress or watcher.poll_count != refreshed["poll_count"] + 1:
            raise RuntimeError("scan progress incorrectly became a completed heartbeat")
        check("scan_in_progress", "degraded")
        scan_delay_seconds = time.monotonic() - scan_started
    finally:
        completion_started = datetime.now(timezone.utc)
        release.set()
        worker.join(15)
        producer._iter_jsonl_files = original_walk
    if worker.is_alive() or errors:
        raise RuntimeError(f"private poll did not finish: {errors}")
    completed = json.loads(health.read_text())
    completed_health = {
        "poll_count": completed["poll_count"],
        "updated_at": completed["updated_at"],
        "previous_poll_count": refreshed["poll_count"],
        "previous_updated_at": refreshed["updated_at"],
        "actual_poll_count": watcher.poll_count,
        "completion_started_at": completion_started.isoformat(),
        "completion_finished_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_completion(completed_health, refreshed["poll_count"])
    check("scan_recovery", "running")
    check("stopped", "stopped", process="stopped")
    refreshed["updated_at"] = (
        datetime.fromisoformat(refreshed["updated_at"]).astimezone(timezone(timedelta(hours=3))).isoformat()
    )
    health.write_text(json.dumps(refreshed))
    check("timezone", "running")
    health.write_text('{"updated_at": "not-a-date", "poll_count": 2, "alert_reasons": []}')
    check("malformed", "unknown")
    health.unlink()
    check("missing", "unknown")
    return {
        "schema": 2,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": {p: digest(ROOT / p) for p in SOURCES},
        "binary_sha256": digest(binary),
        "compile_sources": compile_sources,
        "stores": stores,
        "completed_health": completed_health,
        "method": "private Python poll and bridge queue -> compiled production Swift reader/status",
        "producer_poll_counts": [payload["poll_count"], refreshed["poll_count"]],
        "watcher_queued_chunks": len(queued),
        "scan_progress": {
            "stat_calls": len(progress),
            "independent_stores": len(stores[1:]),
            "completed_poll": completed_health["poll_count"],
            "delay_seconds": scan_delay_seconds,
        },
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": 2, "status": "failed"}
    try:
        with tempfile.TemporaryDirectory(prefix="private-heartbeat-") as temp:
            report.update(measure(Path(temp), args.out.parent))
        report["status"] = "measured"
    except Exception as error:
        report["error"] = str(error)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
