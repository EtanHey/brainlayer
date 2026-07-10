#!/usr/bin/env python3
"""Measure legacy and Phase 1 writer-open latency on an explicit DB copy."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from brainlayer.runtime_store import WriterRuntimeStore

_TELEMETRY_ENV = {
    "BRAINLAYER_WRITER_TELEMETRY": "1",
    "BRAINLAYER_WRITER_TELEMETRY_FTS_SAMPLE_TTL_SECONDS": "0",
}
_MUTATION_TOKENS = ("CREATE ", "DROP ", "ALTER ", "INSERT ", "UPDATE ", "DELETE ", "OPTIMIZE")
_CORPUS_TABLE_TOKENS = ("FROM CHUNKS ", "FROM CHUNKS_FTS", "FROM CHUNK_FTS_ROWIDS")


def _canonical_db_path() -> Path:
    from brainlayer.paths import get_db_path

    return get_db_path()


def _assert_copy_path(db_path: Path) -> Path:
    resolved = db_path.expanduser().resolve()
    if resolved == _canonical_db_path().expanduser().resolve():
        raise PermissionError("runtime-open benchmark refuses the canonical BrainLayer database")
    return resolved


@contextmanager
def _temporary_environment(updates: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _timing_summary(durations_ms: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(_percentile(durations_ms, 0.50), 3),
        "p95_ms": round(_percentile(durations_ms, 0.95), 3),
        "p99_ms": round(_percentile(durations_ms, 0.99), 3),
        "max_ms": round(max(durations_ms), 3),
    }


def _runtime_finished_events(telemetry_path: Path) -> list[dict[str, Any]]:
    if not telemetry_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in telemetry_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "txn_finished" and event.get("operation") == "runtime_open":
            events.append(event)
    return events


def benchmark_runtime_open(
    db_path: Path,
    *,
    iterations: int,
    telemetry_path: Path,
) -> dict[str, Any]:
    """Measure runtime opens and prove telemetry contains no corpus work."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    path = _assert_copy_path(Path(db_path))
    telemetry_path = Path(telemetry_path).expanduser().resolve()
    if telemetry_path in {path, _canonical_db_path().expanduser().resolve()}:
        raise PermissionError("telemetry path must differ from both the benchmark copy and canonical database")
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.unlink(missing_ok=True)

    durations_ms: list[float] = []
    environment = {
        **_TELEMETRY_ENV,
        "BRAINLAYER_WRITER_TELEMETRY_PATH": str(telemetry_path),
        "BRAINLAYER_WRITER_HEARTBEAT_DIR": str(telemetry_path.parent / "runtime-open-heartbeats"),
        "BRAINLAYER_WRITER_PIDFILE_DIR": str(telemetry_path.parent / "runtime-open-pidfiles"),
        "BRAINLAYER_RUNTIME_STORE": "runtime",
    }
    with _temporary_environment(environment):
        for _ in range(iterations):
            started = time.perf_counter()
            store = WriterRuntimeStore(path)
            store.close()
            durations_ms.append((time.perf_counter() - started) * 1000.0)

    events = _runtime_finished_events(telemetry_path)
    if len(events) != iterations:
        raise RuntimeError(f"expected telemetry for {iterations} runtime opens, got {len(events)}")
    statements = [statement for event in events for statement in event.get("statements", [])]
    normalized = [str(statement.get("normalized_sql") or "").upper() for statement in statements]
    statement_violations = sorted(
        {statement for statement in normalized if any(token in statement for token in _MUTATION_TOKENS)}
    )
    corpus_scan_statements = sorted(
        {statement for statement in normalized if any(token in statement for token in _CORPUS_TABLE_TOKENS)}
    )
    fingerprints = sorted(
        {str(event["schema_fingerprint"]) for event in events if event.get("schema_fingerprint")}
    )
    return {
        "mode": "runtime",
        "db_path": str(path),
        "db_bytes": path.stat().st_size,
        "iterations": iterations,
        **_timing_summary(durations_ms),
        "runtime_open_events": len(events),
        "schema_fingerprints": fingerprints,
        "statement_violations": statement_violations,
        "corpus_scan_statements": corpus_scan_statements,
    }


def benchmark_legacy_open(
    db_path: Path,
    *,
    iterations: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Measure the rollback constructor in killable subprocesses on a copy."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    path = _assert_copy_path(Path(db_path))
    durations_ms: list[float] = []
    timeouts = 0
    code = "from pathlib import Path; from brainlayer.vector_store import VectorStore; VectorStore(Path(__import__('sys').argv[1])).close()"
    environment = {**os.environ, "BRAINLAYER_WRITER_TELEMETRY": "0"}
    for _ in range(iterations):
        started = time.perf_counter()
        try:
            subprocess.run(
                [sys.executable, "-c", code, str(path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=environment,
            )
            durations_ms.append((time.perf_counter() - started) * 1000.0)
        except subprocess.TimeoutExpired:
            durations_ms.append(timeout_seconds * 1000.0)
            timeouts += 1
            break
    return {
        "mode": "legacy",
        "db_path": str(path),
        "db_bytes": path.stat().st_size,
        "iterations_requested": iterations,
        "iterations_completed": len(durations_ms) - timeouts,
        "timeouts": timeouts,
        "timeout_seconds": timeout_seconds,
        **_timing_summary(durations_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path, help="Explicit non-canonical database copy")
    parser.add_argument("--mode", choices=("runtime", "legacy"), default="runtime")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--telemetry-path", type=Path, default=Path("runtime-open-benchmark.jsonl"))
    parser.add_argument("--legacy-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    if args.mode == "runtime":
        result = benchmark_runtime_open(
            args.db_path,
            iterations=args.iterations,
            telemetry_path=args.telemetry_path,
        )
    else:
        result = benchmark_legacy_open(
            args.db_path,
            iterations=args.iterations,
            timeout_seconds=args.legacy_timeout_seconds,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
