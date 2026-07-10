from __future__ import annotations

import pytest

import scripts.benchmark_runtime_store_open as benchmark_module
from brainlayer.runtime_store import OfflineMigrator
from scripts.benchmark_runtime_store_open import benchmark_runtime_open


def test_runtime_open_benchmark_refuses_canonical_path(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical" / "brainlayer.db"
    monkeypatch.setattr("scripts.benchmark_runtime_store_open._canonical_db_path", lambda: canonical)

    with pytest.raises(PermissionError, match="canonical"):
        benchmark_runtime_open(canonical, iterations=1, telemetry_path=tmp_path / "telemetry.jsonl")


def test_runtime_open_benchmark_reports_percentiles_and_zero_corpus_scans(tmp_path):
    db_path = tmp_path / "copy" / "brainlayer.db"
    telemetry_path = tmp_path / "runtime-open.jsonl"
    OfflineMigrator(db_path).close()

    result = benchmark_runtime_open(db_path, iterations=5, telemetry_path=telemetry_path)

    assert result["mode"] == "runtime"
    assert result["iterations"] == 5
    assert result["runtime_open_events"] == 5
    assert result["p50_ms"] <= result["p99_ms"] <= result["max_ms"]
    assert result["p99_ms"] < 100
    assert result["schema_fingerprints"]
    assert result["statement_violations"] == []
    assert result["corpus_scan_statements"] == []


def test_runtime_open_benchmark_requires_positive_iterations(tmp_path):
    with pytest.raises(ValueError, match="iterations"):
        benchmark_runtime_open(
            tmp_path / "copy.db",
            iterations=0,
            telemetry_path=tmp_path / "telemetry.jsonl",
        )


@pytest.mark.parametrize("telemetry_target", ["copy", "canonical"])
def test_runtime_open_benchmark_refuses_database_as_telemetry_path(tmp_path, monkeypatch, telemetry_target):
    canonical = tmp_path / "canonical" / "brainlayer.db"
    copy_path = tmp_path / "copy" / "brainlayer.db"
    monkeypatch.setattr("scripts.benchmark_runtime_store_open._canonical_db_path", lambda: canonical)
    OfflineMigrator(copy_path).close()
    target = copy_path if telemetry_target == "copy" else canonical

    with pytest.raises(PermissionError, match="telemetry path"):
        benchmark_runtime_open(copy_path, iterations=1, telemetry_path=target)


def test_runtime_open_benchmark_rejects_incomplete_telemetry(tmp_path, monkeypatch):
    db_path = tmp_path / "copy" / "brainlayer.db"
    OfflineMigrator(db_path).close()
    monkeypatch.setattr(benchmark_module, "_runtime_finished_events", lambda _path: [])

    with pytest.raises(RuntimeError, match="expected telemetry for 2 runtime opens, got 0"):
        benchmark_runtime_open(
            db_path,
            iterations=2,
            telemetry_path=tmp_path / "runtime-open.jsonl",
        )
