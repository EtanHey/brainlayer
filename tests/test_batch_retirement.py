"""Retired batch entry points leave historical checkpoints and result files alone."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", ["module", "wrapper", "monitor", "orchestrator"])
def test_retired_batch_entrypoints_are_transport_free_and_preserve_files(tmp_path, entry):
    data = tmp_path / "backfill_data"
    data.mkdir()
    result_file = data / "batch_history.jsonl"
    result_file.write_text('{"key":"historical","response":"saved"}\n')
    checkpoint = tmp_path / "enrichment_checkpoints.db"
    checkpoint.write_bytes(b"historical checkpoint bytes")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    commands = {
        "module": [sys.executable, "-m", "brainlayer.cloud_backfill", "--help"],
        "wrapper": [sys.executable, str(ROOT / "scripts/cloud_backfill.py"), "--help"],
        "monitor": [sys.executable, str(ROOT / "scripts/monitor_batch_reenrichment.py"), "--help"],
        "orchestrator": ["bash", str(ROOT / "scripts/backfill_orchestrate.sh")],
    }
    result = subprocess.run(
        commands[entry],
        cwd=tmp_path,
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "PYTHONPATH": str(ROOT / "src"),
            "BRAINLAYER_DB": str(tmp_path / "unused.db"),
            "GOOGLE_API_KEY": "",
            "GOOGLE_GENERATIVE_AI_API_KEY": "",
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "retired" in (result.stdout + result.stderr).lower()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("flag", ["--resume", "--submit-only", "--status", "--dry-run"])
def test_batch_module_rejects_stale_modes_before_database_open(monkeypatch, capsys, flag):
    from brainlayer import cloud_backfill

    def forbidden(*args, **kwargs):
        pytest.fail("retired batch command opened a database")

    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-key")
    monkeypatch.setattr(cloud_backfill, "VectorStore", forbidden)
    monkeypatch.setattr(sys, "argv", ["batch", flag])
    assert cloud_backfill.main() == 1
    assert "retired" in capsys.readouterr().err


def test_batch_submission_orchestrator_is_removed():
    from brainlayer import cloud_backfill

    assert not hasattr(cloud_backfill, "run_full_backfill")


@pytest.mark.parametrize("name", ["process_pending_jobs_once", "resume_backfill"])
def test_batch_remote_resume_executors_are_removed(name):
    from brainlayer import cloud_backfill

    assert not hasattr(cloud_backfill, name)


def test_batch_remote_submission_sender_is_removed():
    from brainlayer import cloud_backfill

    assert not hasattr(cloud_backfill, "submit_gemini_batch")


def test_batch_preview_export_producer_is_removed():
    from brainlayer import cloud_backfill

    assert not hasattr(cloud_backfill, "export_unenriched_chunks")


@pytest.mark.parametrize("name", ["export_backlog_drain_chunks", "_init_sanitizer", "build_batch_request_line"])
def test_batch_backlog_export_helpers_are_removed(name):
    from brainlayer import cloud_backfill

    assert not hasattr(cloud_backfill, name)
