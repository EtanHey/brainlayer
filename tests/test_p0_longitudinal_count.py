from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from brainlayer import p0_longitudinal_count as counter

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_p0_counter_normalizes_created_at_offsets(tmp_path):
    db_path = tmp_path / "counter.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (created_at TEXT, content TEXT)")
        conn.execute(
            "INSERT INTO chunks (created_at, content) VALUES (?, ?)",
            ("2026-05-16T14:00:00+00:00", 'MCP BrainLayer Memory: Invalid JSON-RPC message: {"jsonrpc":"2.0"}'),
        )

    rows = counter.run_count(db_path)

    assert rows == [{"day": "2026-05-16", "new_audit_chunks": 1}]


def test_p0_counter_counts_spaced_brain_search_box(tmp_path):
    db_path = tmp_path / "counter.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (created_at TEXT, content TEXT)")
        conn.execute(
            "INSERT INTO chunks (created_at, content) VALUES (?, ?)",
            ("2026-05-16T14:00:00+00:00", '┌─ brain_search: "audit recursion" ─ 1 result'),
        )

    rows = counter.run_count(db_path)

    assert rows == [{"day": "2026-05-16", "new_audit_chunks": 1}]


def test_p0_counter_counts_leading_whitespace_brain_search_box(tmp_path):
    db_path = tmp_path / "counter.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (created_at TEXT, content TEXT)")
        conn.execute(
            "INSERT INTO chunks (created_at, content) VALUES (?, ?)",
            ("2026-05-16T14:00:00+00:00", '\n  ┌─ brain_search: "audit recursion" ─ 1 result'),
        )

    rows = counter.run_count(db_path)

    assert rows == [{"day": "2026-05-16", "new_audit_chunks": 1}]


def test_p0_counter_counts_entity_mcp_boxes(tmp_path):
    db_path = tmp_path / "counter.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (created_at TEXT, content TEXT)")
        for content in ("┌─entity: Etan", "┌─ entity: Etan", "┌─ entity search: Etan"):
            conn.execute(
                "INSERT INTO chunks (created_at, content) VALUES (?, ?)",
                ("2026-05-16T14:00:00+00:00", content),
            )

    rows = counter.run_count(db_path)

    assert rows == [{"day": "2026-05-16", "new_audit_chunks": 3}]


def test_p0_counter_wrapper_runs_from_source_checkout_without_install(tmp_path):
    db_path = tmp_path / "counter.db"
    output_dir = tmp_path / "reports"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (created_at TEXT, content TEXT)")
        conn.execute(
            "INSERT INTO chunks (created_at, content) VALUES (?, ?)",
            ("2026-05-16T14:00:00+00:00", '┌─ brain_search: "audit recursion" ─ 1 result'),
        )

    env = {
        **os.environ,
        "BRAINLAYER_DB": str(db_path),
        "BRAINLAYER_P0_COUNTER_DIR": str(output_dir),
        "PYTHONPATH": "",
    }
    result = subprocess.run(
        [sys.executable, "-S", str(REPO_ROOT / "scripts" / "p0_longitudinal_count.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["total_new_audit_chunks"] == 1
    assert Path(payload["output_path"]).is_file()


def test_p0_counter_cutoff_uses_since_parameter(tmp_path):
    db_path = tmp_path / "counter.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (created_at TEXT, content TEXT)")
        conn.execute(
            "INSERT INTO chunks (created_at, content) VALUES (?, ?)",
            ("2026-05-16T14:00:00+00:00", 'MCP BrainLayer Memory: Invalid JSON-RPC message: {"jsonrpc":"2.0"}'),
        )

    rows = counter.run_count(db_path, since="2026-05-16T14:30:00+00:00")

    assert rows == []


def test_p0_counter_payload_surfaces_sqlite_error_without_positive_verdict(tmp_path):
    db_path = tmp_path / "counter.db"
    db_path.write_bytes(b"")

    since_utc = datetime.fromisoformat(counter.SINCE).astimezone(timezone.utc)
    payload = counter.build_payload(db_path, now=since_utc + timedelta(days=8))

    with pytest.raises(sqlite3.OperationalError, match="no such table: chunks"):
        counter.run_count(db_path)
    assert payload["rows"] == []
    assert payload["total_new_audit_chunks"] == 0
    assert "no such table: chunks" in payload["count_error"]
    assert payload["verdict_ready"] is False
    assert payload["structural_fix_p_lt_0_001"] is False
    assert payload["brain_store_verdict_content"] is None


def test_p0_counter_main_writes_diagnostic_json_for_missing_db(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "missing.db"
    output_dir = tmp_path / "reports"
    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_P0_COUNTER_DIR", str(output_dir))

    exit_code = counter.main()

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["db_path"] == str(db_path)
    assert payload["count_error"]
    assert Path(payload["output_path"]).is_file()
    persisted = json.loads(Path(payload["output_path"]).read_text(encoding="utf-8"))
    assert persisted["count_error"] == payload["count_error"]


def test_p0_counter_verdict_waits_for_full_168_hours(tmp_path):
    db_path = tmp_path / "counter.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (created_at TEXT, content TEXT)")

    since_utc = datetime.fromisoformat(counter.SINCE).astimezone(timezone.utc)
    payload = counter.build_payload(db_path, now=since_utc + timedelta(days=7, seconds=-1))

    assert payload["elapsed_days"] == 6
    assert payload["verdict_ready"] is False
    assert payload["brain_store_verdict_content"] is None


def test_p0_counter_verdict_ready_after_full_168_hours(tmp_path):
    db_path = tmp_path / "counter.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (created_at TEXT, content TEXT)")

    since_utc = datetime.fromisoformat(counter.SINCE).astimezone(timezone.utc)
    payload = counter.build_payload(db_path, now=since_utc + timedelta(days=7))

    assert payload["elapsed_days"] == 7
    assert payload["verdict_ready"] is True
    assert payload["structural_fix_p_lt_0_001"] is True
