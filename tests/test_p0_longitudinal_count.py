from __future__ import annotations

import importlib.util
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_counter_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "p0_longitudinal_count.py"
    spec = importlib.util.spec_from_file_location("p0_longitudinal_count", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_p0_counter_normalizes_created_at_offsets(tmp_path):
    counter = _load_counter_module()
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
    counter = _load_counter_module()
    db_path = tmp_path / "counter.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (created_at TEXT, content TEXT)")
        conn.execute(
            "INSERT INTO chunks (created_at, content) VALUES (?, ?)",
            ("2026-05-16T14:00:00+00:00", '┌─ brain_search: "audit recursion" ─ 1 result'),
        )

    rows = counter.run_count(db_path)

    assert rows == [{"day": "2026-05-16", "new_audit_chunks": 1}]


def test_p0_counter_cutoff_uses_since_parameter(tmp_path):
    counter = _load_counter_module()
    db_path = tmp_path / "counter.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (created_at TEXT, content TEXT)")
        conn.execute(
            "INSERT INTO chunks (created_at, content) VALUES (?, ?)",
            ("2026-05-16T14:00:00+00:00", 'MCP BrainLayer Memory: Invalid JSON-RPC message: {"jsonrpc":"2.0"}'),
        )

    rows = counter.run_count(db_path, since="2026-05-16T14:30:00+00:00")

    assert rows == []


def test_p0_counter_verdict_waits_for_full_168_hours(tmp_path):
    counter = _load_counter_module()
    db_path = tmp_path / "counter.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (created_at TEXT, content TEXT)")

    since_utc = datetime.fromisoformat(counter.SINCE).astimezone(timezone.utc)
    payload = counter.build_payload(db_path, now=since_utc + timedelta(days=7, seconds=-1))

    assert payload["elapsed_days"] == 6
    assert payload["verdict_ready"] is False
    assert payload["brain_store_verdict_content"] is None


def test_p0_counter_verdict_ready_after_full_168_hours(tmp_path):
    counter = _load_counter_module()
    db_path = tmp_path / "counter.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (created_at TEXT, content TEXT)")

    since_utc = datetime.fromisoformat(counter.SINCE).astimezone(timezone.utc)
    payload = counter.build_payload(db_path, now=since_utc + timedelta(days=7))

    assert payload["elapsed_days"] == 7
    assert payload["verdict_ready"] is True
    assert payload["structural_fix_p_lt_0_001"] is True
