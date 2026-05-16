from __future__ import annotations

import importlib.util
import sqlite3
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
