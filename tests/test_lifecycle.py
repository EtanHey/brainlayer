"""Tests for C1 (stale MCP cleanup) and C2 (WAL checkpoint) scripts.

Uses mocking — no real processes killed, no real DB touched.
"""

import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
HOOKS_DIR = Path(__file__).parent.parent / "hooks"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(HOOKS_DIR))


# ── C1: cleanup_stale_mcp tests ─────────────────────────────────────────────


class TestParseEtime:
    def test_seconds_only(self):
        from cleanup_stale_mcp import parse_etime

        assert parse_etime("00:45") == 45

    def test_minutes_seconds(self):
        from cleanup_stale_mcp import parse_etime

        assert parse_etime("05:30") == 330

    def test_hours_minutes_seconds(self):
        from cleanup_stale_mcp import parse_etime

        assert parse_etime("02:30:00") == 9000

    def test_days_hours_minutes_seconds(self):
        from cleanup_stale_mcp import parse_etime

        assert parse_etime("1-12:00:00") == 129600

    def test_zero(self):
        from cleanup_stale_mcp import parse_etime

        assert parse_etime("00:00") == 0


class TestClassifyStale:
    def test_orphan_ppid_1(self):
        from cleanup_stale_mcp import classify_stale

        procs = [{"pid": 100, "ppid": 1, "etime": "00:30", "command": "brainlayer-mcp", "age_seconds": 30}]
        stale, active = classify_stale(procs)
        assert len(stale) == 1
        assert stale[0]["reason"] == "orphaned (ppid=1)"
        assert len(active) == 0

    def test_young_process_with_live_parent(self):
        from cleanup_stale_mcp import classify_stale

        procs = [{"pid": 100, "ppid": 200, "etime": "00:30", "command": "brainlayer-mcp", "age_seconds": 30}]
        stale, active = classify_stale(procs)
        assert len(stale) == 0
        assert len(active) == 1

    @patch("cleanup_stale_mcp.parent_has_active_session", return_value=False)
    def test_old_process_idle_parent(self, mock_active):
        from cleanup_stale_mcp import classify_stale

        procs = [{"pid": 100, "ppid": 200, "etime": "07:00:00", "command": "brainlayer-mcp", "age_seconds": 25200}]
        stale, active = classify_stale(procs)
        assert len(stale) == 1
        assert "stale" in stale[0]["reason"]

    @patch("cleanup_stale_mcp.parent_has_active_session", return_value=True)
    def test_old_process_active_parent(self, mock_active):
        from cleanup_stale_mcp import classify_stale

        procs = [{"pid": 100, "ppid": 200, "etime": "07:00:00", "command": "brainlayer-mcp", "age_seconds": 25200}]
        stale, active = classify_stale(procs)
        assert len(stale) == 0
        assert len(active) == 1


class TestGetMcpProcesses:
    @patch("cleanup_stale_mcp.subprocess.check_output")
    def test_parses_ps_output(self, mock_ps):
        mock_ps.return_value = (
            "  PID  PPID     ELAPSED COMMAND\n"
            "  100     1       05:30 /usr/bin/python3 brainlayer-mcp\n"
            "  200   150       02:00 bun voicelayer-mcp\n"
            "  300   150       01:00 /usr/bin/node server.js\n"
        )
        from cleanup_stale_mcp import get_mcp_processes

        procs = get_mcp_processes()
        assert len(procs) == 2
        assert procs[0]["pid"] == 100
        assert procs[1]["pid"] == 200

    @patch("cleanup_stale_mcp.subprocess.check_output", side_effect=subprocess.SubprocessError)
    def test_handles_ps_failure(self, mock_ps):
        from cleanup_stale_mcp import get_mcp_processes

        assert get_mcp_processes() == []


class TestKillProcesses:
    @patch("cleanup_stale_mcp.os.kill")
    def test_dry_run_does_not_kill(self, mock_kill):
        from cleanup_stale_mcp import kill_processes

        stale = [{"pid": 100, "reason": "orphaned"}]
        killed = kill_processes(stale, dry_run=True)
        mock_kill.assert_not_called()
        assert killed == 0

    @patch("cleanup_stale_mcp.os.kill")
    def test_kill_sends_sigterm(self, mock_kill):
        from cleanup_stale_mcp import kill_processes

        stale = [{"pid": 100, "reason": "orphaned"}]
        killed = kill_processes(stale, dry_run=False)
        mock_kill.assert_called_once_with(100, signal.SIGTERM)
        assert killed == 1

    @patch("cleanup_stale_mcp.os.kill", side_effect=ProcessLookupError)
    def test_handles_already_dead(self, mock_kill):
        from cleanup_stale_mcp import kill_processes

        stale = [{"pid": 100, "reason": "orphaned"}]
        killed = kill_processes(stale, dry_run=False)
        assert killed == 0


# ── C2: wal_checkpoint tests ────────────────────────────────────────────────


class TestWalCheckpoint:
    def test_checkpoint_on_real_db(self, tmp_path):
        """Create a real SQLite DB in WAL mode, write data, checkpoint."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, data TEXT)")
        for i in range(100):
            conn.execute("INSERT INTO t VALUES (?, ?)", (i, f"data_{i}" * 100))
        conn.commit()

        # Open a second connection to keep WAL alive while first closes
        conn2 = sqlite3.connect(db_path)
        conn2.execute("SELECT * FROM t LIMIT 1").fetchone()
        conn.close()

        wal_path = db_path + "-wal"

        from wal_checkpoint import checkpoint

        busy, log_pages, checkpointed = checkpoint(db_path)
        conn2.close()
        # After checkpoint, should succeed
        assert checkpointed >= 0

    def test_format_size(self):
        from wal_checkpoint import format_size

        assert format_size(500) == "500.0B"
        assert format_size(1024) == "1.0KB"
        assert format_size(1024 * 1024) == "1.0MB"
        assert format_size(1024 * 1024 * 1024) == "1.0GB"

    def test_get_wal_size_missing(self):
        from wal_checkpoint import get_wal_size

        assert get_wal_size("/nonexistent/path.db") == 0

    def test_get_wal_size_exists(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        wal_path = db_path + "-wal"
        # Create a fake WAL file
        with open(wal_path, "wb") as f:
            f.write(b"x" * 1024)
        from wal_checkpoint import get_wal_size

        assert get_wal_size(db_path) == 1024


class TestVectorMaintenanceScripts:
    def test_purge_orphaned_vectors_uses_cli_db_path(self, tmp_path):
        from purge_orphaned_vectors import resolve_db_path

        assert resolve_db_path(str(tmp_path / "custom.db")) == tmp_path / "custom.db"

    def test_purge_orphaned_vectors_uses_default_db_path(self, tmp_path, monkeypatch):
        import purge_orphaned_vectors

        monkeypatch.setattr(purge_orphaned_vectors, "get_db_path", lambda: tmp_path / "brainlayer.db")
        assert purge_orphaned_vectors.resolve_db_path() == tmp_path / "brainlayer.db"

    def test_rebuild_vec0_tables_uses_cli_db_path(self, tmp_path):
        from rebuild_vec0_tables import resolve_db_path

        assert resolve_db_path(str(tmp_path / "custom.db")) == tmp_path / "custom.db"

    def test_rebuild_vec0_tables_uses_default_db_path(self, tmp_path, monkeypatch):
        import rebuild_vec0_tables

        monkeypatch.setattr(rebuild_vec0_tables, "get_db_path", lambda: tmp_path / "brainlayer.db")
        assert rebuild_vec0_tables.resolve_db_path() == tmp_path / "brainlayer.db"

    def test_rebuild_vec0_tables_batches_without_materializing(self):
        from rebuild_vec0_tables import batched

        assert list(batched((str(i) for i in range(5)), 2)) == [["0", "1"], ["2", "3"], ["4"]]

    def test_purge_orphaned_vectors_batches_without_materializing(self):
        from purge_orphaned_vectors import batched

        assert list(batched((str(i) for i in range(5)), 2)) == [["0", "1"], ["2", "3"], ["4"]]

    def test_purge_orphaned_vectors_streams_orphan_ids(self):
        from purge_orphaned_vectors import iter_orphan_ids

        class _Result:
            def __iter__(self):
                yield ("chunk-1",)
                yield ("chunk-2",)

            def fetchall(self):  # pragma: no cover - should never be used
                raise AssertionError("fetchall should not be used")

        class _Conn:
            def execute(self, sql):
                return _Result()

        assert list(iter_orphan_ids(_Conn(), "chunk_vectors_rowids")) == ["chunk-1", "chunk-2"]

    def test_purge_orphaned_vectors_raises_when_delete_errors_occur(self, monkeypatch):
        import purge_orphaned_vectors

        monkeypatch.setattr(purge_orphaned_vectors.time, "time", lambda: 1_000_000.0)

        class _CountResult:
            def fetchone(self):
                return (2,)

        class _IterResult:
            def __iter__(self):
                yield ("chunk-1",)
                yield ("chunk-2",)

        class _Conn:
            def execute(self, sql, params=None):
                if sql.lstrip().startswith("SELECT COUNT(*)"):
                    return _CountResult()
                if sql.lstrip().startswith("SELECT vr.id"):
                    return _IterResult()
                if sql.startswith("DELETE FROM") and params == ("chunk-2",):
                    raise RuntimeError("busy")
                return None

        with pytest.raises(RuntimeError, match="Failed to purge 1 orphaned vectors from chunk_vectors"):
            purge_orphaned_vectors.purge_vec_table(
                _Conn(), "chunk_vectors", "chunk_vectors_rowids", "chunk_vectors"
            )

    def test_rebuild_vec0_tables_read_embedding_or_raise_returns_embedding(self):
        from rebuild_vec0_tables import read_embedding_or_raise

        class _Result:
            def fetchone(self):
                return (b"embedding",)

        class _Conn:
            def execute(self, sql, params):
                return _Result()

        assert read_embedding_or_raise(_Conn(), "chunk_vectors", "chunk-1") == b"embedding"

    def test_rebuild_vec0_tables_read_embedding_or_raise_raises_on_missing_row(self):
        from rebuild_vec0_tables import read_embedding_or_raise

        class _Result:
            def fetchone(self):
                return None

        class _Conn:
            def execute(self, sql, params):
                return _Result()

        with pytest.raises(RuntimeError, match="Missing embedding"):
            read_embedding_or_raise(_Conn(), "chunk_vectors", "chunk-1")

    def test_rebuild_vec0_tables_read_embedding_or_raise_raises_on_execute_error(self):
        from rebuild_vec0_tables import read_embedding_or_raise

        class _Conn:
            def execute(self, sql, params):
                raise RuntimeError("busy")

        with pytest.raises(RuntimeError, match="Failed to read embedding"):
            read_embedding_or_raise(_Conn(), "chunk_vectors", "chunk-1")

    def test_rebuild_vec0_tables_ensure_restore_succeeded_raises_when_backup_must_be_kept(self):
        from rebuild_vec0_tables import ensure_restore_succeeded

        with pytest.raises(RuntimeError, match="_tmp_vec_backup"):
            ensure_restore_succeeded(1, "_tmp_vec_backup", "chunk_vectors")


# ── Session cleanup hook tests ──────────────────────────────────────────────


class TestSessionCleanupHook:
    @staticmethod
    def _load_hook():
        import importlib.util

        spec = importlib.util.spec_from_file_location("session_cleanup", HOOKS_DIR / "session-cleanup.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_hook_imports(self):
        """Verify the hook module can be imported."""
        mod = self._load_hook()
        assert hasattr(mod, "cleanup_stale_mcp")
        assert hasattr(mod, "wal_checkpoint")
        assert hasattr(mod, "main")

    def test_hook_parse_etime(self):
        mod = self._load_hook()
        assert mod.parse_etime("1-02:30:00") == 95400
        assert mod.parse_etime("05:30") == 330
