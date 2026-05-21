from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

from brainlayer import drain
from brainlayer.vector_store import VectorStore


def _expected_pidfile(pidfile_dir: Path, db_path: Path) -> Path:
    return pidfile_dir / f"brainlayer-writer-{db_path.name}.pid"


def test_pidfile_created_on_rw_init(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    store = VectorStore(db_path)
    try:
        pidfile = _expected_pidfile(pidfile_dir, db_path)
        assert pidfile.exists()
        assert pidfile.read_text(encoding="utf-8").strip() == str(os.getpid())
    finally:
        store.close()


def test_pidfile_blocks_second_writer(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    store = VectorStore(db_path)
    try:
        script = """
from pathlib import Path
from brainlayer.vector_store import VectorStore, WriterInUseError

try:
    VectorStore(Path(__import__("os").environ["DB_PATH"]))
except WriterInUseError as exc:
    print(type(exc).__name__)
    print(str(exc))
    raise SystemExit(0)
raise SystemExit("second writer unexpectedly acquired the pidfile")
"""
        env = {
            **os.environ,
            "BRAINLAYER_WRITER_PIDFILE_DIR": str(pidfile_dir),
            "DB_PATH": str(db_path),
            "PYTHONPATH": f"{Path(__file__).resolve().parents[1] / 'src'}:{os.environ.get('PYTHONPATH', '')}",
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "WriterInUseError" in result.stdout
        assert f"another writer is using {db_path} (pid {os.getpid()})" in result.stdout
    finally:
        store.close()


def test_stale_pidfile_cleaned_up(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    pidfile_dir.mkdir()
    pidfile = _expected_pidfile(pidfile_dir, db_path)
    pidfile.write_text("999999", encoding="utf-8")

    store = VectorStore(db_path)
    try:
        assert pidfile.exists()
        assert pidfile.read_text(encoding="utf-8").strip() == str(os.getpid())
    finally:
        store.close()


def test_pidfile_removed_on_close(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    store = VectorStore(db_path)
    pidfile = _expected_pidfile(pidfile_dir, db_path)
    assert pidfile.exists()

    store.close()

    assert not pidfile.exists()


def test_readonly_init_skips_pidfile(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    writer = VectorStore(db_path)
    writer.close()

    reader = VectorStore(db_path, readonly=True)
    try:
        pidfile = _expected_pidfile(pidfile_dir, db_path)
        assert not pidfile.exists()
        assert reader._writer_pidfile_acquired is False
        assert not hasattr(reader, "_writer_pidfile_path_value")
    finally:
        reader.close()


def test_init_retries_10_with_extended_backoff():
    assert VectorStore._INIT_MAX_RETRIES == 10
    delays = [
        min(VectorStore._INIT_BASE_DELAY * (2**attempt), VectorStore._INIT_MAX_DELAY)
        for attempt in range(VectorStore._INIT_MAX_RETRIES)
    ]
    assert delays == [0.5, 1, 2, 4, 8, 16, 30, 30, 30, 30]
    assert sum(delays) <= 600


def test_drain_busy_timeout_30s(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_DRAIN_BUSY_TIMEOUT_MS", "30000")

    conn = drain._open_connection(tmp_path / "drain.db")
    try:
        busy_timeout_ms = conn.cursor().execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy_timeout_ms >= 30000
    finally:
        conn.close()


def test_enrichment_plist_throttle_interval_at_least_60s():
    plist_path = Path(__file__).resolve().parents[1] / "scripts" / "launchd" / "com.brainlayer.enrichment.plist"
    plist = plistlib.loads(plist_path.read_bytes())

    assert plist["ThrottleInterval"] >= 60
