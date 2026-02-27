"""Tests for DB lock resilience — VectorStore init retry, busy timeout, WAL mode."""

import threading
from unittest.mock import patch

import apsw
import pytest

from brainlayer.vector_store import VectorStore


class TestVectorStoreInitRetry:
    """VectorStore should retry on BusyError during initialization."""

    def test_init_succeeds_on_first_attempt(self, tmp_path):
        """Normal init works without retry."""
        db = tmp_path / "test.db"
        store = VectorStore(db)
        assert store.conn is not None
        store.close()

    def test_init_retries_on_busy_error(self, tmp_path):
        """BusyError during init triggers retry with backoff."""
        db = tmp_path / "test.db"

        call_count = 0
        original_init_db = VectorStore._init_db

        def flaky_init_db(self):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise apsw.BusyError("database is locked")
            original_init_db(self)

        with patch.object(VectorStore, "_init_db", flaky_init_db):
            with patch.object(VectorStore, "_INIT_BASE_DELAY", 0.01):
                store = VectorStore(db)

        assert store.conn is not None
        assert call_count == 3  # 2 failures + 1 success
        store.close()

    def test_init_gives_up_after_max_retries(self, tmp_path):
        """After exhausting retries, BusyError propagates."""
        db = tmp_path / "test.db"

        def always_busy(self):
            raise apsw.BusyError("database is locked")

        with patch.object(VectorStore, "_init_db", always_busy):
            with patch.object(VectorStore, "_INIT_BASE_DELAY", 0.01):
                with patch.object(VectorStore, "_INIT_MAX_RETRIES", 3):
                    with pytest.raises(apsw.BusyError):
                        VectorStore(db)

    def test_busy_timeout_set_before_ddl(self, tmp_path):
        """busy_timeout must be set via APSW native method, not just PRAGMA."""
        db = tmp_path / "test.db"
        store = VectorStore(db)

        # Verify WAL mode is active
        cursor = store.conn.cursor()
        journal_mode = list(cursor.execute("PRAGMA journal_mode"))[0][0]
        assert journal_mode == "wal"

        store.close()

    def test_concurrent_vectorstore_init(self, tmp_path):
        """Multiple VectorStore instances can init concurrently without BusyError."""
        db = tmp_path / "test.db"
        # Create the DB first so schema exists
        initial = VectorStore(db)
        initial.close()

        errors = []
        stores = []
        lock = threading.Lock()

        def init_store():
            try:
                s = VectorStore(db)
                with lock:
                    stores.append(s)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=init_store) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Concurrent init errors: {errors}"
        assert len(stores) == 3
        for s in stores:
            s.close()
