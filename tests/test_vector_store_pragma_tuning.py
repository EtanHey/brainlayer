import pytest

from brainlayer.vector_store import VectorStore


def _pragma_value(conn, name: str) -> int:
    row = conn.cursor().execute(f"PRAGMA {name}").fetchone()
    return int(row[0])


def test_read_connection_sets_mmap_size(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_READ_MMAP_BYTES", raising=False)
    store = VectorStore(tmp_path / "pragma-read.db")
    store.close()

    reader = VectorStore(tmp_path / "pragma-read.db", readonly=True)
    try:
        mmap_size = _pragma_value(reader._get_read_conn(), "mmap_size")
        if mmap_size == 0:
            pytest.skip("SQLite build does not support mmap_size")
        assert mmap_size > 0
    finally:
        reader.close()


def test_read_connection_sets_private_cache_size(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_READ_CACHE_KB", raising=False)
    store = VectorStore(tmp_path / "pragma-cache.db")
    store.close()

    reader = VectorStore(tmp_path / "pragma-cache.db", readonly=True)
    try:
        assert _pragma_value(reader._get_read_conn(), "cache_size") == -64000
    finally:
        reader.close()


def test_writer_init_sets_wal_autocheckpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_WAL_AUTOCHECKPOINT", raising=False)
    store = VectorStore(tmp_path / "pragma-wal.db")
    try:
        assert _pragma_value(store.conn, "wal_autocheckpoint") == 10000
    finally:
        store.close()


def test_readonly_primary_connection_uses_bounded_reader_pragmas(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_READ_BUSY_TIMEOUT_MS", raising=False)
    store = VectorStore(tmp_path / "pragma-readonly-primary.db")
    store.close()

    reader = VectorStore(tmp_path / "pragma-readonly-primary.db", readonly=True)
    try:
        assert _pragma_value(reader.conn, "busy_timeout") == 5000
        assert _pragma_value(reader.conn, "query_only") == 1
        assert _pragma_value(reader.conn, "wal_autocheckpoint") == 0
    finally:
        reader.close()


def test_threadlocal_read_connection_uses_bounded_reader_pragmas(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_READ_BUSY_TIMEOUT_MS", raising=False)
    store = VectorStore(tmp_path / "pragma-threadlocal-read.db")
    store.close()

    reader = VectorStore(tmp_path / "pragma-threadlocal-read.db", readonly=True)
    try:
        conn = reader._get_read_conn()
        assert _pragma_value(conn, "busy_timeout") == 5000
        assert _pragma_value(conn, "query_only") == 1
        assert _pragma_value(conn, "wal_autocheckpoint") == 0
    finally:
        reader.close()


def test_read_busy_timeout_env_override_applies_to_read_connections(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_READ_BUSY_TIMEOUT_MS", "750")
    store = VectorStore(tmp_path / "pragma-read-busy-env.db")
    store.close()

    reader = VectorStore(tmp_path / "pragma-read-busy-env.db", readonly=True)
    try:
        assert _pragma_value(reader.conn, "busy_timeout") == 750
        assert _pragma_value(reader._get_read_conn(), "busy_timeout") == 750
    finally:
        reader.close()


def test_pragma_env_overrides_apply_to_new_connections(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_WAL_AUTOCHECKPOINT", "2222")
    monkeypatch.setenv("BRAINLAYER_READ_MMAP_BYTES", "0")
    monkeypatch.setenv("BRAINLAYER_READ_CACHE_KB", "12345")

    store = VectorStore(tmp_path / "pragma-overrides.db")
    try:
        assert _pragma_value(store.conn, "wal_autocheckpoint") == 2222
    finally:
        store.close()

    reader = VectorStore(tmp_path / "pragma-overrides.db", readonly=True)
    try:
        conn = reader._get_read_conn()
        assert _pragma_value(conn, "mmap_size") == 0
        assert _pragma_value(conn, "cache_size") == -12345
    finally:
        reader.close()


def test_invalid_pragma_env_values_fall_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_WAL_AUTOCHECKPOINT", "not-an-int")
    monkeypatch.setenv("BRAINLAYER_READ_MMAP_BYTES", "not-an-int")
    monkeypatch.setenv("BRAINLAYER_READ_CACHE_KB", "not-an-int")

    store = VectorStore(tmp_path / "pragma-invalid-env.db")
    try:
        assert _pragma_value(store.conn, "wal_autocheckpoint") == 10000
    finally:
        store.close()

    reader = VectorStore(tmp_path / "pragma-invalid-env.db", readonly=True)
    try:
        conn = reader._get_read_conn()
        assert _pragma_value(conn, "mmap_size") > 0
        assert _pragma_value(conn, "cache_size") == -64000
    finally:
        reader.close()
