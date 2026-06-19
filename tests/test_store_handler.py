"""Tests for MCP store handler responses."""

import asyncio
import json
import threading
import time
from unittest.mock import MagicMock, patch

import apsw
import pytest


@pytest.mark.asyncio
async def test_busy_queue_fallback_returns_queued_chunk_id(tmp_path):
    """DB-busy fallback returns the durable queue chunk ID, not a sentinel."""
    from brainlayer.mcp.store_handler import _store

    queue_dir = tmp_path / "queue"

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("locked")),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        texts, structured = await _store(
            content="test memory",
            memory_type="note",
            project="test",
        )

    queued_files = list(queue_dir.glob("mcp-*.jsonl"))
    assert len(queued_files) == 1
    queued_event = json.loads(queued_files[0].read_text())
    expected_chunk_id = queued_event["chunk_id"]

    assert expected_chunk_id != "queued"
    assert structured["chunk_id"] == expected_chunk_id
    assert structured["queued"] is True
    assert any(expected_chunk_id in item.text for item in texts)


@pytest.mark.asyncio
async def test_busy_queue_fallback_returns_loud_deferred_receipt(tmp_path):
    """DB-busy fallback is a structured DEFERRED receipt, not quiet success."""
    from brainlayer.mcp.store_handler import _store

    queue_dir = tmp_path / "queue"

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("locked")),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        texts, structured = await _store(
            content="deferred receipt must be loud",
            memory_type="note",
            project="test",
        )

    queued_files = list(queue_dir.glob("mcp-*.jsonl"))
    assert len(queued_files) == 1
    queued_event = json.loads(queued_files[0].read_text())

    assert structured["status"] == "DEFERRED"
    assert structured["deferred"]["reason"] == "DB_BUSY"
    assert structured["deferred"]["chunk_id"] == queued_event["chunk_id"]
    assert structured["deferred"]["queue_path"] == str(queued_files[0])
    assert structured["deferred"]["action"] == "queued_for_drain"
    assert any("DEFERRED" in item.text for item in texts)


@pytest.mark.asyncio
async def test_store_validates_before_busy_deferral(tmp_path):
    """Busy before store_memory must not queue requests that validation would reject."""
    from brainlayer.mcp.store_handler import _store

    queue_dir = tmp_path / "queue"

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store", side_effect=apsw.BusyError("database is locked")),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        result = await _store(
            content="invalid memory type should never be queued",
            memory_type="invalid_type",
            project="test",
        )

    assert result.isError is True
    assert "Validation error" in result.content[0].text
    assert not list(queue_dir.glob("mcp-*.jsonl"))


@pytest.mark.asyncio
async def test_store_busy_budget_defers_promptly_and_pending_flush_replays(tmp_path, monkeypatch):
    """A held write lock must hit the bounded store budget, defer, and replay later."""
    from brainlayer.mcp.store_handler import _flush_pending_stores, _store

    pending_path = tmp_path / "pending-stores.jsonl"
    attempts = 0

    def held_write_lock_store_memory(**kwargs):
        nonlocal attempts
        attempts += 1
        time.sleep(0.075)
        raise apsw.BusyError("database is locked")

    monkeypatch.setenv("BRAINLAYER_STORE_BUSY_BUDGET_MS", "100")
    monkeypatch.setattr("brainlayer.mcp.store_handler._retry_delay", 0.001)

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store", return_value=MagicMock()),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.queue_io.enqueue_store", side_effect=RuntimeError("force legacy pending queue")),
        patch("brainlayer.mcp.store_handler._get_pending_store_path", return_value=pending_path),
        patch("brainlayer.store.store_memory", side_effect=held_write_lock_store_memory),
    ):
        started = time.perf_counter()
        texts, structured = await _store(
            content="queued within busy budget",
            memory_type="note",
            project="test",
        )
        elapsed = time.perf_counter() - started

    assert elapsed < 0.22
    assert attempts <= 2
    assert structured["status"] == "DEFERRED"
    assert structured["deferred"]["action"] == "queued_for_replay"
    assert structured["deferred"]["queue_path"] == str(pending_path)
    assert any("queued" in item.text.lower() for item in texts)

    queued_event = json.loads(pending_path.read_text())
    assert queued_event["chunk_id"] == structured["chunk_id"]
    assert queued_event["content"] == "queued within busy budget"

    with (
        patch("brainlayer.mcp.store_handler._get_pending_store_path", return_value=pending_path),
        patch("brainlayer.store.store_memory", return_value={"id": structured["chunk_id"], "related": []}) as replay,
    ):
        flushed = _flush_pending_stores(MagicMock(), MagicMock())

    assert flushed == 1
    assert not pending_path.exists()
    assert replay.call_args.kwargs["chunk_id"] == structured["chunk_id"]
    assert replay.call_args.kwargs["content"] == "queued within busy budget"


@pytest.mark.asyncio
async def test_store_busy_budget_restores_timeout_between_concurrent_retries(monkeypatch):
    """Overlapping retry sleeps must not snapshot another store call's temporary timeout."""
    from brainlayer.mcp import store_handler

    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql):
            assert sql == "PRAGMA busy_timeout"
            return self

        def fetchone(self):
            return (self.conn.timeout_ms,)

    class FakeConn:
        def __init__(self):
            self.timeout_ms = 5000

        def cursor(self):
            return FakeCursor(self)

        def setbusytimeout(self, timeout_ms):
            self.timeout_ms = timeout_ms

    store = MagicMock()
    store.conn = FakeConn()

    def locked_store_memory(**kwargs):
        raise apsw.BusyError("database is locked")

    monkeypatch.setenv("BRAINLAYER_STORE_BUSY_BUDGET_MS", "1000")
    monkeypatch.setattr(store_handler, "_retry_delay", 0.02)
    monkeypatch.setattr(store_handler, "_RETRY_MAX_ATTEMPTS", 2)

    results = await asyncio.gather(
        store_handler._store_memory_with_retries(locked_store_memory, store=store),
        store_handler._store_memory_with_retries(locked_store_memory, store=store),
        return_exceptions=True,
    )

    assert all(isinstance(result, apsw.BusyError) for result in results)
    assert store.conn.timeout_ms == 5000


@pytest.mark.asyncio
async def test_store_busy_budget_bounds_store_memory_inner_retry_loop(tmp_path, monkeypatch):
    """The MCP budget must bound store_memory's internal BEGIN IMMEDIATE retry loop."""
    import brainlayer.store as store_module
    from brainlayer.mcp.store_handler import _store

    pending_path = tmp_path / "pending-stores.jsonl"
    real_sleep = time.sleep
    begin_attempts = 0

    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, *args):
            nonlocal begin_attempts
            if sql == "PRAGMA busy_timeout":
                return self
            if sql == "BEGIN IMMEDIATE":
                begin_attempts += 1
                real_sleep(self.conn.timeout_ms / 1000)
                raise apsw.BusyError("database is locked")
            raise AssertionError(f"unexpected SQL: {sql}")

        def fetchone(self):
            return (self.conn.timeout_ms,)

    class FakeConn:
        def __init__(self):
            self.timeout_ms = 5000

        def cursor(self):
            return FakeCursor(self)

        def setbusytimeout(self, timeout_ms):
            self.timeout_ms = timeout_ms

    store = MagicMock()
    store.conn = FakeConn()

    monkeypatch.setenv("BRAINLAYER_STORE_BUSY_BUDGET_MS", "80")
    monkeypatch.setattr("brainlayer.mcp.store_handler._retry_delay", 0.001)
    monkeypatch.setattr(store_module.time, "sleep", lambda delay: real_sleep(min(delay, 0.001)))

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store", return_value=store),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.queue_io.enqueue_store", side_effect=RuntimeError("force legacy pending queue")),
        patch("brainlayer.mcp.store_handler._get_pending_store_path", return_value=pending_path),
    ):
        started = time.perf_counter()
        _texts, structured = await _store(
            content="inner retries must respect mcp busy budget",
            memory_type="note",
            project="test",
        )
        elapsed = time.perf_counter() - started

    assert elapsed < 0.18
    assert begin_attempts <= 2
    assert structured["status"] == "DEFERRED"
    assert structured["deferred"]["action"] == "queued_for_replay"


@pytest.mark.asyncio
async def test_store_busy_budget_refreshes_after_timeout_lock_wait(monkeypatch):
    """Time spent waiting for the timeout lock must count against the store budget."""
    from brainlayer.mcp import store_handler

    real_sleep = time.sleep

    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql):
            assert sql == "PRAGMA busy_timeout"
            return self

        def fetchone(self):
            return (self.conn.timeout_ms,)

    class FakeConn:
        def __init__(self):
            self.timeout_ms = 5000

        def cursor(self):
            return FakeCursor(self)

        def setbusytimeout(self, timeout_ms):
            self.timeout_ms = timeout_ms

    store = MagicMock()
    store.conn = FakeConn()

    def locked_store_memory(**kwargs):
        real_sleep(store.conn.timeout_ms / 1000)
        raise apsw.BusyError("database is locked")

    monkeypatch.setenv("BRAINLAYER_STORE_BUSY_BUDGET_MS", "200")
    monkeypatch.setattr(store_handler, "_RETRY_MAX_ATTEMPTS", 1)

    store_handler._STORE_BUSY_TIMEOUT_LOCK.acquire()
    timer = threading.Timer(0.15, store_handler._STORE_BUSY_TIMEOUT_LOCK.release)
    timer.start()
    try:
        started = time.perf_counter()
        result = await store_handler._store_memory_with_retries(locked_store_memory, store=store)
    except apsw.BusyError as exc:
        result = exc
        elapsed = time.perf_counter() - started
    finally:
        timer.cancel()
        if store_handler._STORE_BUSY_TIMEOUT_LOCK.locked():
            store_handler._STORE_BUSY_TIMEOUT_LOCK.release()

    assert isinstance(result, apsw.BusyError)
    assert elapsed < 0.28


@pytest.mark.asyncio
async def test_store_busy_budget_covers_cold_vector_store_init(tmp_path, monkeypatch):
    """The store budget must include cold VectorStore init before queuing."""
    import brainlayer.store  # noqa: F401
    from brainlayer.mcp import _shared
    from brainlayer.mcp.store_handler import _store
    from brainlayer.vector_store import VectorStore

    pending_path = tmp_path / "pending-stores.jsonl"
    db_path = tmp_path / "busy-init.db"
    real_sleep = time.sleep
    init_attempts = 0

    def busy_init(self):
        nonlocal init_attempts
        init_attempts += 1
        real_sleep(0.075)
        raise apsw.BusyError("database is locked")

    monkeypatch.setenv("BRAINLAYER_STORE_BUSY_BUDGET_MS", "100")
    monkeypatch.setattr(_shared, "_vector_store", None)
    monkeypatch.setattr("brainlayer.paths.get_db_path", lambda: db_path)
    monkeypatch.setattr(VectorStore, "_INIT_MAX_RETRIES", 4)
    monkeypatch.setattr(VectorStore, "_INIT_BASE_DELAY", 0.001)
    monkeypatch.setattr(VectorStore, "_INIT_MAX_DELAY", 0.001)
    monkeypatch.setattr(VectorStore, "_acquire_writer_pidfile", lambda self: None)
    monkeypatch.setattr(VectorStore, "_release_writer_pidfile", lambda self: None)
    monkeypatch.setattr(VectorStore, "_init_db", busy_init)

    with (
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.queue_io.enqueue_store", side_effect=RuntimeError("force legacy pending queue")),
        patch("brainlayer.mcp.store_handler._get_pending_store_path", return_value=pending_path),
    ):
        started = time.perf_counter()
        _texts, structured = await _store(
            content="cold vector store init must respect busy budget",
            memory_type="note",
            project="test",
        )
        elapsed = time.perf_counter() - started

    assert elapsed < 0.22
    assert init_attempts <= 2
    assert structured["status"] == "DEFERRED"
    assert structured["deferred"]["action"] == "queued_for_replay"


@pytest.mark.asyncio
async def test_store_busy_budget_restores_cold_vector_store_write_timeout(tmp_path, monkeypatch):
    """A cold store init may be capped, but the singleton must return to normal writes."""
    import brainlayer.vector_store as vector_store_module
    from brainlayer.mcp import _shared
    from brainlayer.mcp.store_handler import _store
    from brainlayer.vector_store import VectorStore

    pending_path = tmp_path / "pending-stores.jsonl"
    db_path = tmp_path / "cold-timeout.db"

    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql):
            assert sql == "PRAGMA busy_timeout"
            return self

        def fetchone(self):
            return (self.conn.timeout_ms,)

    class FakeConn:
        def __init__(self):
            self.timeout_ms = vector_store_module._DEFAULT_BUSY_TIMEOUT_MS

        def cursor(self):
            return FakeCursor(self)

        def setbusytimeout(self, timeout_ms):
            self.timeout_ms = timeout_ms

        def readonly(self, _database):
            return False

        def close(self):
            pass

    def fake_init_db(self):
        self.conn = FakeConn()
        self.conn.setbusytimeout(vector_store_module._write_busy_timeout_ms())

    monkeypatch.setenv("BRAINLAYER_STORE_BUSY_BUDGET_MS", "100")
    monkeypatch.setattr(_shared, "_vector_store", None)
    monkeypatch.setattr("brainlayer.paths.get_db_path", lambda: db_path)
    monkeypatch.setattr(VectorStore, "_INIT_MAX_RETRIES", 1)
    monkeypatch.setattr(VectorStore, "_acquire_writer_pidfile", lambda self: None)
    monkeypatch.setattr(VectorStore, "_release_writer_pidfile", lambda self: None)
    monkeypatch.setattr(VectorStore, "_init_db", fake_init_db)
    monkeypatch.setattr("brainlayer.mcp.store_handler._retry_delay", 0.001)

    with (
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.queue_io.enqueue_store", side_effect=RuntimeError("force legacy pending queue")),
        patch("brainlayer.mcp.store_handler._get_pending_store_path", return_value=pending_path),
        patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("database is locked")),
    ):
        _texts, structured = await _store(
            content="cold vector store timeout must restore",
            memory_type="note",
            project="test",
        )

    assert structured["status"] == "DEFERRED"
    assert _shared._vector_store.conn.timeout_ms == vector_store_module._DEFAULT_BUSY_TIMEOUT_MS


def test_store_busy_budget_refreshes_cold_init_timeout_between_statements(tmp_path, monkeypatch):
    """Cold init must spend one absolute store budget across multiple DDL statements."""
    import brainlayer.vector_store as vector_store_module
    from brainlayer.vector_store import VectorStore, temporary_write_busy_timeout_ms

    class StopInit(Exception):
        pass

    fake_clock = {"now": 0.0}
    seen_timeouts = []

    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, *args):
            if self.conn.exec_trace is not None:
                bindings = args[0] if args else None
                result = self.conn.exec_trace(self, sql, bindings)
                if result is False:
                    raise apsw.ExecTraceAbort("exec trace aborted")
            seen_timeouts.append(self.conn.timeout_ms)
            fake_clock["now"] += 0.04
            if len(seen_timeouts) == 3:
                raise StopInit
            return self

        def fetchone(self):
            return None

        def __iter__(self):
            return iter(())

    class FakeConnection:
        def __init__(self, path):
            self.path = path
            self.timeout_ms = None
            self.exec_trace = None

        def setbusytimeout(self, timeout_ms):
            self.timeout_ms = timeout_ms

        def getexectrace(self):
            return self.exec_trace

        def setexectrace(self, trace):
            self.exec_trace = trace

        def enableloadextension(self, enabled):
            self.load_extension_enabled = enabled

        def loadextension(self, path):
            self.loaded_extension = path

        def cursor(self):
            return FakeCursor(self)

        def readonly(self, name):
            return False

    monkeypatch.setattr(vector_store_module.apsw, "Connection", FakeConnection)
    monkeypatch.setattr(vector_store_module.sqlite_vec, "loadable_path", lambda: "sqlite-vec")
    monkeypatch.setattr(vector_store_module.time, "monotonic", lambda: fake_clock["now"])

    store = object.__new__(VectorStore)
    store.db_path = tmp_path / "cold-init-deadline.db"

    with temporary_write_busy_timeout_ms(200, deadline=0.2), pytest.raises(StopInit):
        store._init_db()

    assert len(seen_timeouts) == 3
    assert seen_timeouts[1] < seen_timeouts[0]
    assert seen_timeouts[2] < seen_timeouts[1]


@pytest.mark.asyncio
async def test_store_busy_budget_bounds_singleton_init_lock_wait(tmp_path, monkeypatch):
    """Waiting for another cold VectorStore init must not exceed the store budget."""
    from brainlayer.mcp import _shared
    from brainlayer.mcp.store_handler import _store

    pending_path = tmp_path / "pending-stores.jsonl"
    db_path = tmp_path / "singleton-lock.db"

    class FakeVectorStore:
        def __init__(self, path):
            self.db_path = path
            self.conn = MagicMock()

    monkeypatch.setenv("BRAINLAYER_STORE_BUSY_BUDGET_MS", "80")
    monkeypatch.setattr(_shared, "_vector_store", None)
    monkeypatch.setattr("brainlayer.paths.get_db_path", lambda: db_path)
    monkeypatch.setattr("brainlayer.vector_store.VectorStore", FakeVectorStore)

    with (
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.queue_io.enqueue_store", side_effect=RuntimeError("force legacy pending queue")),
        patch("brainlayer.mcp.store_handler._get_pending_store_path", return_value=pending_path),
        patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("database is locked")),
    ):
        _shared._store_lock.acquire()
        timer = threading.Timer(0.24, _shared._store_lock.release)
        timer.start()
        try:
            started = time.perf_counter()
            _texts, structured = await _store(
                content="store init lock wait must respect busy budget",
                memory_type="note",
                project="test",
            )
            elapsed = time.perf_counter() - started
        finally:
            timer.cancel()
            if _shared._store_lock.locked():
                _shared._store_lock.release()

    assert elapsed < 0.18
    assert structured["status"] == "DEFERRED"
    assert structured["deferred"]["action"] == "queued_for_replay"
    assert _shared._vector_store is None


@pytest.mark.asyncio
async def test_store_busy_budget_disables_store_memory_internal_busy_sleep(monkeypatch):
    """Outer MCP retries should own busy backoff instead of holding the timeout lock."""
    from brainlayer.mcp import store_handler

    real_sleep = time.sleep

    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql):
            assert sql == "PRAGMA busy_timeout"
            return self

        def fetchone(self):
            return (self.conn.timeout_ms,)

    class FakeConn:
        def __init__(self):
            self.timeout_ms = 5000

        def cursor(self):
            return FakeCursor(self)

        def setbusytimeout(self, timeout_ms):
            self.timeout_ms = timeout_ms

    store = MagicMock()
    store.conn = FakeConn()
    retry_flags = []

    def internally_retrying_store_memory(**kwargs):
        retry_flags.append(kwargs.get("retry_on_busy"))
        if kwargs.get("retry_on_busy") is not False:
            real_sleep(0.12)
        raise apsw.BusyError("database is locked")

    monkeypatch.setenv("BRAINLAYER_STORE_BUSY_BUDGET_MS", "80")
    monkeypatch.setattr(store_handler, "_RETRY_MAX_ATTEMPTS", 1)

    started = time.perf_counter()
    with pytest.raises(apsw.BusyError):
        await store_handler._store_memory_with_retries(internally_retrying_store_memory, store=store)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.16
    assert retry_flags == [False]


@pytest.mark.asyncio
async def test_store_busy_budget_bounds_supersede_update(tmp_path, monkeypatch):
    """The same store budget must bound the post-store supersede write."""
    from brainlayer.mcp.store_handler import _store

    pending_path = tmp_path / "pending-stores.jsonl"
    real_sleep = time.sleep

    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql):
            assert sql == "PRAGMA busy_timeout"
            return self

        def fetchone(self):
            return (self.conn.timeout_ms,)

    class FakeConn:
        def __init__(self):
            self.timeout_ms = 250

        def cursor(self):
            return FakeCursor(self)

        def setbusytimeout(self, timeout_ms):
            self.timeout_ms = timeout_ms

    store = MagicMock()
    store.conn = FakeConn()

    def successful_store_memory(**kwargs):
        return {"id": kwargs["chunk_id"], "related": []}

    def locked_supersede(_old_chunk_id, _new_chunk_id):
        real_sleep(store.conn.timeout_ms / 1000)
        raise apsw.BusyError("database is locked")

    store.supersede_chunk.side_effect = locked_supersede

    monkeypatch.setenv("BRAINLAYER_STORE_BUSY_BUDGET_MS", "80")

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store", return_value=store),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.queue_io.enqueue_store", side_effect=RuntimeError("force legacy pending queue")),
        patch("brainlayer.mcp.store_handler._get_pending_store_path", return_value=pending_path),
        patch("brainlayer.store.store_memory", side_effect=successful_store_memory),
    ):
        started = time.perf_counter()
        _texts, structured = await _store(
            content="supersede update must respect busy budget",
            memory_type="note",
            project="test",
            supersedes="manual-old",
        )
        elapsed = time.perf_counter() - started

    assert elapsed < 0.18
    assert structured["status"] == "DEFERRED"
    assert structured["deferred"]["action"] == "queued_for_replay"
    assert json.loads(pending_path.read_text())["supersedes"] == "manual-old"


@pytest.mark.asyncio
async def test_store_preassigns_same_chunk_id_across_busy_retry(tmp_path, monkeypatch):
    """The MCP handler promises one chunk ID before the first write attempt."""
    from brainlayer.mcp.store_handler import _store

    queue_dir = tmp_path / "queue"
    seen_chunk_ids = []

    def flaky_store_memory(**kwargs):
        seen_chunk_ids.append(kwargs.get("chunk_id"))
        if len(seen_chunk_ids) == 1:
            raise apsw.BusyError("locked")
        return {"id": kwargs.get("chunk_id") or "store-generated-id", "related": []}

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.store.store_memory", side_effect=flaky_store_memory),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        monkeypatch.setattr("brainlayer.mcp.store_handler._retry_delay", 0.001)
        texts, structured = await _store(
            content="retry should keep promised id",
            memory_type="note",
            project="test",
        )

    assert len(seen_chunk_ids) == 2
    assert seen_chunk_ids[0] is not None
    assert seen_chunk_ids == [structured["chunk_id"], structured["chunk_id"]]
    assert structured["chunk_id"].startswith("manual-")
    assert any(structured["chunk_id"] in item.text for item in texts)
    assert not list(queue_dir.glob("mcp-*.jsonl"))


@pytest.mark.asyncio
async def test_busy_queue_fallback_flushes_promised_chunk_id(tmp_path, monkeypatch):
    """DB-busy queue and drain replay persist the exact caller-visible chunk ID."""
    from brainlayer.drain import drain_once
    from brainlayer.mcp.store_handler import _store
    from brainlayer.vector_store import VectorStore

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    store = VectorStore(db_path)
    store.close()
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("locked")),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        texts, structured = await _store(
            content="queued flush must preserve promised id",
            memory_type="note",
            project="test",
        )

    queued_files = list(queue_dir.glob("mcp-*.jsonl"))
    assert len(queued_files) == 1
    queued_event = json.loads(queued_files[0].read_text())
    promised_chunk_id = structured["chunk_id"]

    assert promised_chunk_id == queued_event["chunk_id"]
    assert promised_chunk_id.startswith("manual-")
    assert structured["queued"] is True
    assert any(promised_chunk_id in item.text for item in texts)

    assert drain_once(db_path=db_path, queue_dir=queue_dir, log_path=tmp_path / "drain.log") == 1

    conn = apsw.Connection(str(db_path))
    try:
        row = conn.cursor().execute("SELECT id, content FROM chunks WHERE id = ?", (promised_chunk_id,)).fetchone()
    finally:
        conn.close()

    assert row == (promised_chunk_id, "queued flush must preserve promised id")


@pytest.mark.asyncio
async def test_busy_queue_fallback_flushes_reservation_timestamp_and_project(tmp_path, monkeypatch):
    """Queued stores persist the reservation-time created_at and queued project."""
    from brainlayer.drain import drain_once
    from brainlayer.mcp.store_handler import _store
    from brainlayer.vector_store import VectorStore

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    store = VectorStore(db_path)
    store.close()
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="brainlayer"),
        patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("locked")),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        monkeypatch.setattr("brainlayer.mcp.store_handler._retry_delay", 0.001)
        texts, structured = await _store(
            content="queued flush must keep reservation metadata",
            memory_type="note",
            project="brainlayer",
        )

    queued_files = list(queue_dir.glob("mcp-*.jsonl"))
    assert len(queued_files) == 1
    queued_event = json.loads(queued_files[0].read_text())
    promised_chunk_id = structured["chunk_id"]
    reservation_created_at = queued_event["created_at"]

    assert promised_chunk_id == queued_event["chunk_id"]
    assert queued_event["project"] == "brainlayer"
    assert structured["queued"] is True
    assert any(promised_chunk_id in item.text for item in texts)

    assert drain_once(db_path=db_path, queue_dir=queue_dir, log_path=tmp_path / "drain.log") == 1

    conn = apsw.Connection(str(db_path))
    try:
        row = (
            conn.cursor()
            .execute(
                "SELECT created_at, project FROM chunks WHERE id = ?",
                (promised_chunk_id,),
            )
            .fetchone()
        )
    finally:
        conn.close()

    assert row == (reservation_created_at, "brainlayer")


@pytest.mark.asyncio
async def test_writer_in_use_error_queues_instead_of_erroring(tmp_path):
    """Writer pidfile contention queues the store instead of returning an MCP error."""
    from brainlayer.mcp.store_handler import _store
    from brainlayer.vector_store import WriterInUseError

    queue_dir = tmp_path / "queue"

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch(
            "brainlayer.store.store_memory",
            side_effect=WriterInUseError("another writer is using brainlayer.db (pid 123)"),
        ),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        texts, structured = await _store(
            content="queued under held writer pidfile",
            memory_type="note",
            project="test",
        )

    assert structured["queued"] is True
    assert structured["chunk_id"] != "queued"
    assert any("queued" in item.text.lower() for item in texts)
    assert len(list(queue_dir.glob("mcp-*.jsonl"))) == 1


@pytest.mark.asyncio
async def test_sqlite_prepare_lock_error_queues_instead_of_erroring(tmp_path):
    """Prepare-time transient SQLite lock failures are queueable store failures."""
    from brainlayer.mcp.store_handler import _store

    queue_dir = tmp_path / "queue"

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch(
            "brainlayer.store.store_memory",
            side_effect=RuntimeError("SQLite prepare failed: database schema is locked"),
        ),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        texts, structured = await _store(
            content="queued under prepare-time sqlite lock",
            memory_type="note",
            project="test",
        )

    assert structured["queued"] is True
    assert structured["chunk_id"] != "queued"
    assert any("queued" in item.text.lower() for item in texts)
    assert len(list(queue_dir.glob("mcp-*.jsonl"))) == 1


@pytest.mark.asyncio
async def test_busy_queue_fallback_preserves_supersedes(tmp_path):
    """DB-busy fallback queues supersedes so the deferred write keeps lifecycle intent."""
    from brainlayer.mcp.store_handler import _store

    queue_dir = tmp_path / "queue"

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("locked")),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        await _store(
            content="replacement memory",
            memory_type="note",
            project="test",
            supersedes="manual-old1234",
        )

    queued_files = list(queue_dir.glob("mcp-*.jsonl"))
    assert len(queued_files) == 1
    queued_event = json.loads(queued_files[0].read_text())
    assert queued_event["supersedes"] == "manual-old1234"


@pytest.mark.asyncio
async def test_arbitrated_queue_fallback_returns_queued_chunk_id(tmp_path, monkeypatch):
    """Arbitrated fallback also reports the queued event's real chunk ID."""
    from brainlayer.mcp.store_handler import _store

    monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
    with (
        patch("brainlayer.queue_io.get_queue_dir", return_value=tmp_path),
        patch("brainlayer.search_repo.clear_hybrid_search_cache"),
    ):
        texts, structured = await _store(content="arbitrated queued memory", memory_type="note", project="test")

    queued_files = list(tmp_path.glob("mcp-*.jsonl"))
    assert len(queued_files) == 1
    queued_event = json.loads(queued_files[0].read_text())
    expected_chunk_id = queued_event["chunk_id"]

    assert expected_chunk_id != "queued"
    assert structured["chunk_id"] == expected_chunk_id
    assert structured["queued"] is True
    assert structured["status"] == "DEFERRED"
    assert structured["deferred"]["reason"] == "ARBITRATED"
    assert structured["deferred"]["queue_path"] == str(queued_files[0])
    assert any(expected_chunk_id in item.text for item in texts)
