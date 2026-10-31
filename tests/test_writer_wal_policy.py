"""Connection-local writer policy, exercised through each real entrypoint."""

import sqlite3

import apsw
import pytest

from brainlayer import decay_job, drain, maintenance
from brainlayer.runtime_store import open_writer_store
from brainlayer.vector_store import VectorStore


def _policy(conn):
    return tuple(conn.execute(f"PRAGMA {name}").fetchone()[0] for name in ("journal_size_limit", "wal_autocheckpoint"))


@pytest.mark.parametrize("entrypoint", ["vector", "runtime", "drain", "decay", "maintenance"])
@pytest.mark.parametrize(
    "limit,pages,expected",
    [
        (None, None, (256_000_000, 10_000)),
        ("65536", "17", (65_536, 17)),
        ("invalid", "invalid", (256_000_000, 10_000)),
        ("0", "0", (-1, 0)),
    ],
)
def test_each_writer_sets_policy_on_open_and_reopen(tmp_path, monkeypatch, entrypoint, limit, pages, expected):
    monkeypatch.setenv("BRAINLAYER_RUNTIME_STORE", "runtime")
    for name, value in (("BRAINLAYER_WAL_SIZE_LIMIT_BYTES", limit), ("BRAINLAYER_WAL_AUTOCHECKPOINT", pages)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    path = tmp_path / "writer.db"
    VectorStore(path).close()

    for _ in range(2):
        if entrypoint in {"vector", "runtime", "drain"}:
            opener = {"vector": VectorStore, "runtime": open_writer_store, "drain": drain._open_connection}[entrypoint]
            opened = opener(path)
            try:
                assert _policy(opened if entrypoint == "drain" else opened.conn) == expected
            finally:
                opened.close()
        elif entrypoint == "decay":
            connections = []
            real_connection = apsw.Connection

            def capture(*args, **kwargs):
                conn = real_connection(*args, **kwargs)
                connections.append(conn)
                return conn

            with monkeypatch.context() as patch:
                patch.setattr(decay_job.apsw, "Connection", capture)
                try:
                    decay_job.run_decay_job(path)
                    assert len(connections) == 1
                    assert _policy(connections[0]) == expected
                finally:
                    for conn in connections:
                        conn.close()
        else:
            observed = []
            real_connect = sqlite3.connect

            class ObservedConnection(sqlite3.Connection):
                def close(self):
                    observed.append(_policy(self))
                    super().close()

            def capture_vacuum(*args, **kwargs):
                return real_connect(*args, factory=ObservedConnection, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(maintenance.sqlite3, "connect", capture_vacuum)
                maintenance._vacuum(path)
            assert observed == [expected]


def test_readonly_connections_do_not_receive_writer_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_WAL_SIZE_LIMIT_BYTES", "65536")
    monkeypatch.setenv("BRAINLAYER_WAL_AUTOCHECKPOINT", "17")
    path = tmp_path / "readonly.db"
    VectorStore(path).close()
    reader = VectorStore(path, readonly=True)
    try:
        for conn in (reader.conn, reader._get_read_conn()):
            assert _policy(conn) == (-1, 0)
            assert conn.execute("PRAGMA query_only").fetchone() == (1,)
    finally:
        reader.close()


def test_wal_can_exceed_limit_while_pinned_then_recycles(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_RUNTIME_STORE", "runtime")
    monkeypatch.delenv("BRAINLAYER_WAL_SIZE_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("BRAINLAYER_WAL_AUTOCHECKPOINT", raising=False)
    path = tmp_path / "pinned.db"
    VectorStore(path).close()
    store = open_writer_store(path)
    writer = store.conn
    reader = apsw.Connection(str(path), flags=apsw.SQLITE_OPEN_READONLY)
    limit = 256_000_000
    try:
        writer.execute("CREATE TABLE payload(data BLOB)")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        reader.execute("BEGIN")
        assert reader.execute("SELECT COUNT(*) FROM payload").fetchone() == (0,)
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO payload VALUES (zeroblob(?))", (limit + 1_000_000,))
        writer.execute("COMMIT")
        wal = path.with_name(path.name + "-wal")
        assert wal.stat().st_size > limit
        writer.setbusytimeout(0)
        busy, frames, backfilled = writer.execute("PRAGMA wal_checkpoint(RESTART)").fetchone()
        assert busy == 1 and frames > backfilled
        assert reader.execute("SELECT COUNT(*) FROM payload").fetchone() == (0,)
        reader.execute("ROLLBACK")
        assert writer.execute("PRAGMA wal_checkpoint(RESTART)").fetchone()[0] == 0
        # RESTART makes the next write eligible to reset/recycle the WAL. No TRUNCATE:
        # the retained-file limit itself must reclaim the oversized allocation.
        writer.execute("INSERT INTO payload VALUES (zeroblob(1))")
        assert wal.stat().st_size <= limit
        assert reader.execute("SELECT length(data) FROM payload ORDER BY rowid").fetchall() == [
            (limit + 1_000_000,),
            (1,),
        ]
    finally:
        reader.close()
        store.close()
