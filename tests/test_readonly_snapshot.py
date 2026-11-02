import threading
from contextlib import closing
from types import SimpleNamespace

import apsw
import pytest
import sqlite_vec

from brainlayer import readonly_snapshot as snap


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "source.db"
    with closing(apsw.Connection(str(path))) as c:
        c.enable_load_extension(True)
        c.load_extension(sqlite_vec.loadable_path())
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE chunks(id); INSERT INTO chunks VALUES (1)")
        c.execute("CREATE VIRTUAL TABLE chunk_vectors USING vec0(embedding float[4])")
        c.execute("INSERT INTO chunk_vectors(embedding) VALUES (?)", (sqlite_vec.serialize_float32([1, 2, 3, 4]),))
        c.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(content)")
        c.execute("INSERT INTO chunks_fts VALUES ('memory')")
    return path


def take(db, **kw):
    return snap.take_snapshot(
        db, db.parent / "out.db", **({"deadline_s": 10, "wal_budget_bytes": 10**8, "min_free_bytes": 0} | kw)
    )


def clean(db):
    assert not (db.parent / "out.db").exists()
    assert not list(db.parent.glob(".readonly-snapshot-*"))
    assert not any(t.name == "snapshot-budget" for t in threading.enumerate())
    with closing(apsw.Connection(str(db))) as c:
        assert c.execute("PRAGMA wal_checkpoint(TRUNCATE)").get[0] == 0


def test_snapshot_during_open_writer(db):
    c = apsw.Connection(str(db))
    try:
        c.execute("BEGIN IMMEDIATE; INSERT INTO chunks VALUES (2)")
        result = take(db)
        assert result.reason is None, result
        output = db.parent / "out.db"
        assert output.stat().st_mode & 0o777 == 0o600
        assert result.bytes_written == output.stat().st_size
        assert snap.snapshot_counts(output, quick_check=True) == dict(chunks=1, chunk_vectors=1, chunks_fts=1)
        c.execute("COMMIT")
        assert snap.snapshot_counts(db)["chunks"] == 2
    finally:
        c.close()


@pytest.mark.parametrize("kw,reason", [({"deadline_s": 0}, "deadline"), ({"wal_budget_bytes": 0}, "wal_budget")])
def test_deadline_and_wal(db, kw, reason):
    with closing(apsw.Connection(str(db))) as writer:
        writer.execute("INSERT INTO chunks VALUES (2)")
        result = take(db, **kw)
    assert result.reason == reason
    clean(db)
    assert take(db).reason is None


@pytest.mark.parametrize("call,reason", [(1, "source_free_space"), (2, "target_free_space")])
def test_free_space(db, monkeypatch, call, reason):
    calls = iter(range(1, 20))
    monkeypatch.setattr(
        snap, "measure_disk_space", lambda *a, **k: SimpleNamespace(free_now_bytes=0 if next(calls) == call else 100)
    )
    assert take(db, min_free_bytes=1).reason == reason
    clean(db)


def test_existing_target(db):
    target = db.parent / "out.db"
    target.write_bytes(b"preserve")
    assert take(db).reason == "target_exists"
    assert target.read_bytes() == b"preserve"


def test_publish_race(db, monkeypatch):
    original = snap.os.link

    def race(source, target, **kw):
        target.write_bytes(b"peer")
        original(source, target, **kw)

    monkeypatch.setattr(snap.os, "link", race)
    assert take(db).reason == "target_exists"
    assert (db.parent / "out.db").read_bytes() == b"peer"
    assert not list(db.parent.glob(".readonly-snapshot-*"))


@pytest.mark.parametrize(
    "reason", "deadline monitor_error output_identity source_identity io_error sqlite_error".split()
)
def test_mid_copy_abort(db, monkeypatch, reason):
    original = snap._open_reader

    class Reader:
        c = original(db)

        def __getattr__(self, name):
            return getattr(self.c, name)

        def execute(self, sql, *args):
            if reason == "deadline":
                monkeypatch.setattr(snap, "time", SimpleNamespace(monotonic=lambda: 10**20))
            elif reason == "monitor_error":

                def failed(*a, **k):
                    raise OSError("injected statfs error")

                monkeypatch.setattr(snap, "measure_disk_space", failed)
                threading.Event().wait(0.1)
            elif reason == "sqlite_error":
                self.c.set_progress_handler(lambda: True, 1, id="inject")
            elif reason == "output_identity":
                value = self.c.execute(sql, *args)
                attempt = next(db.parent.glob(".readonly-snapshot-*/snapshot.db"))
                attempt.unlink()
                attempt.write_bytes(b"peer")
                return value
            elif reason == "source_identity":
                value = self.c.execute(sql, *args)
                db.rename(db.with_suffix(".old"))
                apsw.Connection(str(db)).close()
                return value
            else:
                raise OSError("injected I/O error")
            return self.c.execute(sql, *args)

    monkeypatch.setattr(snap, "_open_reader", lambda p: Reader())
    result = take(db)
    assert result.reason == reason, result
    assert not (db.parent / "out.db").exists()
    if reason == "output_identity":
        assert next(db.parent.glob(".readonly-snapshot-*/snapshot.db")).read_bytes() == b"peer"
        assert "retained unconfirmed" in result.detail
    else:
        clean(db)


def test_cleanup_failure_rolls_back_publication(db, monkeypatch):
    original = snap._sync_directory

    def fail(path):
        if (db.parent / "out.db").exists() and not list(db.parent.glob(".readonly-snapshot-*")):
            raise OSError("injected cleanup fsync failure")
        original(path)

    monkeypatch.setattr(snap, "_sync_directory", fail)
    assert take(db).reason == "cleanup_error"
    clean(db)
