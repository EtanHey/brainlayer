"""Keep-warm and targeted recovery regressions for the 22 Nov swap incident."""

import threading
from types import SimpleNamespace

import pytest

from brainlayer.brainbar_hybrid_helper import HybridSearchHelper


def test_keep_warm_repages_evicted_model_before_request(tmp_path, monkeypatch):
    import brainlayer.mcp._shared as shared

    resident = False
    touched = []

    def embed(_query):
        nonlocal resident
        resident = True

    class Cursor:
        def execute(self, sql, _bindings=()):
            touched.append(sql)
            return self

        def fetchone(self):
            return (1,)

    monkeypatch.setattr(shared, "_get_embedding_model", lambda: SimpleNamespace(embed_query=embed))
    monkeypatch.setattr(
        shared,
        "_get_search_vector_store",
        lambda: SimpleNamespace(_binary_index_available=True, _read_cursor=Cursor),
    )
    helper = HybridSearchHelper(tmp_path / "h.sock", tmp_path / "db", request_deadline_seconds=0.1)
    helper._ready = True
    helper._keep_warm_vector_id = "sample"
    helper._next_keep_warm = 0
    helper._keep_warm_if_due()

    def request(_):
        assert resident, "an evicted model would miss the request deadline"
        return {"ok": True}

    helper._handle_request = request
    assert helper._dispatch_with_deadline({"method": "brain_search"}) == {"ok": True}
    assert touched == ["SELECT embedding FROM chunk_vectors_binary WHERE chunk_id = ?"]
    assert helper.last_keep_warm_cost["cpu_seconds"] >= 0


def test_keep_warm_wedge_trips_short_deadline(tmp_path):
    expired = threading.Event()
    helper = HybridSearchHelper(tmp_path / "h.sock", tmp_path / "db", on_deadline_expired=lambda _: expired.set())
    helper.keep_warm_deadline_seconds = 0.02
    helper._ready, helper._next_keep_warm = True, 0
    helper._keep_warm_tick = lambda: expired.wait(2)
    helper._keep_warm_if_due()
    assert expired.is_set()


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0", "bad"])
def test_invalid_keep_warm_interval_uses_default(tmp_path, monkeypatch, value):
    monkeypatch.setenv("BRAINLAYER_HYBRID_KEEP_WARM_S", value)
    helper = HybridSearchHelper(tmp_path / "h.sock", tmp_path / "db")
    assert helper.keep_warm_interval_seconds == 60


def test_due_tick_waits_for_request_to_finish(tmp_path, monkeypatch):
    import socket
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    helper = HybridSearchHelper(Path("h.sock"), tmp_path / "db")
    entered, release, ticked = threading.Event(), threading.Event(), threading.Event()

    def warm():
        helper._ready, helper._next_keep_warm = True, float("inf")

    def request(_):
        helper._next_keep_warm = 0
        entered.set()
        assert release.wait(2)
        assert not ticked.is_set()
        return {"ok": True}

    helper.warm = warm
    helper._handle_request = request

    def tick():
        assert release.is_set()
        ticked.set()
        helper.stop()

    helper._keep_warm_tick = tick
    # Signal readiness at listen, without a sleep race or opening a second client.
    listening = threading.Event()
    original_listen = socket.socket.listen

    def listen(server, backlog):
        original_listen(server, backlog)
        listening.set()

    monkeypatch.setattr(socket.socket, "listen", listen)
    worker = threading.Thread(target=helper.serve_forever)
    worker.start()
    try:
        assert listening.wait(2)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2)
            client.connect(str(helper.socket_path))
            client.sendall(b'{"method":"brain_search"}\n')
            assert entered.wait(2)
            assert not ticked.is_set()
            release.set()
            assert b'"ok":true' in client.recv(4096)
        assert ticked.wait(2)
    finally:
        release.set()
        helper.stop()
        worker.join(3)
    assert not worker.is_alive()


def test_two_overruns_back_off_across_respawns_and_success_resets(tmp_path, monkeypatch):
    import brainlayer.brainbar_hybrid_helper as module

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(module.time, "monotonic", lambda: 1000.0)

    class ExpiredTimer:
        daemon = True

        def __init__(self, _seconds, callback):
            self.callback = callback

        def start(self):
            self.callback()

        def cancel(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(module.threading, "Timer", ExpiredTimer)

    def respawn():
        return HybridSearchHelper(tmp_path / "h.sock", tmp_path / "db", on_deadline_expired=lambda _: None)

    for count in (1, 2):
        helper = respawn()
        helper._ready, helper._next_keep_warm = True, 0
        helper._keep_warm_tick = lambda: None
        helper._keep_warm_if_due()
        assert helper._keep_warm_overruns == count

    replacement = respawn()
    assert replacement._next_keep_warm == 1300
    replacement._ready = True
    replacement._keep_warm_tick = lambda: pytest.fail("tick ran inside cooldown")
    replacement._keep_warm_if_due()
    replacement._handle_request = lambda _: {"ok": True}
    assert replacement._dispatch_with_deadline({"method": "status"}) == {"ok": True}
    # A successful tick after cooldown clears the persisted streak.
    monkeypatch.setattr(module.time, "monotonic", lambda: 1301.0)
    monkeypatch.setattr(ExpiredTimer, "start", lambda _: None)
    replacement._keep_warm_tick = lambda: None
    replacement._keep_warm_if_due()
    assert respawn()._keep_warm_overruns == 0


def test_vector_sample_reads_real_vec0_embedding_by_primary_key(tmp_path):
    import sqlite3

    import sqlite_vec

    with sqlite3.connect(":memory:") as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute(
            "CREATE VIRTUAL TABLE chunk_vectors_binary USING vec0(chunk_id TEXT PRIMARY KEY, embedding BIT[1024])"
        )
        conn.execute("INSERT INTO chunk_vectors_binary VALUES (?, vec_bit(?))", ("sample", bytes(128)))
        # Production table name, real vec0 schema; no rowid column exists here.
        helper = HybridSearchHelper(tmp_path / "h.sock", tmp_path / "db")
        store = SimpleNamespace(_binary_index_available=True, _read_cursor=conn.cursor)
        helper._touch_vector_sample(store)
        assert helper._keep_warm_vector_id == "sample"
        helper._touch_vector_sample(store)
        assert helper._keep_warm_vector_id == "sample"
