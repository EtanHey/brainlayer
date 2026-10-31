"""A real contended WAL, production queue/drain/hotlane, and independent receipts."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = [pytest.mark.recovery_oracle, pytest.mark.timeout(30, method="thread")]


@pytest.mark.parametrize("slow_acks", [1, 6])
def test_ack_p95_uses_enough_real_receipts_for_an_isolated_slow_ack(world, monkeypatch, slow_acks):
    from brainlayer import queue_io

    monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
    enqueue = queue_io.enqueue_store
    calls = 0

    def slow_initial_acks(**kwargs):
        nonlocal calls
        calls += 1
        if calls <= slow_acks:
            time.sleep(0.6)
        return enqueue(**kwargs)

    monkeypatch.setattr(queue_io, "enqueue_store", slow_initial_acks)
    for name in world.names:
        world.store(name)
    world.drain()
    world.assert_acknowledged()
    assert max(r["ack_ms"] for r in world.receipts) >= 600
    if slow_acks == 1:
        world.assert_latency()
    else:
        with pytest.raises(AssertionError, match="ack p95"):
            world.assert_latency()


@pytest.mark.parametrize("deferred_seeds", [False, True], ids=["default-seeds", "deferred-seeds"])
def test_lock_recovery(world, monkeypatch, deferred_seeds):
    with monkeypatch.context() as seed_setup:
        if deferred_seeds:
            seed_setup.setenv("BRAINLAYER_ARBITRATED", "1")
        world.store("otter")
        world.store("ibis")
    if deferred_seeds:
        assert [r["status"] for r in world.receipts] == ["DEFERRED", "DEFERRED"]
    # A seed store can exhaust its 25ms budget and acknowledge DEFERRED.
    # Materialize seeds before locking: drain cannot do so under the holder,
    # and an empty hotlane scan correctly skips the contended write entirely.
    world.drain()
    assert world.writer.conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (2,)
    world.writer.conn.execute("DELETE FROM chunk_vectors")
    holder = world.connection()
    holder.execute("BEGIN IMMEDIATE")
    started = time.monotonic()
    try:
        world.store(world.names[0])
        with ThreadPoolExecutor(max_workers=2) as workers:
            draining = workers.submit(world.drain)
            embedding = workers.submit(world.hotlane)
            batch_started = started
            for index, name in enumerate(world.names[1:], start=2):
                world.store(name)
                # Preserve the original 3s producer bound per six-store batch
                # as the p95 workload grows from six to 100 contended stores.
                if index % 6 == 0 or index == len(world.names):
                    assert time.monotonic() - batch_started < 3
                    batch_started = time.monotonic()
            assert draining.result(timeout=3) == 0
            assert embedding.result(timeout=3) == 0
            assert world.hotlane_busy == 1, "hotlane did not contend for the lock"
            time.sleep(max(0, 0.3 - (time.monotonic() - started)))
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    # Only stores made under the holder count; deferred seeds would satisfy this vacuously.
    assert any(r["status"] == "DEFERRED" for r in world.receipts[2:])
    from .faults import catch_up

    convergence = catch_up(world, timeout=5)
    world.assert_roundtrip()
    world.assert_acknowledged()
    metrics = {**world.assert_latency(), **convergence}
    print("recovery_oracle_receipt=" + json.dumps({"fault": "lock", "seed": world.seed, **metrics}, sort_keys=True))


@pytest.mark.parametrize("broken", ["skip_replay", "lose_vector", "lose_fts", "duplicate_row", "slow_ack", "long_lock"])
def test_oracles_reject_broken_variants(world, monkeypatch, broken):
    monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
    world.store("otter")
    if broken == "slow_ack":
        from brainlayer import queue_io

        original = queue_io.enqueue_store

        def slow(**kwargs):
            time.sleep(0.6)
            return original(**kwargs)

        monkeypatch.setattr(queue_io, "enqueue_store", slow)
        world.store("ibis")
    if broken == "long_lock":
        from brainlayer import drain

        original = drain._apply_event

        def slow(conn, event):
            time.sleep(0.6)
            return original(conn, event)

        monkeypatch.setattr(drain, "_apply_event", slow)
    if broken != "skip_replay":
        world.drain()
        world.hotlane()
    if broken == "lose_vector":
        world.writer.conn.execute("DELETE FROM chunk_vectors")
    if broken == "lose_fts":
        world.writer.conn.execute("DELETE FROM chunks_fts")
    if broken == "duplicate_row":
        world.writer.conn.execute(
            "INSERT INTO chunks (id, content, metadata, source_file, source) "
            "SELECT 'broken-copy', content, metadata, source_file, source FROM chunks LIMIT 1"
        )
    oracle = world.assert_latency if broken in {"slow_ack", "long_lock"} else world.assert_acknowledged
    messages = {
        "skip_replay": "missing acknowledged identity",
        "lose_vector": "vector invisible",
        "lose_fts": "FTS invisible",
        "duplicate_row": "duplicate rows",
        "slow_ack": "ack p95",
        "long_lock": "drain lock held",
    }
    with pytest.raises(AssertionError, match=messages[broken]):
        if broken == "lose_vector":
            world.assert_roundtrip()
        else:
            oracle()


@pytest.mark.parametrize("path", ["/tmp/brainbar.sock", "/Users/any/.local/share/brainlayer/db", "../escape"])
def test_paths_fail_closed(world, path):
    with pytest.raises(ValueError, match="outside disposable universe"):
        world.guard(path)


def test_environment_override_and_symlink_cannot_escape(world, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_DB", "/tmp/brainbar.sock")
    with pytest.raises(ValueError):
        world.drain()
    link = world.root / "escape"
    link.symlink_to(world.root.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        world.guard(link / "outside.db")


@pytest.mark.parametrize("delayed_writer", [False, True], ids=["default-writer", "delayed-writer"])
def test_all_success_outcomes_are_accounted_for(world, monkeypatch, delayed_writer):
    from brainlayer import runtime_store
    from brainlayer.mcp import store_handler

    if delayed_writer:
        original_open = runtime_store.open_writer_store
        join_started = threading.Event()
        delayed = False

        def open_writer(*args, **kwargs):
            nonlocal delayed
            if not delayed:
                delayed = True
                # Start the delay only once the fixture is joining, so it always
                # outlasts the former 3s join regardless of acknowledgement time.
                assert join_started.wait(timeout=30)
                time.sleep(3.2)
            return original_open(*args, **kwargs)

        monkeypatch.setattr(runtime_store, "open_writer_store", open_writer)

    first = world.store("otter", before_join=join_started.set if delayed_writer else None)
    if delayed_writer:
        assert delayed, "precondition: store must start its background writer"
    world.store("otter")
    with monkeypatch.context() as merge:
        merge.setattr(store_handler, "_new_manual_chunk_id", lambda: first["chunk_id"])
        world.store("otter", content=world.text("otter") + " Additional details about the river habitat.")
    monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
    world.store("ibis")
    world.drain()
    world.hotlane()
    assert {r["status"] for r in world.receipts} == {"STORED", "DUPLICATE", "MERGED", "DEFERRED"}
    world.assert_acknowledged()
    world.assert_latency()
