"""A real contended WAL, production queue/drain/hotlane, and independent receipts."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = [pytest.mark.recovery_oracle, pytest.mark.timeout(30)]


def test_lock_recovery(world):
    world.store("otter")
    world.store("ibis")
    world.writer.conn.execute("DELETE FROM chunk_vectors")
    holder = world.connection()
    holder.execute("BEGIN IMMEDIATE")
    started = time.monotonic()
    try:
        world.store(world.names[0])
        with ThreadPoolExecutor(max_workers=2) as workers:
            draining = workers.submit(world.drain)
            embedding = workers.submit(world.hotlane)
            for name in world.names[1:]:
                world.store(name)
            assert draining.result(timeout=3) == 0
            assert embedding.result(timeout=3) == 0
            assert world.hotlane_busy == 1, "hotlane did not contend for the lock"
            assert time.monotonic() - started < 3
            time.sleep(max(0, 0.3 - (time.monotonic() - started)))
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert any(r["status"] == "DEFERRED" for r in world.receipts)
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


def test_all_success_outcomes_are_accounted_for(world, monkeypatch):
    from brainlayer.mcp import store_handler

    first = world.store("otter")
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
