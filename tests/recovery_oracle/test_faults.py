"""Controlled process cuts and provider outages must converge without lost receipts."""

import json

import pytest

from .faults import catch_up, cut_worker, index_batch

pytestmark = [pytest.mark.recovery_oracle, pytest.mark.timeout(30, method="thread")]


@pytest.mark.parametrize("fault", ["index", "encoder_raise", "encoder_stall", "drain_restart"])
def test_fault_recovery(world, monkeypatch, fault):
    monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
    for name in world.names:
        world.store(name)
    if fault == "index":
        cut_worker(world, "index")
        assert world.writer.conn.execute("SELECT COUNT(*) FROM chunks WHERE source='oracle-index'").fetchone() == (1,)
        index_batch(world)
    elif fault == "encoder_raise":

        def failed(_text):
            raise RuntimeError("seeded encoder unavailable")

        world.drain(embed_fn=failed)
        assert world.hotlane(embed_fn=failed) == 0
        assert world.writer.conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone() == (0,)
    else:
        cut_worker(world, fault)
        if fault == "encoder_stall":
            assert world.writer.conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (0,)
        else:
            assert world.writer.conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (1,)
        assert list(world.queue.glob("*.jsonl")), "cut did not retain replay work"
    metrics = catch_up(world, timeout=5)
    world.assert_roundtrip()
    world.assert_acknowledged(extra_ids={"indexa", "indexb"} if fault == "index" else ())
    print(
        "recovery_oracle_receipt="
        + json.dumps({"fault": fault, "seed": world.seed, **metrics, **world.assert_latency()}, sort_keys=True)
    )


@pytest.mark.parametrize("broken", ["skip_replay", "skip_hotlane"])
def test_catch_up_oracle_rejects_stuck_backlog(world, monkeypatch, broken):
    monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
    world.store("otter")
    if broken == "skip_replay":
        monkeypatch.setattr(world, "drain", lambda: 0)
    else:
        monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
        monkeypatch.setattr(world, "hotlane", lambda: 0)
    with pytest.raises(AssertionError, match="backlog did not converge"):
        catch_up(world, timeout=0.2)


@pytest.mark.parametrize("missing", ["transactions", "statements", "BEGIN IMMEDIATE", "COMMIT"])
def test_lock_measurement_rejects_missing_transaction_evidence(world, monkeypatch, missing):
    monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
    world.store("otter")
    world.drain()
    path = world.guard(world.root / "writer.jsonl")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    committed = [e for e in events if e.get("event") == "txn_finished" and e.get("outcome") == "commit"]
    assert committed, "negative control requires real committed transaction evidence"
    assert any(e.get("event") == "ingest_committed" for e in events)
    if missing == "transactions":
        events = [e for e in events if e.get("event") != "txn_finished"]
    else:
        for event in committed:
            if missing == "statements":
                del event["statements"]
            else:
                event["statements"] = [s for s in event["statements"] if s["normalized_sql"] != missing]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(AssertionError, match="missing.*drain"):
        world.lock_holds()
