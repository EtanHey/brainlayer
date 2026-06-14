import importlib.util
import json
import os
import sqlite3
import sys
import time
from pathlib import Path


def _load_benchmark_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "hot_currentness_benchmark.py"
    spec = importlib.util.spec_from_file_location("hot_currentness_benchmark_test", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_embed(text: str) -> list[float]:
    seed = sum(ord(char) for char in text) % 17
    return [float(seed + idx) / 1000.0 for idx in range(1024)]


def _has_vector(db_path: Path, chunk_id: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        return (
            conn.execute(
                "SELECT COUNT(*) FROM chunk_vectors_rowids WHERE id = ?",
                (chunk_id,),
            ).fetchone()[0]
            == 1
        )


def test_store_probe_hot_embeds_fresh_chunk_and_still_drains_old_backlog(tmp_path):
    benchmark = _load_benchmark_module()
    from brainlayer.store import store_memory
    from brainlayer.vector_store import VectorStore

    db_path = tmp_path / "hot-currentness.db"
    store = VectorStore(db_path)
    old_ids = []
    try:
        for index in range(3):
            old = store_memory(
                store=store,
                embed_fn=None,
                content=f"old backlog currentness drain marker {index}",
                memory_type="note",
                project="brainlayer-test",
                created_at=f"2026-06-13T00:0{index}:00+00:00",
            )
            old_ids.append(old["id"])
    finally:
        store.close()

    fresh_content = "fresh vector currentness marker should hot embed before backlog"

    def lane_embed(text: str) -> list[float]:
        if text == fresh_content:
            return [1.0] + [0.0] * 1023
        return [0.0, 1.0] + [0.0] * 1022

    before_pending = benchmark.embedding_backlog_metrics(db_path=db_path)["pending_manual_mcp_embeddings"]
    result = benchmark.run_store_probe(
        db_path=db_path,
        content=fresh_content,
        project="brainlayer-test",
        embed_fn=lane_embed,
        embed_after_store=True,
        timeout_s=3.0,
        poll_interval_s=0.01,
    )
    after_pending = benchmark.embedding_backlog_metrics(db_path=db_path)["pending_manual_mcp_embeddings"]

    assert result["hybrid_vector_arm_visible_with_embedding"] is True
    assert result["hybrid_vector_arm_visibility_latency_ms"] >= 0
    assert _has_vector(db_path, result["chunk_id"])
    assert _has_vector(db_path, old_ids[0])
    assert after_pending < before_pending


def test_store_probe_separates_durable_lexical_embedding_and_hybrid_visibility(tmp_path):
    benchmark = _load_benchmark_module()
    db_path = tmp_path / "hot-currentness.db"
    marker = "hcprobealpha unique currentness marker"

    result = benchmark.run_store_probe(
        db_path=db_path,
        content=f"{marker} should become hybrid visible after embedding",
        project="brainlayer-test",
        embed_fn=_fake_embed,
        embed_after_store=True,
        timeout_s=3.0,
        poll_interval_s=0.01,
    )

    assert result["chunk_id"].startswith("manual-")
    assert result["brain_store_call_latency_ms"] >= 0
    assert result["durable_row_latency_ms"] >= 0
    assert result["fts_visibility_latency_ms"] >= 0
    assert result["trigram_visibility_latency_ms"] >= 0
    assert result["embedding_availability_latency_ms"] >= 0
    assert result["hybrid_rrf_visibility_latency_ms"] >= 0
    assert result["hybrid_rrf_visible_with_embedding"] is True


def test_queue_metrics_report_depth_oldest_age_and_throughput(tmp_path):
    benchmark = _load_benchmark_module()
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    (queue_dir / "mcp-test.jsonl").write_text(
        json.dumps(
            {
                "kind": "store_memory",
                "content": "queued content",
                "queued_at": 100.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (queue_dir / "bad.jsonl").write_text("{not-json}\n", encoding="utf-8")

    metrics = benchmark.queue_metrics(queue_dir=queue_dir, now=110.0)

    assert metrics == {
        "queue_depth_files": 2,
        "queue_depth_events": 1,
        "oldest_queued_age_s": 10.0,
    }
    assert benchmark.drain_throughput(events=5, elapsed_s=2.0) == 2.5


def test_store_probe_measures_hybrid_latency_from_original_write(tmp_path):
    benchmark = _load_benchmark_module()
    db_path = tmp_path / "hot-currentness.db"

    def slow_embed(text: str) -> list[float]:
        time.sleep(0.05)
        return _fake_embed(text)

    result = benchmark.run_store_probe(
        db_path=db_path,
        content="hybrid latency should include embedding work from write time",
        project="brainlayer-test",
        embed_fn=slow_embed,
        embed_after_store=True,
        timeout_s=3.0,
        poll_interval_s=0.01,
    )

    assert result["hybrid_rrf_visible_with_embedding"] is True
    assert result["hybrid_rrf_visibility_latency_ms"] >= 100.0


def test_queue_drain_scenario_uses_isolated_queue_when_parent_has_backlog(tmp_path):
    benchmark = _load_benchmark_module()
    db_path = tmp_path / "hot-currentness.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    backlog = {
        "kind": "store_memory",
        "chunk_id": "manual-backlog",
        "content": "older unrelated backlog should not be drained by benchmark",
        "project": "brainlayer-test",
        "source": "mcp",
    }
    (queue_dir / "aaa-backlog.jsonl").write_text(json.dumps(backlog) + "\n", encoding="utf-8")

    result = benchmark.run_queue_drain_scenario(
        db_path=db_path,
        queue_dir=queue_dir,
        project="brainlayer-test",
        count=1,
        embed_fn=_fake_embed,
        label="queue-drain",
    )

    assert result["durable_rows"] == 1
    assert result["embedded_rows"] == 1
    assert result["drained_events"] == 1
    assert (queue_dir / "aaa-backlog.jsonl").exists()
    assert Path(result["benchmark_queue_dir"]).parent == queue_dir
    with sqlite3.connect(db_path) as conn:
        source = conn.execute("SELECT source FROM chunks WHERE id = ?", (result["queued_ids"][0],)).fetchone()
    assert source == ("mcp",)


def test_queue_drain_scenario_can_disable_real_drain_embeddings(tmp_path, monkeypatch):
    benchmark = _load_benchmark_module()
    db_path = tmp_path / "hot-currentness.db"
    queue_dir = tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "1")

    def fail_if_loaded():
        raise AssertionError("real embedding model should not be loaded")

    import brainlayer.drain as drain

    monkeypatch.setattr(drain, "_default_embed_fn", fail_if_loaded)

    result = benchmark.run_queue_drain_scenario(
        db_path=db_path,
        queue_dir=queue_dir,
        project="brainlayer-test",
        count=1,
        embed_fn=None,
        embed_after_drain=False,
        label="queue-drain",
    )

    assert result["durable_rows"] == 1
    assert result["embedded_rows"] == 0


def test_queue_drain_scenario_enables_embeddings_even_when_env_disables_drain(tmp_path, monkeypatch):
    benchmark = _load_benchmark_module()
    db_path = tmp_path / "hot-currentness.db"
    queue_dir = tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    result = benchmark.run_queue_drain_scenario(
        db_path=db_path,
        queue_dir=queue_dir,
        project="brainlayer-test",
        count=1,
        embed_fn=_fake_embed,
        embed_after_drain=True,
        label="queue-drain",
    )

    assert result["durable_rows"] == 1
    assert result["embedded_rows"] == 1
    assert os.environ["BRAINLAYER_DRAIN_EMBED"] == "0"


def test_main_accepts_fresh_database_for_backlog_metrics(tmp_path, monkeypatch, capsys):
    benchmark = _load_benchmark_module()
    db_path = tmp_path / "fresh.db"
    queue_dir = tmp_path / "queue"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hot_currentness_benchmark.py",
            "--db",
            str(db_path),
            "--queue-dir",
            str(queue_dir),
            "--scenario",
            "store-no-embed",
            "--count",
            "0",
            "--timeout-s",
            "0.01",
            "--no-real-embed",
        ],
    )

    assert benchmark.main() == 0
    output = json.loads(capsys.readouterr().out)

    assert db_path.exists()
    assert output["embedding_backlog_before"]["pending_manual_mcp_embeddings"] == 0
