import importlib.util
import json
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

    metrics = benchmark.queue_metrics(queue_dir=queue_dir, now=110.0)

    assert metrics == {
        "queue_depth_files": 1,
        "queue_depth_events": 1,
        "oldest_queued_age_s": 10.0,
    }
    assert benchmark.drain_throughput(events=5, elapsed_s=2.0) == 2.5
