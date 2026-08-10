import importlib.util
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "hybrid_backfill_sample.py"
    spec = importlib.util.spec_from_file_location("hybrid_backfill_sample", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diverse_sample_includes_small_strata_before_proportional_fill():
    mod = _load_module()
    rows = [
        {"id": f"user-{i}", "content_type": "user_message", "source": "mcp", "char_count": 120}
        for i in range(20)
    ]
    rows += [
        {"id": f"assistant-{i}", "content_type": "assistant_text", "source": "realtime", "char_count": 900}
        for i in range(10)
    ]
    rows += [{"id": "rare-code", "content_type": "ai_code", "source": "digest", "char_count": 1500}]

    sample = mod.diverse_sample(rows, limit=10, seed=7)

    keys = {(row["content_type"], row["source"], mod.char_bucket(row["char_count"])) for row in sample}
    assert ("ai_code", "digest", "long") in keys
    assert ("assistant_text", "realtime", "medium") in keys
    assert ("user_message", "mcp", "short") in keys
    assert len(sample) == 10


def test_chunked_preserves_order_and_sizes():
    mod = _load_module()
    chunks = list(mod.chunked(range(11), 5))

    assert chunks == [list(range(5)), list(range(5, 10)), [10]]
