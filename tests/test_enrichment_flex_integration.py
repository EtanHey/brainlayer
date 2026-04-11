import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore


@pytest.mark.slow
def test_sustained_rate_no_contention(tmp_path, monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = VectorStore(tmp_path / "test.db")
    try:
        chunk_ids = []
        for index in range(100):
            result = store_memory(
                store=store,
                embed_fn=None,
                content=f"Decision {index}: keep writes serialized through one queue.",
                memory_type="decision",
                project="brainlayer",
                tags=["architecture"],
                importance=8,
            )
            chunk_ids.append(result["id"])

        call_times = []
        call_lock = threading.Lock()

        class FakeClient:
            class _Models:
                def generate_content(self, **kwargs):  # noqa: ARG002
                    with call_lock:
                        call_times.append(time.monotonic())
                    time.sleep(0.05)
                    return SimpleNamespace(
                        text=json.dumps(
                            {
                                "summary": "serialized enrichment",
                                "tags": ["architecture"],
                                "importance": 8,
                                "intent": "deciding",
                                "entities": [],
                            }
                        )
                    )

            def __init__(self):
                self.models = self._Models()

        sanitizer = SimpleNamespace(
            sanitize=lambda text, metadata=None: SimpleNamespace(sanitized=text, replacements=[], pii_detected=False),
        )

        monkeypatch.setattr(controller, "_get_gemini_client", lambda: FakeClient())
        monkeypatch.setattr(controller, "AUTO_ENRICH_ENABLED", True)
        monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: sanitizer))

        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(
                pool.map(lambda chunk_id: controller.enrich_single(store, chunk_id, max_retries=0), chunk_ids)
            )

        assert all(result is not None for result in results)
        assert len(call_times) == 100

        steady_state_times = call_times[10:]
        elapsed = steady_state_times[-1] - steady_state_times[0]
        observed_rate = len(steady_state_times) / elapsed if elapsed > 0 else float("inf")
        assert observed_rate <= 5.5

        enriched_rows = list(
            store.conn.cursor().execute(
                "SELECT COUNT(*) FROM chunks WHERE resolved_query = ?",
                ("serialized enrichment",),
            )
        )
        assert enriched_rows[0][0] == 0

        summaries = list(
            store.conn.cursor().execute(
                "SELECT COUNT(*) FROM chunks WHERE summary = ?",
                ("serialized enrichment",),
            )
        )
        assert summaries[0][0] == 100
    finally:
        store.close()
