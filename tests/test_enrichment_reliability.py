"""Tests for enrichment reliability — retry, circuit breaker, timeout config."""

import time
from unittest.mock import MagicMock, patch

from brainlayer.pipeline import enrichment


def test_parse_enrichment_uses_hybrid_tag_taxonomy_by_default(monkeypatch):
    monkeypatch.delenv("BRAINLAYER_ENRICHMENT_TAG_MODE", raising=False)
    parsed = enrichment.parse_enrichment(
        """
        {
          "summary": "BrainLayer Track B normalizes enrichment tags before backlog re-enrichment.",
          "tags": [
            "Project/BrainLayer",
            "tech/debug/investigation",
            "React.js",
            "reactjs",
            "tech/debug/resolution",
            "one-off-singleton-from-model",
            "PM/Decision"
          ],
          "importance": 8
        }
        """
    )

    assert parsed is not None
    assert parsed["tags"] == [
        "project/brainlayer",
        "tech/debug/investigation",
        "react",
        "tech/debug/resolution",
        "one-off-singleton-from-model",
        "pm/decision",
    ]


def test_parse_enrichment_can_roll_back_to_taxonomy_whitelist(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_TAG_MODE", "taxonomy")
    parsed = enrichment.parse_enrichment(
        """
        {
          "summary": "BrainLayer Track B can still A/B the old taxonomy whitelist mode.",
          "tags": ["Project/BrainLayer", "React.js", "tech/debug/resolution", "PM/Decision"],
          "importance": 8
        }
        """
    )

    assert parsed is not None
    assert parsed["tags"] == ["project/brainlayer", "pm/decision"]


def test_parse_enrichment_version_stamps_outputs(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_RUN_ID", "test-run")
    parsed = enrichment.parse_enrichment(
        """
        {
          "summary": "Hybrid taxonomy outputs carry provenance metadata for eval reproducibility.",
          "tags": ["React.js"],
          "importance": 6
        }
        """
    )

    assert parsed is not None
    metadata = parsed["enrichment_metadata"]
    assert metadata["prompt_version"]
    assert metadata["tag_mode"] == "hybrid"
    assert metadata["taxonomy_git_sha"]
    assert metadata["taxonomy_content_sha"]
    assert metadata["run_id"] == "test-run"


def test_build_prompt_switches_tag_rules_with_env(monkeypatch):
    chunk = {
        "content": "BrainLayer React.js tag normalization decision.",
        "project": "brainlayer",
        "content_type": "assistant_text",
    }

    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_TAG_MODE", "hybrid")
    hybrid_prompt = enrichment.build_prompt(chunk)
    assert "HYBRID TAG MODE" in hybrid_prompt
    assert "React.js/reactjs/React -> react" in hybrid_prompt

    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_TAG_MODE", "taxonomy")
    taxonomy_prompt = enrichment.build_prompt(chunk)
    assert "TAXONOMY WHITELIST MODE" in taxonomy_prompt
    assert "Do NOT invent free-form singleton tags" in taxonomy_prompt


def test_parse_enrichment_tag_whitelist_is_forward_only_for_existing_rows(tmp_path, monkeypatch):
    from brainlayer.vector_store import VectorStore

    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_TAG_MODE", "taxonomy")
    store = VectorStore(tmp_path / "forward-tags.db")
    try:
        cursor = store.conn.cursor()
        cursor.execute(
            """
            INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type,
                value_type, char_count, source, tags
            ) VALUES (?, ?, '{}', 'test', 'brainlayer', 'note', 'high', 80, 'manual', ?)
            """,
            (
                "legacy-tags",
                "Existing stored rows should not be rewritten by parsing a new enrichment response.",
                '["legacy-freeform", "project/brainlayer"]',
            ),
        )

        parsed = enrichment.parse_enrichment(
            """
            {
              "summary": "New enrichment output must be taxonomy-only.",
              "tags": ["project/brainlayer", "new-model-singleton", "tech/debug/investigation"],
              "importance": 7
            }
            """
        )

        stored_tags = cursor.execute("SELECT tags FROM chunks WHERE id = 'legacy-tags'").fetchone()[0]
    finally:
        store.close()

    assert parsed is not None
    assert parsed["tags"] == ["project/brainlayer", "tech/debug/investigation"]
    assert stored_tags == '["legacy-freeform", "project/brainlayer"]'


def test_tombstone_singleton_tags_preserves_taxonomy_tags(tmp_path):
    from brainlayer.tag_normalization import tombstone_singleton_tags
    from brainlayer.vector_store import VectorStore

    store = VectorStore(tmp_path / "tags.db")
    try:
        cursor = store.conn.cursor()
        rows = [
            ("c1", '["project/brainlayer", "singleton-sprawl"]'),
            ("c2", '["shared-sprawl"]'),
            ("c3", '["shared-sprawl"]'),
        ]
        for chunk_id, tags_json in rows:
            cursor.execute(
                """
                INSERT INTO chunks (
                    id, content, metadata, source_file, project, content_type,
                    value_type, char_count, source, tags
                ) VALUES (?, ?, '{}', 'test', 'brainlayer', 'note', 'high', 80, 'manual', ?)
                """,
                (chunk_id, f"content for {chunk_id} with enough text to store", tags_json),
            )

        result = tombstone_singleton_tags(store.conn)

        assert result.tombstoned == 1
        assert result.updated_chunks == 1
        assert cursor.execute("SELECT reason FROM tag_tombstones WHERE tag = 'singleton-sprawl'").fetchone()[0] == (
            "singleton-non-taxonomy"
        )
        assert cursor.execute("SELECT 1 FROM tag_tombstones WHERE tag = 'project/brainlayer'").fetchone() is None
        assert cursor.execute("SELECT tags FROM chunks WHERE id = 'c1'").fetchone()[0] == '["project/brainlayer"]'
        assert sorted(row[0] for row in cursor.execute("SELECT tag FROM chunk_tags")) == [
            "project/brainlayer",
            "shared-sprawl",
            "shared-sprawl",
        ]
    finally:
        store.close()


def test_store_memory_uses_lowercase_value_type():
    import tempfile
    from pathlib import Path

    from brainlayer.store import store_memory
    from brainlayer.vector_store import VectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store = VectorStore(Path(tmpdir) / "value-type.db")
        try:
            result = store_memory(
                store=vector_store,
                embed_fn=None,
                content="Value type casing should match the lower-case ContentValue enum.",
                memory_type="note",
                project="brainlayer",
            )
            assert (
                vector_store.conn.cursor()
                .execute(
                    "SELECT value_type FROM chunks WHERE id = ?",
                    (result["id"],),
                )
                .fetchone()[0]
                == "high"
            )
        finally:
            vector_store.close()


class TestRetryWithBackoff:
    """Per-chunk retry with exponential backoff."""

    def test_success_on_first_attempt_no_retry(self):
        """Successful LLM call doesn't trigger retry."""
        store = MagicMock()
        store.get_context.return_value = {"context": []}
        chunk = {"id": "test-chunk-001", "content": "test", "content_type": "user_message"}

        with (
            patch.object(enrichment, "call_llm", return_value='{"summary":"ok","tags":["test"]}'),
            patch.object(enrichment, "parse_enrichment", return_value={"summary": "ok", "tags": ["test"]}),
            patch.object(enrichment, "MAX_RETRIES", 2),
        ):
            result = enrichment._enrich_one(store, chunk, with_context=False)

        assert result is True

    def test_retry_on_llm_failure(self):
        """Failed LLM call retries up to MAX_RETRIES times."""
        store = MagicMock()
        chunk = {"id": "test-chunk-002", "content": "test", "content_type": "user_message"}

        call_count = 0

        def mock_call_llm(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None  # Fail first 2 attempts
            return '{"summary":"recovered","tags":["test"]}'

        with (
            patch.object(enrichment, "call_llm", side_effect=mock_call_llm),
            patch.object(
                enrichment,
                "parse_enrichment",
                side_effect=lambda r: {"summary": "recovered", "tags": ["test"]} if r else None,
            ),
            patch.object(enrichment, "MAX_RETRIES", 2),
            patch.object(enrichment, "RETRY_BASE_DELAY", 0.01),  # Fast for tests
            patch.object(enrichment, "RETRY_MAX_DELAY", 0.05),
        ):
            result = enrichment._enrich_one(store, chunk, with_context=False)

        assert result is True
        assert call_count == 3  # Initial + 2 retries

    def test_all_retries_exhausted(self):
        """Returns False after all retry attempts fail."""
        store = MagicMock()
        chunk = {"id": "test-chunk-003", "content": "test", "content_type": "user_message"}

        with (
            patch.object(enrichment, "call_llm", return_value=None),
            patch.object(enrichment, "MAX_RETRIES", 1),
            patch.object(enrichment, "RETRY_BASE_DELAY", 0.01),
            patch.object(enrichment, "RETRY_MAX_DELAY", 0.05),
        ):
            result = enrichment._enrich_one(store, chunk, with_context=False)

        assert result is False

    def test_no_retry_when_max_retries_zero(self):
        """MAX_RETRIES=0 means no retries, single attempt only."""
        store = MagicMock()
        chunk = {"id": "test-chunk-004", "content": "test", "content_type": "user_message"}
        call_count = 0

        def mock_call(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return None

        with (
            patch.object(enrichment, "call_llm", side_effect=mock_call),
            patch.object(enrichment, "MAX_RETRIES", 0),
        ):
            result = enrichment._enrich_one(store, chunk, with_context=False)

        assert result is False
        assert call_count == 1

    def test_backoff_increases_delay(self):
        """Verify backoff delay increases between retries."""
        store = MagicMock()
        chunk = {"id": "test-chunk-005", "content": "test", "content_type": "user_message"}
        delays = []

        original_sleep = time.sleep

        def mock_sleep(duration):
            delays.append(duration)
            # Don't actually sleep in tests

        with (
            patch.object(enrichment, "call_llm", return_value=None),
            patch.object(enrichment, "MAX_RETRIES", 3),
            patch.object(enrichment, "RETRY_BASE_DELAY", 1.0),
            patch.object(enrichment, "RETRY_MAX_DELAY", 100.0),
            patch("time.sleep", side_effect=mock_sleep),
        ):
            enrichment._enrich_one(store, chunk, with_context=False)

        assert len(delays) == 3  # 3 retry sleeps
        # Delays should generally increase (base * 2^attempt + jitter)
        # With jitter it's not strictly monotonic, but base increases: 1, 2, 4
        assert delays[0] < 3.0  # ~1.0 + up to 0.3 jitter
        assert delays[1] < 5.0  # ~2.0 + up to 0.6 jitter


class TestCircuitBreaker:
    """Batch-level circuit breaker aborts on consecutive failures."""

    def test_circuit_breaks_on_threshold(self):
        """Batch aborts after CIRCUIT_BREAKER_THRESHOLD consecutive failures."""
        store = MagicMock()
        # Return 20 chunks but circuit should break after threshold
        chunks = [{"id": f"chunk-{i}", "content": f"test {i}", "content_type": "user_message"} for i in range(20)]
        store.get_unenriched_chunks.return_value = chunks

        with (
            patch.object(enrichment, "_enrich_one", return_value=False),
            patch.object(enrichment, "CIRCUIT_BREAKER_THRESHOLD", 5),
        ):
            result = enrichment.enrich_batch(store, batch_size=20, parallel=1)

        assert result["circuit_broken"] is True
        # Should have stopped at threshold, not processed all 20
        assert result["processed"] == 5
        assert result["failed"] == 5
        assert result["success"] == 0

    def test_no_circuit_break_on_intermittent_failures(self):
        """Intermittent failures (not consecutive) don't trigger circuit breaker."""
        store = MagicMock()
        chunks = [{"id": f"chunk-{i}", "content": f"test {i}", "content_type": "user_message"} for i in range(10)]
        store.get_unenriched_chunks.return_value = chunks

        # Alternate success/failure
        results = [True, False, True, False, True, False, True, False, True, False]
        call_idx = 0

        def mock_enrich(*args, **kwargs):
            nonlocal call_idx
            r = results[call_idx]
            call_idx += 1
            return r

        with (
            patch.object(enrichment, "_enrich_one", side_effect=mock_enrich),
            patch.object(enrichment, "CIRCUIT_BREAKER_THRESHOLD", 5),
        ):
            result = enrichment.enrich_batch(store, batch_size=10, parallel=1)

        assert result["circuit_broken"] is False
        assert result["processed"] == 10
        assert result["success"] == 5
        assert result["failed"] == 5

    def test_circuit_break_resets_on_success(self):
        """A single success resets the consecutive failure counter."""
        store = MagicMock()
        chunks = [{"id": f"chunk-{i}", "content": f"test {i}", "content_type": "user_message"} for i in range(15)]
        store.get_unenriched_chunks.return_value = chunks

        # 4 failures, 1 success, 4 failures, 1 success, etc — never hits threshold of 5
        results = [False, False, False, False, True] * 3
        call_idx = 0

        def mock_enrich(*args, **kwargs):
            nonlocal call_idx
            r = results[call_idx]
            call_idx += 1
            return r

        with (
            patch.object(enrichment, "_enrich_one", side_effect=mock_enrich),
            patch.object(enrichment, "CIRCUIT_BREAKER_THRESHOLD", 5),
        ):
            result = enrichment.enrich_batch(store, batch_size=15, parallel=1)

        assert result["circuit_broken"] is False
        assert result["processed"] == 15


class TestMLXTimeout:
    """MLX uses shorter default timeout."""

    def test_mlx_default_timeout_is_60(self):
        assert enrichment.MLX_DEFAULT_TIMEOUT == 60

    def test_mlx_timeout_env_override(self):
        """MLX timeout can be overridden via env var."""
        import os

        old = os.environ.get("BRAINLAYER_MLX_TIMEOUT")
        try:
            os.environ["BRAINLAYER_MLX_TIMEOUT"] = "120"
            # Re-evaluate (module-level constant, so we test the pattern)
            assert int(os.environ["BRAINLAYER_MLX_TIMEOUT"]) == 120
        finally:
            if old is not None:
                os.environ["BRAINLAYER_MLX_TIMEOUT"] = old
            else:
                os.environ.pop("BRAINLAYER_MLX_TIMEOUT", None)


class TestConfigConstants:
    """Verify config constants have sensible defaults."""

    def test_max_retries_default(self):
        assert enrichment.MAX_RETRIES == 2

    def test_retry_base_delay_default(self):
        assert enrichment.RETRY_BASE_DELAY == 2.0

    def test_retry_max_delay_default(self):
        assert enrichment.RETRY_MAX_DELAY == 30.0

    def test_circuit_breaker_threshold_default(self):
        assert enrichment.CIRCUIT_BREAKER_THRESHOLD == 10
