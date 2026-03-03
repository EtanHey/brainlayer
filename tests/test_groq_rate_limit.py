"""Tests for Groq rate limiting in enrichment pipeline (Task A1).

Verifies that the enrichment pipeline throttles Groq API calls to stay
under the free tier rate limit (~30 req/min).
"""

import time
from unittest.mock import MagicMock, patch


class TestGroqRateLimiting:
    """Groq enrichment should throttle calls to avoid 429 rate limits."""

    def test_groq_calls_have_minimum_delay(self):
        """Sequential Groq enrichment calls should have at least GROQ_RATE_LIMIT_DELAY between them."""
        from brainlayer.pipeline.enrichment import GROQ_RATE_LIMIT_DELAY

        assert GROQ_RATE_LIMIT_DELAY >= 2.0, f"GROQ_RATE_LIMIT_DELAY should be >= 2.0s, got {GROQ_RATE_LIMIT_DELAY}"

    def test_call_groq_sleeps_between_calls(self):
        """call_groq should enforce minimum delay between consecutive calls."""
        from brainlayer.pipeline import enrichment

        call_times = []

        def fake_post(*args, **kwargs):
            call_times.append(time.monotonic())
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "choices": [{"message": {"content": '{"summary":"test"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
            resp.raise_for_status = MagicMock()
            return resp

        with (
            patch("brainlayer.pipeline.enrichment.requests.post", side_effect=fake_post),
            patch("brainlayer.pipeline.enrichment._log_glm_usage"),
            patch("brainlayer.pipeline.enrichment.GROQ_API_KEY", "fake-key"),
        ):
            enrichment.call_groq("prompt 1")
            enrichment.call_groq("prompt 2")
            enrichment.call_groq("prompt 3")

        assert len(call_times) == 3
        # Verify delays between consecutive calls
        for i in range(1, len(call_times)):
            gap = call_times[i] - call_times[i - 1]
            # Allow some tolerance (sleep isn't perfectly precise)
            assert gap >= 1.5, f"Gap between call {i} and {i + 1} was {gap:.2f}s, expected >= 1.5s"

    def test_rate_limit_delay_env_override(self):
        """BRAINLAYER_GROQ_RATE_DELAY env var should override the default delay."""
        import os
        from importlib import reload

        from brainlayer.pipeline import enrichment

        original = os.environ.get("BRAINLAYER_GROQ_RATE_DELAY")
        try:
            os.environ["BRAINLAYER_GROQ_RATE_DELAY"] = "5.0"
            reloaded = reload(enrichment)
            assert reloaded.GROQ_RATE_LIMIT_DELAY == 5.0
        finally:
            if original is None:
                os.environ.pop("BRAINLAYER_GROQ_RATE_DELAY", None)
            else:
                os.environ["BRAINLAYER_GROQ_RATE_DELAY"] = original
            reload(enrichment)  # restore original value


class TestEnrichmentSourcePriority:
    """Enrichment should prioritize Claude Code chunks over YouTube backlog."""

    def test_get_unenriched_prioritizes_claude_code(self):
        """get_unenriched_chunks should return Claude Code chunks before YouTube."""
        import tempfile
        from pathlib import Path

        from brainlayer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(Path(tmpdir) / "test.db")
            cursor = store.conn.cursor()

            # Insert YouTube chunk (older)
            cursor.execute(
                """INSERT INTO chunks (id, content, metadata, source_file, project,
                   content_type, char_count, source, created_at)
                   VALUES (?, ?, '{}', 'yt.jsonl', 'youtube', 'user_message', 100,
                           'youtube', '2026-01-01T00:00:00')""",
                ("yt-1", "some youtube content"),
            )
            # Insert Claude Code chunk (newer)
            cursor.execute(
                """INSERT INTO chunks (id, content, metadata, source_file, project,
                   content_type, char_count, source, created_at)
                   VALUES (?, ?, '{}', 'cc.jsonl', 'brainlayer', 'assistant_text', 100,
                           'claude_code', '2026-03-01T00:00:00')""",
                ("cc-1", "some claude code content"),
            )

            unenriched = store.get_unenriched_chunks(batch_size=10)
            sources = [c.get("source") for c in unenriched]
            # Claude Code should come before YouTube
            if "claude_code" in sources and "youtube" in sources:
                cc_idx = sources.index("claude_code")
                yt_idx = sources.index("youtube")
                assert cc_idx < yt_idx, f"Claude Code (idx={cc_idx}) should rank before YouTube (idx={yt_idx})"
            store.close()
