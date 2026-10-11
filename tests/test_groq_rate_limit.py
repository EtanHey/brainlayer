"""Tests for Groq rate limiting in enrichment pipeline (Task A1).

Verifies that the enrichment pipeline throttles Groq API calls to stay
under the free tier rate limit (~30 req/min).
"""


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
            # Both sources must be present
            assert "claude_code" in sources, f"Expected claude_code in sources, got: {sources}"
            assert "youtube" in sources, f"Expected youtube in sources, got: {sources}"
            # Claude Code should come before YouTube
            cc_idx = sources.index("claude_code")
            yt_idx = sources.index("youtube")
            assert cc_idx < yt_idx, f"Claude Code (idx={cc_idx}) should rank before YouTube (idx={yt_idx})"
            store.close()
