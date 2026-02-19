"""Phase 4 QA: Test Phase 3 core fixes — date filtering, project normalization, metadata.

Tests cover:
1. paths.py — DB path resolution logic
2. Date filtering — date_from/date_to in search and hybrid_search
3. Project name normalization — Claude Code paths, worktrees, clean names
4. Search metadata — created_at and source in results
5. Backfill coverage — all chunks should have created_at
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from brainlayer.mcp import normalize_project_name
from brainlayer.paths import get_db_path
from brainlayer.vector_store import VectorStore

# ============================================================================
# 1. DB Path Resolution
# ============================================================================


class TestDBPathResolution:
    """Test paths.py resolves database location correctly."""

    def test_env_var_override(self, tmp_path):
        """BRAINLAYER_DB env var should take priority over everything."""
        custom = str(tmp_path / "custom.db")
        with patch.dict(os.environ, {"BRAINLAYER_DB": custom}):
            assert get_db_path() == Path(custom)

    def test_legacy_path_preferred_over_canonical(self, tmp_path):
        """Legacy zikaron path should be used if it exists."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove env var if set
            os.environ.pop("BRAINLAYER_DB", None)
            with patch("brainlayer.paths._LEGACY_DB_PATH", tmp_path / "legacy.db"):
                # Create the file so .exists() returns True
                (tmp_path / "legacy.db").touch()
                result = get_db_path()
                assert result == tmp_path / "legacy.db"

    def test_canonical_path_when_legacy_missing(self, tmp_path):
        """Canonical path should be used for fresh installs."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINLAYER_DB", None)
            with patch("brainlayer.paths._LEGACY_DB_PATH", tmp_path / "nonexistent.db"):
                with patch("brainlayer.paths._CANONICAL_DB_PATH", tmp_path / "canonical.db"):
                    result = get_db_path()
                    assert result == tmp_path / "canonical.db"


# ============================================================================
# 2. Date Filtering
# ============================================================================


@pytest.fixture
def store_with_dates(tmp_path):
    """Store with chunks that have different created_at dates."""
    db_path = tmp_path / "test_dates.db"
    store = VectorStore(db_path)

    chunks = [
        {
            "id": "old-chunk",
            "content": "Authentication using JWT tokens for the API",
            "metadata": {"role": "assistant"},
            "source_file": "/session/old.jsonl",
            "project": "my-project",
            "content_type": "assistant_text",
            "char_count": 50,
            "source": "claude_code",
            "created_at": "2026-01-15T10:00:00+00:00",
        },
        {
            "id": "mid-chunk",
            "content": "Database migration strategy for PostgreSQL upgrade",
            "metadata": {"role": "assistant"},
            "source_file": "/session/mid.jsonl",
            "project": "my-project",
            "content_type": "assistant_text",
            "char_count": 55,
            "source": "claude_code",
            "created_at": "2026-02-01T10:00:00+00:00",
        },
        {
            "id": "new-chunk",
            "content": "Deploy React app to Vercel with environment variables",
            "metadata": {"role": "assistant"},
            "source_file": "/session/new.jsonl",
            "project": "my-project",
            "content_type": "assistant_text",
            "char_count": 55,
            "source": "claude_code",
            "created_at": "2026-02-15T10:00:00+00:00",
        },
    ]
    embeddings = [[float(i) / 1024] * 1024 for i in range(3)]
    store.upsert_chunks(chunks, embeddings)
    return store


class TestDateFiltering:
    """Test date_from/date_to search parameters."""

    def test_search_date_from(self, store_with_dates):
        """date_from should exclude chunks before the date."""
        query_embedding = [0.001] * 1024
        results = store_with_dates.search(
            query_embedding=query_embedding,
            n_results=10,
            date_from="2026-02-01",
        )
        ids = results["ids"][0]
        assert "old-chunk" not in ids
        assert "mid-chunk" in ids
        assert "new-chunk" in ids

    def test_search_date_to(self, store_with_dates):
        """date_to should exclude chunks after the date (exclusive comparison)."""
        query_embedding = [0.001] * 1024
        # date_to uses string comparison: "2026-02-01T10:00:00" > "2026-02-01"
        # So date_to="2026-02-02" includes Feb 1 but excludes Feb 15
        results = store_with_dates.search(
            query_embedding=query_embedding,
            n_results=10,
            date_to="2026-02-02",
        )
        ids = results["ids"][0]
        assert "old-chunk" in ids
        assert "mid-chunk" in ids
        assert "new-chunk" not in ids

    def test_search_date_range(self, store_with_dates):
        """date_from + date_to should filter to a specific range."""
        query_embedding = [0.001] * 1024
        results = store_with_dates.search(
            query_embedding=query_embedding,
            n_results=10,
            date_from="2026-01-20",
            date_to="2026-02-10",
        )
        ids = results["ids"][0]
        assert "old-chunk" not in ids
        assert "mid-chunk" in ids
        assert "new-chunk" not in ids

    def test_search_no_date_filter_returns_all(self, store_with_dates):
        """No date filter should return all chunks."""
        query_embedding = [0.001] * 1024
        results = store_with_dates.search(
            query_embedding=query_embedding,
            n_results=10,
        )
        assert len(results["ids"][0]) == 3

    def test_hybrid_search_date_from(self, store_with_dates):
        """Hybrid search should also respect date_from."""
        query_embedding = [0.001] * 1024
        results = store_with_dates.hybrid_search(
            query_embedding=query_embedding,
            query_text="authentication JWT",
            n_results=10,
            date_from="2026-02-01",
        )
        # Old chunk about auth/JWT should be excluded by date
        ids = results["ids"][0]
        assert "old-chunk" not in ids

    def test_hybrid_search_date_to(self, store_with_dates):
        """Hybrid search should also respect date_to."""
        query_embedding = [0.001] * 1024
        results = store_with_dates.hybrid_search(
            query_embedding=query_embedding,
            query_text="deploy React",
            n_results=10,
            date_to="2026-02-10",
        )
        ids = results["ids"][0]
        assert "new-chunk" not in ids


# ============================================================================
# 3. Project Name Normalization
# ============================================================================


class TestProjectNameNormalization:
    """Test normalize_project_name handles Claude Code paths correctly."""

    def test_none_input(self):
        assert normalize_project_name(None) is None

    def test_empty_string(self):
        assert normalize_project_name("") is None

    def test_dash_only(self):
        assert normalize_project_name("-") is None

    def test_whitespace(self):
        assert normalize_project_name("   ") is None

    def test_clean_name_passthrough(self):
        """Already-clean project names should pass through."""
        assert normalize_project_name("golems") == "golems"

    def test_worktree_suffix_stripped(self):
        """Worktree suffixes should be removed."""
        assert normalize_project_name("golems-nightshift-1770775282043") == "golems"
        assert normalize_project_name("golems-haiku-1234567890") == "golems"
        assert normalize_project_name("golems-worktree-9999999999") == "golems"

    def test_claude_code_simple_path(self, tmp_path):
        """Simple Claude Code path: -Users-name-Gits-projectname."""
        # Create a mock directory structure
        gits_dir = tmp_path / "Gits"
        gits_dir.mkdir()
        (gits_dir / "myproject").mkdir()

        segments = f"-{tmp_path.name}-Gits-myproject"
        # We need to mock the path correctly
        with patch("os.path.isdir") as mock_isdir:
            mock_isdir.side_effect = lambda p: p.endswith("/myproject")
            result = normalize_project_name("-Users-etanheyman-Gits-myproject")
            # Should try to match "myproject" against filesystem
            # Since we mock isdir to return True for myproject, it should return it
            assert result == "myproject"

    def test_claude_code_compound_name(self):
        """Compound names like rudy-monorepo should try filesystem lookup."""
        # When os.path.isdir returns True for "rudy-monorepo", use the compound name
        with patch("os.path.isdir") as mock_isdir:
            mock_isdir.side_effect = lambda p: "rudy-monorepo" in p
            result = normalize_project_name("-Users-etanheyman-Gits-rudy-monorepo")
            assert result == "rudy-monorepo"

    def test_claude_code_compound_name_fallback(self):
        """When no directory matches, fall back to first segment."""
        with patch("os.path.isdir", return_value=False):
            result = normalize_project_name("-Users-etanheyman-Gits-rudy-monorepo")
            assert result == "rudy"

    def test_claude_code_desktop_gits(self):
        """Old-style Desktop/Gits path should work too."""
        with patch("os.path.isdir") as mock_isdir:
            mock_isdir.side_effect = lambda p: "golems" in p
            result = normalize_project_name("-Users-etanheyman-Desktop-Gits-golems")
            assert result == "golems"

    def test_no_gits_segment(self):
        """Paths without 'Gits' should return None."""
        result = normalize_project_name("-Users-etanheyman-Documents-stuff")
        assert result is None


# ============================================================================
# 4. Search Metadata
# ============================================================================


class TestSearchMetadata:
    """Test that search results include created_at and source."""

    def test_search_results_include_created_at(self, store_with_dates):
        """Search results should include created_at in metadata."""
        query_embedding = [0.001] * 1024
        results = store_with_dates.search(
            query_embedding=query_embedding,
            n_results=3,
        )
        for meta in results["metadatas"][0]:
            assert "created_at" in meta, f"Missing created_at in {meta}"

    def test_search_results_include_source(self, store_with_dates):
        """Search results should include source in metadata."""
        query_embedding = [0.001] * 1024
        results = store_with_dates.search(
            query_embedding=query_embedding,
            n_results=3,
        )
        for meta in results["metadatas"][0]:
            assert "source" in meta, f"Missing source in {meta}"
            assert meta["source"] == "claude_code"

    def test_hybrid_search_results_include_metadata(self, store_with_dates):
        """Hybrid search results should also include created_at and source."""
        query_embedding = [0.001] * 1024
        results = store_with_dates.hybrid_search(
            query_embedding=query_embedding,
            query_text="database",
            n_results=3,
        )
        for meta in results["metadatas"][0]:
            assert "created_at" in meta
            assert "source" in meta


# ============================================================================
# 5. Chunk Boundary Improvements
# ============================================================================


class TestChunkBoundaries:
    """Test sentence-aware chunking from Phase 3."""

    def test_sentence_splitting(self):
        """Chunks should split at sentence boundaries, not mid-sentence."""
        from brainlayer.pipeline.chunk import _split_at_sentences

        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence is longer and has more words."
        chunks = _split_at_sentences(text, target_size=50)
        # Each chunk should end at a sentence boundary (or be the last chunk)
        for chunk in chunks[:-1]:
            assert chunk.rstrip().endswith((".")) or chunk.rstrip().endswith(("!")) or chunk.rstrip().endswith(("?")), (
                f"Chunk doesn't end at sentence boundary: '{chunk}'"
            )

    def test_short_text_not_split(self):
        """Text shorter than target should not be split."""
        from brainlayer.pipeline.chunk import _split_at_sentences

        text = "Short text."
        chunks = _split_at_sentences(text, target_size=2000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_paragraph_splits_at_sentences(self):
        """chunk_content should split long paragraphs at sentence boundaries."""
        from brainlayer.pipeline.chunk import chunk_content
        from brainlayer.pipeline.classify import ClassifiedContent, ContentType, ContentValue

        # Create a long paragraph that exceeds TARGET_CHUNK_SIZE (2000 chars)
        sentences = [f"This is sentence number {i} with some padding words to make it longer." for i in range(60)]
        long_text = " ".join(sentences)  # ~3000 chars

        classified = ClassifiedContent(
            content=long_text,
            content_type=ContentType.ASSISTANT_TEXT,
            value=ContentValue.HIGH,
            metadata={"role": "assistant"},
        )
        chunks = chunk_content(classified)
        assert len(chunks) > 1, "Long text should be split into multiple chunks"


# ============================================================================
# 6. Created_at in upsert_chunks
# ============================================================================


class TestCreatedAtUpsert:
    """Test that created_at is properly stored during chunk insertion."""

    def test_created_at_stored(self, tmp_path):
        """Chunks with created_at should have it stored in the DB."""
        store = VectorStore(tmp_path / "test.db")
        chunks = [
            {
                "id": "dated-chunk",
                "content": "Test content with a date",
                "metadata": {},
                "source_file": "/test.jsonl",
                "project": "test",
                "content_type": "user_message",
                "char_count": 30,
                "source": "claude_code",
                "created_at": "2026-02-19T10:00:00+00:00",
            }
        ]
        store.upsert_chunks(chunks, [[0.1] * 1024])

        cursor = store.conn.cursor()
        row = list(cursor.execute("SELECT created_at FROM chunks WHERE id = 'dated-chunk'"))
        assert row[0][0] == "2026-02-19T10:00:00+00:00"

    def test_created_at_null_when_missing(self, tmp_path):
        """Chunks without created_at should have NULL in the DB."""
        store = VectorStore(tmp_path / "test.db")
        chunks = [
            {
                "id": "undated-chunk",
                "content": "Test content without a date",
                "metadata": {},
                "source_file": "/test.jsonl",
                "project": "test",
                "content_type": "user_message",
                "char_count": 30,
                "source": "claude_code",
            }
        ]
        store.upsert_chunks(chunks, [[0.1] * 1024])

        cursor = store.conn.cursor()
        row = list(cursor.execute("SELECT created_at FROM chunks WHERE id = 'undated-chunk'"))
        assert row[0][0] is None
