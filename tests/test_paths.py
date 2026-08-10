"""Tests for brainlayer.paths — DB path resolution."""

import os
from unittest.mock import patch

import pytest

import brainlayer.paths as brainlayer_paths
from brainlayer.paths import get_db_path, resolve_db_path


class TestGetDbPath:
    """Test DB path resolution order."""

    def test_env_var_override(self, tmp_path):
        """BRAINLAYER_DB env var takes highest priority."""
        db_path = tmp_path / "missing-parent" / "custom.db"
        with patch.dict(os.environ, {"BRAINLAYER_DB": str(db_path)}):
            assert get_db_path() == db_path
            assert not db_path.parent.exists()

    def test_canonical_path_fresh_install(self, tmp_path, monkeypatch):
        """Canonical path used when no DB exists yet."""
        canonical = tmp_path / "brainlayer" / "brainlayer.db"
        with patch("brainlayer.paths._CANONICAL_DB_PATH", canonical):
            monkeypatch.delenv("BRAINLAYER_DB", raising=False)
            result = get_db_path()
            assert result == canonical
            assert canonical.parent.exists()  # Parent dir created

    def test_resolve_db_path_does_not_create_parent(self, tmp_path, monkeypatch):
        canonical = tmp_path / "brainlayer" / "brainlayer.db"
        with patch("brainlayer.paths._CANONICAL_DB_PATH", canonical):
            monkeypatch.delenv("BRAINLAYER_DB", raising=False)

            assert resolve_db_path() == canonical
            assert not canonical.parent.exists()

    @pytest.mark.integration
    def test_real_db_exists(self):
        """The real production DB exists at the resolved path."""
        from brainlayer.paths import DEFAULT_DB_PATH

        assert DEFAULT_DB_PATH.exists(), f"DB not found at {DEFAULT_DB_PATH}"
        assert DEFAULT_DB_PATH.stat().st_size > 1_000_000, "DB too small — might be empty"


def test_spotlight_exclusion_accepts_marker_on_directory(tmp_path):
    marker_name = ".metadata_never_index"
    (tmp_path / marker_name).touch()

    assert brainlayer_paths.is_spotlight_excluded(tmp_path)


def test_spotlight_exclusion_accepts_marker_on_ancestor(tmp_path):
    marker_name = ".metadata_never_index"
    (tmp_path / marker_name).touch()
    child = tmp_path / "logs" / "nested"
    child.mkdir(parents=True)

    assert brainlayer_paths.is_spotlight_excluded(child)


def test_spotlight_exclusion_rejects_unmarked_path(tmp_path):
    child = tmp_path / "queue"
    child.mkdir()

    assert not brainlayer_paths.is_spotlight_excluded(child)
