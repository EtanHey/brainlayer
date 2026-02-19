"""Tests for brainlayer.paths — DB path resolution."""

import os
from unittest.mock import patch

import pytest

from brainlayer.paths import get_db_path


class TestGetDbPath:
    """Test DB path resolution order."""

    def test_env_var_override(self, tmp_path):
        """BRAINLAYER_DB env var takes highest priority."""
        db_path = tmp_path / "custom.db"
        with patch.dict(os.environ, {"BRAINLAYER_DB": str(db_path)}):
            assert get_db_path() == db_path

    def test_legacy_path_if_exists(self, tmp_path):
        """Legacy zikaron path used when it exists."""
        legacy = tmp_path / "zikaron.db"
        legacy.touch()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("brainlayer.paths._LEGACY_DB_PATH", legacy),
            patch("brainlayer.paths._CANONICAL_DB_PATH", tmp_path / "brainlayer.db"),
        ):
            # Remove env var if set
            os.environ.pop("BRAINLAYER_DB", None)
            assert get_db_path() == legacy

    def test_canonical_path_fresh_install(self, tmp_path):
        """Canonical path used when no legacy DB exists."""
        canonical = tmp_path / "brainlayer" / "brainlayer.db"
        legacy = tmp_path / "nonexistent" / "zikaron.db"
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("brainlayer.paths._LEGACY_DB_PATH", legacy),
            patch("brainlayer.paths._CANONICAL_DB_PATH", canonical),
        ):
            os.environ.pop("BRAINLAYER_DB", None)
            result = get_db_path()
            assert result == canonical
            assert canonical.parent.exists()  # Parent dir created

    @pytest.mark.integration
    def test_real_db_exists(self):
        """The real production DB exists at the resolved path."""
        from brainlayer.paths import DEFAULT_DB_PATH

        assert DEFAULT_DB_PATH.exists(), f"DB not found at {DEFAULT_DB_PATH}"
        assert DEFAULT_DB_PATH.stat().st_size > 1_000_000, "DB too small — might be empty"
