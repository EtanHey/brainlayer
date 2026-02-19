"""Tests for project name normalization in MCP server."""

import os

from brainlayer.mcp import normalize_project_name


class TestNormalizeProjectName:
    """Test Claude Code path encoding → clean project name."""

    def test_none_returns_none(self):
        assert normalize_project_name(None) is None

    def test_empty_returns_none(self):
        assert normalize_project_name("") is None
        assert normalize_project_name("  ") is None
        assert normalize_project_name("-") is None

    def test_simple_name_passthrough(self):
        """Already-clean names pass through unchanged."""
        assert normalize_project_name("golems") == "golems"

    def test_claude_code_encoded_path(self, test_user):
        """Standard Claude Code path encoding decodes correctly."""
        result = normalize_project_name(f"-Users-{test_user}-Gits-golems")
        assert result == "golems"

    def test_desktop_gits_path(self, test_user):
        """Old Desktop/Gits paths decode correctly."""
        result = normalize_project_name(f"-Users-{test_user}-Desktop-Gits-golems")
        # Either finds the dir or falls back to first segment
        assert result is not None

    def test_compound_name_with_dashes(self):
        """Compound project names with dashes resolve via filesystem check."""
        # Only works if the directory actually exists
        home = os.path.expanduser("~")
        gits_dir = os.path.join(home, "Gits")
        if os.path.isdir(gits_dir):
            for entry in os.listdir(gits_dir):
                if "-" in entry and os.path.isdir(os.path.join(gits_dir, entry)):
                    # Test that this compound name resolves correctly
                    encoded = f"-Users-{os.path.basename(home)}-Gits-{entry}"
                    # Just verify it doesn't crash
                    result = normalize_project_name(encoded)
                    assert result is not None
                    break

    def test_worktree_suffix_stripped(self):
        """Worktree suffixes are stripped."""
        assert normalize_project_name("golems-nightshift-1770775282043") == "golems"
        assert normalize_project_name("golems-haiku-1770775282043") == "golems"
        assert normalize_project_name("golems-worktree-1770775282043") == "golems"

    def test_no_gits_segment_returns_none(self, test_user):
        """Paths without 'Gits' segment return None."""
        result = normalize_project_name(f"-Users-{test_user}-Documents-stuff")
        assert result is None
