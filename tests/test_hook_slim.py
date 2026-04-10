"""Tests for hook-slim changes: capped results, truncated snippets, token budget.

Validates that UserPromptSubmit hook injection stays under ~400 tokens (1600 chars)
after the R77 slimming changes.
"""

import importlib.util
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent / "hooks"

MAX_INJECTION_CHARS = 1600  # ~400 tokens at 4 chars/token


def load_hook_module():
    """Import the prompt-search hook as a module."""
    spec = importlib.util.spec_from_file_location(
        "brainlayer_prompt_search",
        HOOKS_DIR / "brainlayer-prompt-search.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def hook():
    return load_hook_module()


class TestSnippetTruncation:
    """Each result snippet must be truncated to ~80 chars."""

    def test_truncate_default_is_80_chars(self, hook):
        long_text = "A" * 200
        result = hook.truncate(long_text)
        assert len(result) <= 83  # 80 + "..."

    def test_truncate_short_text_unchanged(self, hook):
        short_text = "Short text here"
        result = hook.truncate(short_text)
        assert result == short_text

    def test_truncate_respects_custom_max(self, hook):
        text = "A" * 200
        result = hook.truncate(text, max_chars=50)
        assert len(result) <= 53  # 50 + "..."

    def test_truncate_prefers_sentence_boundary(self, hook):
        # Text with a sentence boundary within search window
        text = "First sentence here. Second sentence that extends past the limit by quite a bit more."
        result = hook.truncate(text)
        assert result.endswith("...")
        assert len(result) <= 83

    def test_truncate_cuts_at_last_sentence_end_before_limit(self, hook):
        text = (
            "Short sentence. Another complete sentence. "
            "This final sentence extends well beyond the truncation limit and should be omitted."
        )

        result = hook.truncate(text, max_chars=55)

        assert result == "Short sentence. Another complete sentence...."


class TestResultCountCap:
    """Result count must be capped at MAX_ADAPTIVE_INJECTION (3)."""

    def test_max_adaptive_injection_is_3(self, hook):
        assert hook.MAX_ADAPTIVE_INJECTION == 3

    def test_select_adaptive_injection_rows_caps_at_3(self, hook):
        # Create 10 high-scoring rows
        rows = [
            {
                "id": f"chunk-{i}",
                "content": f"Content {i}",
                "rrf_score": 0.02 - i * 0.001,
                "project": "test",
                "tags": "[]",
            }
            for i in range(10)
        ]
        selected = hook.select_adaptive_injection_rows(rows)
        assert len(selected) <= 3

    def test_light_mode_caps_at_2(self, hook):
        rows = [
            {
                "id": f"chunk-{i}",
                "content": f"Content {i}",
                "rrf_score": 0.02 - i * 0.001,
                "project": "test",
                "tags": "[]",
            }
            for i in range(10)
        ]
        selected = hook.select_adaptive_injection_rows(rows, light_mode=True)
        assert len(selected) <= 2


class TestInjectionTokenBudget:
    """Total injection output must stay under ~400 tokens (1600 chars)."""

    def _make_rows(self, n, content_len=300):
        """Create n fake result rows with long content."""
        return [
            (
                f"chunk-{i}",
                "X" * content_len,
                8,  # importance
                "brainlayer",
                '["test"]',
                "2026-04-01T00:00:00Z",
            )
            for i in range(n)
        ]

    def test_3_results_under_budget(self, hook):
        rows = self._make_rows(3)
        lines = []
        hook.inject_search_results(lines, rows, deep=False)
        output = "\n".join(lines)
        assert len(output) < MAX_INJECTION_CHARS, (
            f"Injection output is {len(output)} chars, exceeds {MAX_INJECTION_CHARS} budget"
        )

    def test_5_results_deep_mode_under_budget(self, hook):
        """Even deep mode (which fetches more) should stay reasonable after truncation."""
        rows = self._make_rows(5)
        lines = []
        hook.inject_search_results(lines, rows, deep=True)
        output = "\n".join(lines)
        # Deep mode may be slightly larger but should still be under 2x budget
        assert len(output) < MAX_INJECTION_CHARS * 2, (
            f"Deep injection output is {len(output)} chars, exceeds {MAX_INJECTION_CHARS * 2}"
        )

    def test_system_nudge_present(self, hook):
        rows = self._make_rows(1)
        lines = []
        hook.inject_search_results(lines, rows, deep=False)
        output = "\n".join(lines)
        assert "brain_search" in output
        assert "BrainLayer memory available" in output

    def test_no_importance_field_in_output(self, hook):
        rows = self._make_rows(1)
        lines = []
        hook.inject_search_results(lines, rows, deep=False)
        output = "\n".join(lines)
        assert "imp:" not in output

    def test_each_result_line_under_120_chars(self, hook):
        """Each individual result line (date+project+snippet) should be compact."""
        rows = self._make_rows(3, content_len=500)
        lines = []
        hook.inject_search_results(lines, rows, deep=False)
        # Skip the header line, check result lines
        result_lines = [line for line in lines if line.startswith("- [")]
        for line in result_lines:
            assert len(line) < 120, f"Result line too long ({len(line)} chars): {line[:80]}..."
