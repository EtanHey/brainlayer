"""Tests for score-based adaptive prompt injection in the BrainLayer hook."""

import importlib.util
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent / "hooks"


def load_hook_module(filename: str):
    spec = importlib.util.spec_from_file_location(
        filename.replace("-", "_").replace(".py", ""),
        HOOKS_DIR / filename,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def prompt_search():
    return load_hook_module("brainlayer-prompt-search.py")


def _row(
    chunk_id: str,
    score: float,
    *,
    content: str | None = None,
    project: str | None = "brainlayer",
    tags: str | None = '["memory"]',
):
    return {
        "id": chunk_id,
        "content": content or f"content for {chunk_id}",
        "importance": 5.0,
        "project": project,
        "tags": tags,
        "created_at": "2026-04-05T00:00:00Z",
        "rrf_score": score,
    }


class TestAdaptiveInjectionGating:
    def test_score_gating_high_confidence(self, prompt_search):
        rows = [
            _row("best", 0.033),
            _row("second", 0.019),
            _row("third", 0.017),
            _row("tail", 0.011),
        ]

        selected = prompt_search.select_adaptive_injection_rows(rows, entity_count=1)

        assert 2 <= len(selected) <= 3
        assert [row["id"] for row in selected] == ["best", "third", "second"]

    def test_score_gating_moderate(self, prompt_search):
        rows = [
            _row("best", 0.0145),
            _row("second", 0.013),
            _row("third", 0.012),
            _row("fourth", 0.011),
            _row("fifth", 0.0102),
            _row("sixth", 0.008),
        ]

        selected = prompt_search.select_adaptive_injection_rows(rows)

        # MAX_ADAPTIVE_INJECTION=3, so moderate-confidence returns top 3
        assert [row["id"] for row in selected] == ["best", "third", "second"]

    def test_score_gating_light(self, prompt_search):
        rows = [
            _row("best", 0.009),
            _row("second", 0.008),
            _row("third", 0.006),
        ]

        selected = prompt_search.select_adaptive_injection_rows(rows)

        assert [row["id"] for row in selected] == ["best"]

    def test_score_gating_skip(self, prompt_search):
        rows = [
            _row("best", 0.0049),
            _row("second", 0.004),
        ]

        selected = prompt_search.select_adaptive_injection_rows(rows)

        assert selected == []

    def test_strategic_ordering(self, prompt_search):
        rows = [
            _row("best", 0.015),
            _row("second", 0.014),
            _row("third", 0.013),
            _row("fourth", 0.012),
        ]

        ordered = prompt_search.strategic_reorder(rows)

        assert [row["id"] for row in ordered] == ["best", "third", "fourth", "second"]

    def test_max_injection_cap(self, prompt_search):
        rows = [_row(f"chunk-{idx}", 0.012 - (idx * 0.0001)) for idx in range(10)]

        selected = prompt_search.select_adaptive_injection_rows(rows)

        assert len(selected) == 3  # MAX_ADAPTIVE_INJECTION=3


class TestPollutionFiltering:
    def test_eval_pollution_filter_project(self, prompt_search):
        rows = [
            _row("prod", 0.012, project="brainlayer"),
            _row("eval", 0.020, project="eval-sandbox"),
        ]

        filtered = prompt_search.filter_pollution_rows(rows)

        assert [row["id"] for row in filtered] == ["prod"]

    def test_eval_pollution_filter_tag(self, prompt_search):
        rows = [
            _row("prod", 0.012, tags='["memory","decision"]'),
            _row("eval", 0.020, tags='["eval-test","memory"]'),
        ]

        filtered = prompt_search.filter_pollution_rows(rows)

        assert [row["id"] for row in filtered] == ["prod"]


class TestSearchFallback:
    def test_fallback_to_fts_only(self, prompt_search, monkeypatch):
        fts_rows = [_row("fts-best", 0.0)]

        monkeypatch.setattr(
            prompt_search,
            "run_hybrid_search",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embed failed")),
        )
        monkeypatch.setattr(prompt_search, "run_fts_search", lambda *args, **kwargs: fts_rows)

        rows, used_hybrid = prompt_search.search_prompt_chunks(
            prompt="keyword fallback query",
            db_path="/tmp/test.db",
            keywords=["keyword", "fallback"],
            limit=8,
        )

        assert used_hybrid is False
        assert [row["id"] for row in rows] == ["fts-best"]
