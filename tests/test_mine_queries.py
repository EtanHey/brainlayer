"""Tests for C7 failed query mining script."""

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from mine_failed_queries import _extract_result_count, analyze_queries


class TestExtractResultCount:
    def test_json_total_field(self):
        assert _extract_result_count('{"total": 5, "results": [...]}') == 5

    def test_json_total_zero(self):
        assert _extract_result_count('{"total": 0}') == 0

    def test_n_results_pattern(self):
        assert _extract_result_count("Found 3 results for your query") == 3

    def test_no_results_text(self):
        assert _extract_result_count("No results found for this query") == 0

    def test_no_matches_text(self):
        assert _extract_result_count("No matches in the database") == 0

    def test_nothing_found(self):
        assert _extract_result_count("Nothing found") == 0

    def test_markdown_list_items(self):
        text = "- [chunk1] result one\n- [chunk2] result two\n- [chunk3] result three"
        assert _extract_result_count(text) == 3

    def test_empty_text(self):
        assert _extract_result_count("") == -1

    def test_none_text(self):
        assert _extract_result_count(None) == -1

    def test_unknown_format(self):
        assert _extract_result_count("Some random text without counts") == -1


class TestAnalyzeQueries:
    def test_groups_identical_queries(self):
        results = [
            ("test query", 0, ""),
            ("test query", 0, ""),
        ]
        zero, low, unknown = analyze_queries(results)
        assert len(zero) == 1
        assert zero[0]["times_searched"] == 2

    def test_case_insensitive_grouping(self):
        results = [
            ("Test Query", 0, ""),
            ("test query", 0, ""),
        ]
        zero, low, unknown = analyze_queries(results)
        assert len(zero) == 1
        assert zero[0]["times_searched"] == 2

    def test_zero_vs_low_classification(self):
        results = [
            ("zero query", 0, ""),
            ("low query", 1, ""),
        ]
        zero, low, unknown = analyze_queries(results)
        assert len(zero) == 1
        assert len(low) == 1
        assert zero[0]["normalized"] == "zero query"
        assert low[0]["normalized"] == "low query"

    def test_min_count_filter(self):
        results = [
            ("rare query", 0, ""),
            ("common query", 0, ""),
            ("common query", 0, ""),
        ]
        zero, low, unknown = analyze_queries(results, min_count=2)
        assert len(zero) == 1
        assert zero[0]["normalized"] == "common query"

    def test_project_tracking(self):
        results = [
            ("test query", 0, "brainlayer"),
            ("test query", 0, "golems"),
        ]
        zero, low, unknown = analyze_queries(results)
        assert set(zero[0]["projects"]) == {"brainlayer", "golems"}

    def test_unknown_results(self):
        results = [
            ("unknown query", -1, ""),
        ]
        zero, low, unknown = analyze_queries(results)
        assert len(zero) == 0
        assert len(low) == 0
        assert len(unknown) == 1


class TestExtractQueriesAndResults:
    def test_parses_jsonl_with_tool_calls(self, tmp_path):
        from mine_failed_queries import extract_queries_and_results

        # Create a minimal JSONL session file
        jsonl_path = tmp_path / "test_session.jsonl"
        tool_id = "toolu_test123"

        # Tool use line (assistant calling brain_search)
        tool_use_line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": "mcp__brainlayer__brain_search",
                            "input": {"query": "test search query"},
                        }
                    ]
                },
            }
        )

        # Tool result line
        tool_result_line = json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": [{"type": "text", "text": '{"total": 0}'}],
                        }
                    ]
                },
            }
        )

        jsonl_path.write_text(f"{tool_use_line}\n{tool_result_line}\n")

        results = extract_queries_and_results(str(jsonl_path))
        assert len(results) == 1
        assert results[0][0] == "test search query"
        assert results[0][1] == 0

    def test_handles_missing_file(self):
        from mine_failed_queries import extract_queries_and_results

        results = extract_queries_and_results("/nonexistent/path.jsonl")
        assert results == []

    def test_skips_short_queries(self, tmp_path):
        from mine_failed_queries import extract_queries_and_results

        jsonl_path = tmp_path / "test.jsonl"
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "brain_search",
                            "input": {"query": "hi"},
                        }
                    ]
                },
            }
        )
        jsonl_path.write_text(f"{line}\n")

        results = extract_queries_and_results(str(jsonl_path))
        assert len(results) == 0  # "hi" is too short (< 5 chars)
