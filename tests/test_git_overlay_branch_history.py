"""Focused tests for historical session branch attribution."""

import importlib.util
import json
from pathlib import Path

import pytest

_SOURCE = Path(__file__).parents[1] / "src/brainlayer/pipeline/git_overlay.py"
_SPEC = importlib.util.spec_from_file_location("git_overlay_under_test", _SOURCE)
assert _SPEC and _SPEC.loader
git_overlay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(git_overlay)


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _no_live_branch(cmd, *args, **kwargs):
    if cmd == ["rev-parse", "--abbrev-ref", "HEAD"]:
        raise AssertionError("historical branch must not come from current checkout")
    return None


def test_get_git_context_uses_recorded_branch_without_live_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(git_overlay, "_git_cmd", _no_live_branch)
    context = git_overlay.get_git_context(
        tmp_path / "repo", "start", "end", recorded_branch="feature/login"
    )
    assert context["branch"] == "feature/login"
    assert context["commit_shas"] == []
    assert context["files_changed"] == []


def test_get_git_context_leaves_branch_unknown_when_recording_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(git_overlay, "_git_cmd", _no_live_branch)
    context = git_overlay.get_git_context(tmp_path / "repo", "start", "end")
    assert context["branch"] is None


def test_extract_recorded_branch_requires_one_native_top_level_branch(tmp_path):
    path = tmp_path / "session.jsonl"
    _write_jsonl(path, [
        {"gitBranch": "feature/login", "message": {"gitBranch": "fake"}},
        {"type": "user", "message": {"gitBranch": "fake"}},
        {"gitBranch": "feature/login"},
    ])
    assert git_overlay.extract_recorded_branch(path) == "feature/login"


@pytest.mark.parametrize("value", [None, "", "  ", "HEAD", 3, ["main"]])
def test_extract_recorded_branch_ignores_invalid_or_detached_values(tmp_path, value):
    path = tmp_path / "session.jsonl"
    _write_jsonl(path, [{"gitBranch": value}])
    assert git_overlay.extract_recorded_branch(path) is None


def test_extract_recorded_branch_reports_conflicting_native_values(tmp_path, caplog):
    path = tmp_path / "session.jsonl"
    _write_jsonl(path, [{"gitBranch": "main"}, {"gitBranch": "feature/login"}])
    with caplog.at_level("WARNING"):
        assert git_overlay.extract_recorded_branch(path) is None
    assert "conflicting native gitBranch values" in caplog.text


def test_extract_recorded_branch_skips_malformed_lines(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text('{"gitBranch":"feature/login"}\nnot json\n')
    assert git_overlay.extract_recorded_branch(path) is None
