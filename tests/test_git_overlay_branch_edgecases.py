import builtins
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "src/brainlayer/pipeline/git_overlay.py"
SPEC = importlib.util.spec_from_file_location("git_overlay_edge", SOURCE)
assert SPEC and SPEC.loader
git_overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(git_overlay)


def test_named_and_detached_records_are_ambiguous(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps({"gitBranch": "feature/login"}) + "\n" + json.dumps({"gitBranch": "HEAD"}) + "\n")
    assert git_overlay.extract_recorded_branch(path) is None


def test_read_failure_fails_closed(monkeypatch, tmp_path):
    path = tmp_path / "session.jsonl"
    original_open = builtins.open

    def fail_open(target, *args, **kwargs):
        if target == path:
            raise OSError("unreadable")
        return original_open(target, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_open)
    assert git_overlay.extract_recorded_branch(path) is None


def test_blank_lines_do_not_discard_valid_branch(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text('\n{"gitBranch":"feature/login"}\n\n')
    assert git_overlay.extract_recorded_branch(path) == "feature/login"


@pytest.mark.parametrize("payload", [
    b'{"gitBranch":"feature/login"}\n\xff\n',
    b'\xff{"gitBranch":"feature/login"}\n',
])
def test_invalid_utf8_fails_closed(tmp_path, payload):
    path = tmp_path / "session.jsonl"
    path.write_bytes(payload)
    assert git_overlay.extract_recorded_branch(path) is None
