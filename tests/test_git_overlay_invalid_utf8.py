import importlib.util
import json
from pathlib import Path

_SOURCE = Path(__file__).parents[1] / "src/brainlayer/pipeline/git_overlay.py"
_SPEC = importlib.util.spec_from_file_location("git_overlay_invalid_utf8", _SOURCE)
assert _SPEC and _SPEC.loader
git_overlay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(git_overlay)


def test_session_timestamps_invalid_utf8_fails_closed(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(b'{"timestamp":"2026-01-01T00:00:00Z"}\n\xff\n')
    assert git_overlay.get_session_timestamps(path) == (None, None)


class _FakeStore:
    def __init__(self):
        self.contexts = {}
        self.interactions = []

    def get_session_context(self, session_id):
        return self.contexts.get(session_id)

    def store_session_context(self, **kwargs):
        self.contexts[kwargs["session_id"]] = kwargs

    def store_file_interactions(self, actions):
        self.interactions.extend(actions)
        return len(actions)


def test_overlay_skips_invalid_utf8_and_continues(monkeypatch, tmp_path):
    home = tmp_path / "home"
    project = home / ".claude" / "projects" / "-tmp-overlay-test"
    project.mkdir(parents=True)
    repo = Path("/tmp/overlay/test")
    repo.mkdir(parents=True, exist_ok=True)
    (project / "a-invalid.jsonl").write_bytes(b"{\"timestamp\":\"2026-01-01T00:00:00Z\"}\n\xff\n")
    (project / "b-valid.jsonl").write_text(
        json.dumps({"timestamp": "2026-01-01T00:00:00Z", "gitBranch": "feature/valid"}) + "\n"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    store = _FakeStore()
    result = git_overlay.run_git_overlay(store, project="-tmp-overlay-test")
    assert result == {"sessions_processed": 1, "file_interactions_added": 0}
    assert store.contexts["b-valid"]["branch"] is None
