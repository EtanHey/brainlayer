from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def _git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git_init(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(("git", "init", "-q"), cwd=path, env=_git_env(), check=True)


def _pending_file(repo: Path, relative: str, *, project: str | None = None) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    project_line = f"project: {project}\n" if project else ""
    path.write_text(
        "---\n"
        "intended_brain_store: true\n"
        "importance: 8\n"
        "tags: [user-correction, systems]\n"
        f"{project_line}"
        "timestamp: 2026-06-12T19:54:07+03:00\n"
        "reason: transport_closed\n"
        "retry_attempted: true\n"
        "chunk_id:\n"
        "---\n"
        "original body stays byte-for-byte\n",
        encoding="utf-8",
    )
    return path


def test_attribution_uses_scopes_mapping_for_originating_repo(tmp_path):
    from brainlayer.fallback_replay import load_scope_map, parse_fallback_file

    systems = tmp_path / "systems"
    narration = tmp_path / "narrationlayer"
    _git_init(systems)
    _git_init(narration)
    scopes_path = tmp_path / "scopes.yaml"
    scopes_path.write_text(
        f'scopes:\n  {systems}: "systems"\n  {narration}: "narrationlayer"\ndefault: "all"\n',
        encoding="utf-8",
    )

    systems_entry = parse_fallback_file(
        _pending_file(systems, "docs.local/decisions/systems.md"),
        scope_map=load_scope_map(scopes_path),
    )
    narration_entry = parse_fallback_file(
        _pending_file(narration, "docs.local/decisions/narration.md"),
        scope_map=load_scope_map(scopes_path),
    )

    assert systems_entry.project == "systems"
    assert systems_entry.origin_repo_path == systems
    assert narration_entry.project == "narrationlayer"
    assert narration_entry.origin_repo_path == narration


def test_frontmatter_project_overrides_scopes_mapping(tmp_path):
    from brainlayer.fallback_replay import load_scope_map, parse_fallback_file

    repo = tmp_path / "systems"
    _git_init(repo)
    scopes_path = tmp_path / "scopes.yaml"
    scopes_path.write_text(f'scopes:\n  {repo}: "systems"\n', encoding="utf-8")

    entry = parse_fallback_file(
        _pending_file(repo, "docs.local/decisions/explicit.md", project="cmuxlayer"),
        scope_map=load_scope_map(scopes_path),
    )

    assert entry.project == "cmuxlayer"


def test_replay_preserves_attribution_and_updates_frontmatter_only(tmp_path):
    from brainlayer.fallback_replay import load_scope_map, parse_fallback_file, replay_entry

    repo = tmp_path / "narrationlayer"
    _git_init(repo)
    scopes_path = tmp_path / "scopes.yaml"
    scopes_path.write_text(f'scopes:\n  {repo}: "narrationlayer"\n', encoding="utf-8")
    path = _pending_file(repo, "docs.local/decisions/pending.md")
    original_body = path.read_text(encoding="utf-8").split("---\n", 2)[2]
    calls = []

    def store_func(**kwargs):
        calls.append(kwargs)
        return {"id": "manual-replayed123", "related": []}

    entry = parse_fallback_file(path, scope_map=load_scope_map(scopes_path))
    result = replay_entry(entry, store_func=store_func, replayed_by="phase-1-test")

    assert result.chunk_id == "manual-replayed123"
    assert calls == [
        {
            "content": "original body stays byte-for-byte\n",
            "memory_type": "note",
            "project": "narrationlayer",
            "tags": ["user-correction", "systems"],
            "importance": 8,
            "created_at": "2026-06-12T19:54:07+03:00",
            "fallback_source_path": str(path),
            "origin_repo_path": str(repo),
            "replayed_by": "phase-1-test",
        }
    ]

    updated = path.read_text(encoding="utf-8")
    assert "chunk_id: manual-replayed123" in updated
    assert updated.endswith(original_body)


def test_replay_wraps_scalar_frontmatter_tag_as_single_tag(tmp_path):
    from brainlayer.fallback_replay import parse_fallback_file, replay_entry

    repo = tmp_path / "systems"
    _git_init(repo)
    path = repo / "docs.local" / "decisions" / "pending.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nintended_brain_store: true\ntags: user-correction\nchunk_id:\n---\nscalar tag body\n",
        encoding="utf-8",
    )
    calls = []

    def store_func(**kwargs):
        calls.append(kwargs)
        return {"chunk_id": "manual-scalar-tag"}

    entry = parse_fallback_file(path)
    replay_entry(entry, store_func=store_func, replayed_by="phase-1-test")

    assert calls[0]["tags"] == ["user-correction"]


def test_parse_fallback_file_accepts_frontmatter_closing_delimiter_at_eof(tmp_path):
    from brainlayer.fallback_replay import parse_fallback_file

    repo = tmp_path / "systems"
    _git_init(repo)
    path = repo / "docs.local" / "decisions" / "frontmatter-only.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "intended_brain_store: true\n"
        "memory_type: decision\n"
        "project: systems\n"
        "tags: [fallback]\n"
        "timestamp: 2026-06-12T19:54:07+03:00\n"
        "---",
        encoding="utf-8",
    )

    entry = parse_fallback_file(path)

    assert entry.frontmatter["memory_type"] == "decision"
    assert entry.frontmatter["tags"] == ["fallback"]
    assert entry.frontmatter["timestamp"] == "2026-06-12T19:54:07+03:00"
    assert entry.body == ""


def test_parse_fallback_file_ignores_non_mapping_frontmatter(tmp_path):
    from brainlayer.fallback_replay import is_pending_entry, parse_fallback_file

    repo = tmp_path / "systems"
    _git_init(repo)
    path = repo / "docs.local" / "decisions" / "bad-frontmatter.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n42\n---\nbody\n", encoding="utf-8")

    entry = parse_fallback_file(path)

    assert entry.frontmatter == {}
    assert entry.body == "body\n"
    assert is_pending_entry(entry) is False


def test_replay_errors_when_store_result_has_no_chunk_id(tmp_path):
    from brainlayer.fallback_replay import is_pending_entry, parse_fallback_file, replay_entry

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/missing-id.md")
    entry = parse_fallback_file(path)

    result = replay_entry(entry, store_func=lambda **_kwargs: {"related": []}, replayed_by="phase-1-test")
    updated_entry = parse_fallback_file(path)

    assert result.attempted is True
    assert result.chunk_id is None
    assert result.error == "store result did not include a chunk_id"
    assert is_pending_entry(updated_entry) is True
    assert updated_entry.frontmatter["retry_attempted"] is True


def test_replay_preserves_stored_chunk_id_when_frontmatter_update_fails(tmp_path, monkeypatch):
    import brainlayer.fallback_replay as fallback_replay

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/write-fails.md")
    entry = fallback_replay.parse_fallback_file(path)

    def fail_write(*_args, **_kwargs):
        raise OSError("read-only fallback file")

    monkeypatch.setattr(fallback_replay, "_write_frontmatter", fail_write)

    result = fallback_replay.replay_entry(
        entry, store_func=lambda **_kwargs: {"chunk_id": "manual-stored"}, replayed_by="phase-1-test"
    )

    assert result.attempted is True
    assert result.chunk_id == "manual-stored"
    assert (
        result.error
        == "stored chunk_id=manual-stored but failed to update fallback frontmatter: read-only fallback file"
    )


def test_replay_returns_missing_chunk_id_error_when_marker_write_fails(tmp_path, monkeypatch):
    import brainlayer.fallback_replay as fallback_replay

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/missing-id-and-write-fail.md")
    entry = fallback_replay.parse_fallback_file(path)

    def fail_write(*_args, **_kwargs):
        raise OSError("read-only fallback file")

    monkeypatch.setattr(fallback_replay, "_write_frontmatter", fail_write)

    result = fallback_replay.replay_entry(
        entry, store_func=lambda **_kwargs: {"related": []}, replayed_by="phase-1-test"
    )

    assert result.attempted is True
    assert result.chunk_id is None
    assert (
        result.error == "store result did not include a chunk_id; write_replay_attempt failed: read-only fallback file"
    )


def test_replay_returns_store_error_when_failed_attempt_marker_write_fails(tmp_path, monkeypatch):
    import brainlayer.fallback_replay as fallback_replay

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/store-and-write-fail.md")
    entry = fallback_replay.parse_fallback_file(path)

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    def fail_store(**_kwargs):
        raise RuntimeError("store failed")

    monkeypatch.setattr(fallback_replay, "_write_frontmatter", fail_write)

    result = fallback_replay.replay_entry(entry, store_func=fail_store, replayed_by="phase-1-test")

    assert result.attempted is True
    assert result.chunk_id is None
    assert result.error == "store failed: store failed; write_replay_attempt failed: disk full"


def test_replay_cli_apply_exits_nonzero_when_any_replay_errors(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "replay_brain_store_fallbacks.py"
    spec = importlib.util.spec_from_file_location("replay_brain_store_fallbacks_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    repo = tmp_path / "systems"
    path = _pending_file(repo, "docs.local/decisions/failing.md")
    entry = module.parse_fallback_file(path)

    class DummyStore:
        def close(self):
            pass

    monkeypatch.setattr(module, "load_scope_map", lambda _path: {})
    monkeypatch.setattr(module, "inventory", lambda _root, *, scope_map: ([entry], []))
    monkeypatch.setattr(module, "VectorStore", lambda _db: DummyStore())
    monkeypatch.setattr(module, "store_memory", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("store failed")))
    monkeypatch.setattr(sys, "argv", ["replay_brain_store_fallbacks.py", "--apply", "--gits-root", str(tmp_path)])

    assert module.main() == 1
