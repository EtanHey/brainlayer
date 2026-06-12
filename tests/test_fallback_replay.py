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
