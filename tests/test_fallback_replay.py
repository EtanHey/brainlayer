from __future__ import annotations

import subprocess
import os
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
        "scopes:\n"
        f'  {systems}: "systems"\n'
        f'  {narration}: "narrationlayer"\n'
        'default: "all"\n',
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
