"""Search scope must follow Git repository identity, not worktree names."""

import importlib.util
import io
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from brainlayer import scoping


@pytest.fixture(params=["nested", "t3", "elsewhere"])
def workspace(tmp_path, request):
    repo = tmp_path / "Gits" / "repo-a"
    repo.mkdir(parents=True)

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "init")
    paths = {
        "nested": repo / ".worktrees" / "feature",
        "t3": tmp_path / ".t3" / "worktrees" / "repo-a" / "t3-id",
        "elsewhere": tmp_path / "unrelated-feature",
    }
    worktree = paths[request.param]
    git("worktree", "add", "--detach", str(worktree), "HEAD")
    return repo, worktree


def test_scope_from_worktree(workspace, monkeypatch):
    repo, worktree = workspace
    monkeypatch.chdir(worktree)
    monkeypatch.setattr(scoping, "_load_scopes", lambda: {"scopes": {str(repo): "repo-a"}, "default": "all"})
    assert scoping.resolve_project_scope() == "repo-a"


def test_scope_without_config(workspace, monkeypatch):
    _, worktree = workspace
    monkeypatch.chdir(worktree)
    monkeypatch.setattr(scoping, "_load_scopes", lambda: {})
    assert scoping.resolve_project_scope() == "repo-a"


def test_session_start_returns_repo_results(workspace, monkeypatch, capsys):
    _, worktree = workspace
    spec = importlib.util.spec_from_file_location(
        "session_start", Path(__file__).parents[1] / "hooks/brainlayer-session-start.py"
    )
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    db = worktree.parent / "hook.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE chunks(id TEXT, content TEXT, importance REAL, created_at TEXT, content_type TEXT, project TEXT)"
        )
        conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, content, tags)")
        for project in ("repo-a", "repo-b"):
            content = f"repo-a decision sentinel for {project}"
            conn.execute("INSERT INTO chunks VALUES (?, ?, 8, '2026-11-09', 'text', ?)", (project, content, project))
            conn.execute("INSERT INTO chunks_fts VALUES (?, ?, '')", (project, content))
    monkeypatch.setattr(hook, "get_db_path", lambda: db)
    monkeypatch.setattr(hook, "load_scoped_projects", lambda: {"repo-a": "repo-a"})
    monkeypatch.setattr(hook, "should_activate", lambda _: (True, False))
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps({"cwd": str(worktree)})))
    with pytest.raises(SystemExit):
        hook.main()
    output = capsys.readouterr().out
    assert "decision sentinel for repo-a" in output
    assert "decision sentinel for repo-b" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["search", "recall"])
async def test_handler_returns_repo_results(workspace, monkeypatch, surface):
    from brainlayer.mcp import search_handler as handler
    from brainlayer.vector_store import VectorStore
    from tests.test_consumer_scoping import _embed, _insert_chunk, _ScopedEmbeddingModel

    repo, worktree = workspace
    monkeypatch.chdir(worktree)
    monkeypatch.setattr(scoping, "_load_scopes", lambda: {"scopes": {str(repo): "repo-a"}})
    monkeypatch.setenv("BRAINLAYER_CONSUMER", "worker")
    store = VectorStore(repo.parent / "search.db")
    store._binary_index_available = False
    try:
        for project in ("repo-a", "repo-b"):
            _insert_chunk(
                store,
                chunk_id=project,
                content="worktree sentinel result",
                embedding=_embed("worktree sentinel result"),
                project=project,
            )
        monkeypatch.setattr(handler, "_helper_route_enabled", lambda: False)
        monkeypatch.setattr(handler, "_get_vector_store", lambda: store)
        monkeypatch.setattr(handler, "_get_embedding_model", lambda: _ScopedEmbeddingModel())
        monkeypatch.setattr(handler, "_expanded_fts_query", lambda *a, **kw: None)
        monkeypatch.setattr(handler, "_detect_entities", lambda *a, **kw: [])
        if surface == "search":
            _, result = await handler._brain_search(query="worktree sentinel", source="all", allow_helper_route=False)
        else:
            _, result = await handler._brain_recall(query="worktree sentinel", mode="search")
        assert {row["project"] for row in result["results"]} == {"repo-a"}
    finally:
        store.close()


def test_subdirectory_keeps_configured_scope(workspace, monkeypatch):
    repo, worktree = workspace
    (repo / "packages").mkdir()
    (worktree / "packages").mkdir()
    monkeypatch.chdir(worktree / "packages")
    monkeypatch.setattr(
        scoping, "_load_scopes", lambda: {"scopes": {str(repo): "repo-a", str(repo / "packages"): "repo-a-packages"}}
    )
    assert scoping.resolve_project_scope() == "repo-a-packages"


def test_non_git_path_is_unchanged(tmp_path):
    missing = tmp_path / "nonexistent"
    assert scoping.canonical_workspace_path(str(missing)) == str(missing)
