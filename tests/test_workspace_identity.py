import subprocess

from brainlayer.workspace_identity import canonical_project_from_workspace


def test_repo_root_uses_root_name(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "brainlayer.workspace_identity.subprocess.run",
        lambda *a, **k: type("R", (), {"stdout": f"{tmp_path}\n{tmp_path / '.git'}\n"})(),
    )
    (tmp_path / ".git").mkdir()
    assert canonical_project_from_workspace(str(tmp_path)) == tmp_path.name


def test_worktree_uses_common_git_parent(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    worktree = root / ".worktrees" / "feature"
    common = root / ".git"
    common.mkdir(parents=True)
    worktree.mkdir(parents=True)
    monkeypatch.setattr(
        "brainlayer.workspace_identity.subprocess.run",
        lambda *a, **k: type("R", (), {"stdout": f"{worktree}\n{common}\n"})(),
    )
    assert canonical_project_from_workspace(str(worktree)) == "repo"


def test_invalid_git_identity_falls_back_to_leaf(tmp_path, monkeypatch):
    worktree = tmp_path / ".worktrees" / "feature"
    worktree.mkdir(parents=True)

    def fail(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("brainlayer.workspace_identity.subprocess.run", fail)
    assert canonical_project_from_workspace(str(worktree)) == "feature"


def test_git_identity_is_cached(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "brainlayer.workspace_identity.subprocess.run",
        lambda *a, **k: calls.append(1) or type("R", (), {"stdout": f"{tmp_path}\n{tmp_path / '.git'}\n"})(),
    )
    (tmp_path / ".git").mkdir()
    assert canonical_project_from_workspace(str(tmp_path)) == tmp_path.name
    assert canonical_project_from_workspace(str(tmp_path)) == tmp_path.name
    assert len(calls) == 1


def test_real_linked_worktree_call_sites_from_unrelated_cwd(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "README").write_text("x")
    (repo / "src").mkdir()
    subprocess.run(["git", "-C", str(repo), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    worktree = tmp_path / "linked" / "feature"
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-qb", "feature", str(worktree)], check=True)
    (tmp_path / "unrelated").mkdir()
    monkeypatch.chdir(tmp_path / "unrelated")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong.git"))
    assert canonical_project_from_workspace(str(repo)) == "repo"
    assert canonical_project_from_workspace(str(repo / "src")) == "repo"
    assert canonical_project_from_workspace(str(worktree)) == "repo"
    (worktree / "src").mkdir()
    assert canonical_project_from_workspace(str(worktree / "src")) == "repo"
    from brainlayer.ingest.codex import _infer_project_from_cwd
    from brainlayer.watcher_bridge import _extract_workspace_project

    for workspace in (repo, repo / "src", worktree, worktree / "src"):
        assert _infer_project_from_cwd(str(workspace)) == "repo"
        assert _extract_workspace_project({"cwd": str(workspace)}) == "repo"
