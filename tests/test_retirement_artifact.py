"""Verify immutable snapshots and resource isolation, including shared links."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from scripts.retirement_artifact import private_env, snapshot


@pytest.fixture
def fixture_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = private_env(tmp_path / "home")

    def _clean_test_git_env():
        return {key: value for key, value in env.items() if not key.startswith("GIT_")}

    for args in (
        ["init", "-q"],
        ["config", "user.name", "Fixture"],
        ["config", "user.email", "fixture@example.invalid"],
    ):
        subprocess.run(["git", *args], cwd=repo, env=_clean_test_git_env(), check=True, capture_output=True)
    (repo / "module.py").write_text("VALUE = 1\n")
    (repo / "alias.py").symlink_to("module.py")
    subprocess.run(["git", "add", "."], cwd=repo, env=_clean_test_git_env(), check=True)
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture"],
        cwd=repo,
        env=_clean_test_git_env(),
        check=True,
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, env=_clean_test_git_env(), text=True).strip()
    return repo, sha


def test_snapshot_uses_commit_not_dirty_source(fixture_repo, tmp_path):
    repo, sha = fixture_repo
    (repo / "module.py").write_text("DIRTY = True\n")
    (repo / "untracked.py").write_text("NOT_IN_WHEEL = True\n")
    destination = tmp_path / "archive"
    result = snapshot(repo, sha, destination)
    assert (destination / "module.py").read_text() == "VALUE = 1\n"
    assert (destination / "alias.py").is_symlink()
    assert result["files"]["alias.py"] == hashlib.sha256(b"module.py").hexdigest()
    assert "untracked.py" not in result["files"]
    assert result["sha"] == sha


def test_snapshot_requires_full_sha(fixture_repo, tmp_path):
    repo, sha = fixture_repo
    with pytest.raises(ValueError, match="full immutable"):
        snapshot(repo, sha[:8], tmp_path / "archive")


def test_private_env_discards_real_keys_and_git_config(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-inherited-key")
    monkeypatch.setenv("PYTHONPATH", "synthetic-inherited-src")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    env = private_env(tmp_path / "private")
    assert not set(env) & {"GOOGLE_API_KEY", "PYTHONPATH", "GIT_CONFIG_COUNT"}
    assert Path(env["HOME"]).is_relative_to(tmp_path)
    assert Path(env["BRAINLAYER_DB"]).is_relative_to(tmp_path)


def test_snapshot_refuses_existing_destination(fixture_repo, tmp_path):
    repo, sha = fixture_repo
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        snapshot(repo, sha, destination)
