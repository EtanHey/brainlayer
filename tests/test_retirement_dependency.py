"""The dependency baseline follows Git ancestry without a permanent SHA pin."""

import subprocess

import pytest

from scripts.retirement_artifact import private_env
from scripts.retirement_run import resolve_dependency


def test_dependency_uses_actual_tag_merge_base(tmp_path):
    env = private_env(tmp_path / "home")

    def _clean_test_git_env():
        return {key: value for key, value in env.items() if not key.startswith("GIT_")}

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, env=_clean_test_git_env(), text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Synthetic R11")
    git("config", "user.email", "synthetic@example.invalid")
    git("commit", "--allow-empty", "-m", "synthetic dependency baseline")
    base = git("rev-parse", "HEAD")
    git("tag", "dependency-baseline")
    git("commit", "--allow-empty", "-m", "synthetic candidate")
    head = git("rev-parse", "HEAD")
    assert resolve_dependency(repo, head, "dependency-baseline", env) == base
    assert resolve_dependency(repo, head, "main", env) == head
    with pytest.raises(subprocess.CalledProcessError):
        resolve_dependency(repo, head, "absent-reference", env)


@pytest.mark.parametrize("ref", [None, "", "--all"])
def test_missing_or_option_shaped_dependency_reference_fails(tmp_path, ref):
    with pytest.raises(ValueError):
        resolve_dependency(tmp_path, "a" * 40, ref, {})
