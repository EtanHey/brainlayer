"""Regression coverage for pytest's shared-Git containment boundary."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean_git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, env=_clean_git_env()).strip()


def _linked_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(repo)],
        check=True,
        env=_clean_git_env(),
    )
    _git(repo, "config", "user.name", "Guard Fixture")
    _git(repo, "config", "user.email", "guard@example.com")
    (repo / "seed.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-qm", "first")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "seed.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "second")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")

    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "-q", "-b", "guard-linked", str(linked), "HEAD")
    tests_dir = linked / "tests"
    tests_dir.mkdir()
    shutil.copy2(REPO_ROOT / "tests" / "conftest.py", tests_dir / "conftest.py")
    return repo, linked, Path(_git(linked, "rev-parse", "--absolute-git-dir"))


def _hook_env(repo: Path, linked: Path, git_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_DIR": str(git_dir),
            "GIT_INDEX_FILE": str(git_dir / "index"),
            "GIT_WORK_TREE": str(linked),
            "GIT_COMMON_DIR": str(repo / ".git"),
            "PYTHONPATH": os.pathsep.join(filter(None, (str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")))),
        }
    )
    return env


def _run_nested_pytest(linked: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_probe.py"],
        cwd=linked,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_conftest_scrubs_hook_git_environment_before_tests_and_subprocesses(tmp_path: Path) -> None:
    repo, linked, git_dir = _linked_repo(tmp_path)
    original_head = _git(repo, "rev-parse", "refs/remotes/origin/main")
    original_config = (repo / ".git" / "config").read_bytes()
    (linked / "tests" / "test_probe.py").write_text(
        """
import os
import subprocess


def test_probe(tmp_path):
    inherited = sorted(key for key in os.environ if key.startswith("GIT_"))
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    subprocess.run(["git", "-C", str(fixture), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(fixture), "config", "user.name", "Nested Fixture"], check=True)
    assert inherited == []
""".lstrip(),
        encoding="utf-8",
    )

    result = _run_nested_pytest(linked, _hook_env(repo, linked, git_dir))

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(repo, "rev-parse", "refs/remotes/origin/main") == original_head
    assert (repo / ".git" / "config").read_bytes() == original_config
    assert _git(linked, "rev-parse", "--is-bare-repository") == "false"


def test_session_guard_fails_loudly_when_shared_git_state_changes(tmp_path: Path) -> None:
    repo, linked, git_dir = _linked_repo(tmp_path)
    first = _git(repo, "rev-parse", "HEAD^")
    (linked / "tests" / "test_probe.py").write_text(
        """
import os
import subprocess


def test_probe():
    common = os.environ["TEST_SHARED_GIT_DIR"]
    subprocess.run(["git", "--git-dir", common, "config", "core.bare", "true"], check=True)
    subprocess.run(["git", "--git-dir", common, "config", "user.name", "Mutated Fixture"], check=True)
    subprocess.run(
        ["git", "--git-dir", common, "update-ref", "refs/remotes/origin/main", os.environ["TEST_OLD_HEAD"]],
        check=True,
    )
""".lstrip(),
        encoding="utf-8",
    )
    env = _hook_env(repo, linked, git_dir)
    env["TEST_SHARED_GIT_DIR"] = str(repo / ".git")
    env["TEST_OLD_HEAD"] = first

    result = _run_nested_pytest(linked, env)
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert "shared Git state changed during pytest" in output
    assert "core.bare" in output
    assert "user.*" in output
    assert "origin/main" in output
