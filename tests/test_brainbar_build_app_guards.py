"""Tests for BrainBar canonical build/install guards."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo: Path, branch: str = "main") -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", branch)
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")


def _write_tracked_file(repo: Path, rel_path: str, content: str) -> None:
    path = repo / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    _git(repo, "add", rel_path)


def _commit(repo: Path, message: str) -> None:
    _git(repo, "commit", "-m", message)


def _prepare_build_repo(tmp_path: Path, repo_name: str, branch: str = "main") -> tuple[Path, Path]:
    repo = tmp_path / repo_name
    _init_repo(repo, branch=branch)
    script_dir = repo / "brain-bar"
    script_dir.mkdir(parents=True, exist_ok=True)
    source_script = Path(__file__).resolve().parents[1] / "brain-bar" / "build-app.sh"
    target_script = script_dir / "build-app.sh"
    shutil.copy2(source_script, target_script)
    _write_tracked_file(repo, "README.md", "# test repo\n")
    _write_tracked_file(repo, "brain-bar/build-app.sh", target_script.read_text())
    _commit(repo, "chore: seed build script")
    return repo, target_script


def _run_build_script(
    repo: Path,
    script: Path,
    *,
    canonical_root: Path,
    home: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("BRAINBAR_APP_DIR", None)
    env["HOME"] = str(home)
    env["BRAINBAR_CANONICAL_REPO_ROOT"] = str(canonical_root)
    cmd = ["bash", str(script), "--dry-run"]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
    )


def test_build_app_allows_clean_canonical_repo_in_dry_run(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    home.mkdir()

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
    )

    assert result.returncode == 0
    assert str(home / "Applications" / "BrainBar.app") in result.stdout
    assert "LaunchAgent: canonical install" in result.stdout


def test_build_app_rejects_noncanonical_repo_without_force(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-worktree", branch="feat/ui-guards")
    home = tmp_path / "home"
    home.mkdir()
    canonical_root = tmp_path / "brainlayer-canonical"
    canonical_root.mkdir()

    result = _run_build_script(
        repo,
        script,
        canonical_root=canonical_root,
        home=home,
    )

    assert result.returncode != 0
    assert "--force-worktree-build" in result.stderr


def test_build_app_routes_forced_noncanonical_repo_to_dev_bundle(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-worktree", branch="feat/ui-guards")
    home = tmp_path / "home"
    home.mkdir()
    canonical_root = tmp_path / "brainlayer-canonical"
    canonical_root.mkdir()

    result = _run_build_script(
        repo,
        script,
        canonical_root=canonical_root,
        home=home,
        extra_args=["--force-worktree-build"],
    )

    assert result.returncode == 0
    assert str(home / "Applications" / "BrainBar-DEV-feat-ui-guards.app") in result.stdout
    assert "LaunchAgent: skipped for DEV worktree build" in result.stdout


def test_build_app_rejects_dirty_canonical_repo_without_force(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    home.mkdir()
    (repo / "README.md").write_text("# dirty repo\n")

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
    )

    assert result.returncode != 0
    assert "dirty" in result.stderr.lower()
    assert "README.md" in result.stderr


def test_build_app_allows_dirty_canonical_repo_with_force_dirty(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    home.mkdir()
    (repo / "README.md").write_text("# dirty repo\n")

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_args=["--force-dirty"],
    )

    assert result.returncode == 0
    assert str(home / "Applications" / "BrainBar.app") in result.stdout


def test_build_app_routes_forced_noncanonical_repo_to_sanitized_dev_bundle(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-worktree", branch="feat/space-case")
    home = tmp_path / "home"
    home.mkdir()
    canonical_root = tmp_path / "brainlayer-canonical"
    canonical_root.mkdir()

    result = _run_build_script(
        repo,
        script,
        canonical_root=canonical_root,
        home=home,
        extra_args=["--force-worktree-build"],
    )

    assert result.returncode == 0
    assert str(home / "Applications" / "BrainBar-DEV-feat-space-case.app") in result.stdout


def test_build_app_allows_symlinked_canonical_root_in_dry_run(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    home.mkdir()
    symlink_root = tmp_path / "brainlayer-link"
    symlink_root.symlink_to(repo, target_is_directory=True)

    result = _run_build_script(
        repo,
        script,
        canonical_root=symlink_root,
        home=home,
    )

    assert result.returncode == 0
    assert str(home / "Applications" / "BrainBar.app") in result.stdout


def test_build_app_ignores_parent_brainbar_app_dir_in_canonical_dry_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("BRAINBAR_APP_DIR", str(tmp_path / "leaked.app"))

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
    )

    assert result.returncode == 0
    assert str(home / "Applications" / "BrainBar.app") in result.stdout
    assert "leaked.app" not in result.stdout


def test_build_app_rejects_untracked_dirty_repo_even_when_status_hides_untracked_files(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    home.mkdir()
    _git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "UNTRACKED.txt").write_text("untracked\n")

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
    )

    assert result.returncode != 0
    assert "dirty" in result.stderr.lower()
    assert "UNTRACKED.txt" in result.stderr
