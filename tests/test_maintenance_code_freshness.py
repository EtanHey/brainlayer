"""Maintenance must refuse to run code that is not the merged code.

RED reproduces the 2026-08-05 deploy gap: `com.brainlayer.maintenance-nightly` runs
`~/Gits/brainlayer/.venv/bin/python -m brainlayer.maintenance`, and that venv is an
EDITABLE install — `import brainlayer` resolves to the working tree, not a built
artifact. On 2026-08-05 the tree was 8 commits behind origin/main, so #650's pause-
sentinel fix was merged on GitHub and absent on disk. The job would have run stale
code and reported success.

The property: it either runs the merged code, or it fails LOUDLY. It must never
silently run stale code and report ok.
"""

import os
import subprocess

import pytest

from brainlayer import maintenance


def _clean_test_git_env():
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=_clean_test_git_env(),
    ).stdout.strip()


@pytest.fixture
def source_checkout(monkeypatch):
    monkeypatch.setattr(maintenance, "_git_toplevel", lambda root: root.resolve())


def test_installed_package_nested_in_unrelated_git_repo_skips_source_freshness_guard(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    unrelated_repo = tmp_path / "homebrew"
    unrelated_repo.mkdir()
    _git(unrelated_repo, "init", "--quiet")
    _git(unrelated_repo, "config", "user.name", "Test User")
    _git(unrelated_repo, "config", "user.email", "test@example.com")
    marker = unrelated_repo / "marker"
    marker.write_text("base\n")
    _git(unrelated_repo, "add", "marker")
    _git(unrelated_repo, "commit", "--quiet", "-m", "base")
    merged_sha = _git(unrelated_repo, "rev-parse", "HEAD")
    _git(unrelated_repo, "update-ref", "refs/remotes/origin/main", merged_sha)
    marker.write_text("unrelated local change\n")
    _git(unrelated_repo, "commit", "--quiet", "-am", "unrelated head")

    with pytest.raises(maintenance.MaintenanceAbort, match="STALE"):
        maintenance._assert_running_merged_code(unrelated_repo, strict=True)

    installed_root = (
        unrelated_repo
        / "Cellar"
        / "brainlayer"
        / "1.5.35"
        / "libexec"
        / "venv"
        / "lib"
        / "python3.13"
        / "site-packages"
    )
    installed_root.mkdir(parents=True)
    installed_package = installed_root / "brainlayer"
    installed_package.mkdir()
    (installed_package / "maintenance.py").write_text("# installed package fixture\n")

    maintenance._assert_running_merged_code(installed_root, strict=True)

    assert "installed artifact" in capsys.readouterr().err


def test_source_freshness_ignores_hostile_inherited_git_routing(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    source_repo = tmp_path / "brainlayer-source"
    source_repo.mkdir()
    _git(source_repo, "init", "--quiet")
    _git(source_repo, "config", "user.name", "Source Fixture")
    _git(source_repo, "config", "user.email", "source@example.com")
    source_marker = source_repo / "marker"
    source_marker.write_text("stale\n")
    _git(source_repo, "add", "marker")
    _git(source_repo, "commit", "--quiet", "-m", "stale source")
    stale_sha = _git(source_repo, "rev-parse", "HEAD")
    source_marker.write_text("merged\n")
    _git(source_repo, "commit", "--quiet", "-am", "merged source")
    merged_sha = _git(source_repo, "rev-parse", "HEAD")
    _git(source_repo, "update-ref", "refs/remotes/origin/main", merged_sha)
    _git(source_repo, "checkout", "--quiet", "--detach", stale_sha)

    unrelated_repo = tmp_path / "unrelated"
    unrelated_repo.mkdir()
    _git(unrelated_repo, "init", "--quiet")
    _git(unrelated_repo, "config", "user.name", "Unrelated Fixture")
    _git(unrelated_repo, "config", "user.email", "unrelated@example.com")
    unrelated_marker = unrelated_repo / "marker"
    unrelated_marker.write_text("unrelated\n")
    _git(unrelated_repo, "add", "marker")
    _git(unrelated_repo, "commit", "--quiet", "-m", "unrelated head")
    unrelated_head = _git(unrelated_repo, "rev-parse", "HEAD")
    unrelated_config = _git(unrelated_repo, "config", "--local", "--list")

    monkeypatch.setenv("GIT_DIR", str(unrelated_repo / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(unrelated_repo))

    with pytest.raises(maintenance.MaintenanceAbort, match="STALE"):
        maintenance._assert_running_merged_code(source_repo, strict=True)

    assert _git(source_repo, "rev-parse", "HEAD") == stale_sha
    assert _git(unrelated_repo, "rev-parse", "HEAD") == unrelated_head
    assert _git(unrelated_repo, "config", "--local", "--list") == unrelated_config


def test_aborts_when_worktree_is_behind_merged_head(monkeypatch, source_checkout):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(maintenance, "_git_head_sha", lambda root: "aaaaaaa")
    monkeypatch.setattr(maintenance, "_git_merged_head_sha", lambda root: "bbbbbbb")

    with pytest.raises(maintenance.MaintenanceAbort) as exc:
        maintenance._assert_running_merged_code(maintenance.Path("/repo"))

    msg = str(exc.value)
    assert "aaaaaaa" in msg and "bbbbbbb" in msg, "abort must name both SHAs"
    assert "stale" in msg.lower()


def test_proceeds_when_worktree_matches_merged_head(monkeypatch, source_checkout):
    monkeypatch.setattr(maintenance, "_git_head_sha", lambda root: "deadbee")
    monkeypatch.setattr(maintenance, "_git_merged_head_sha", lambda root: "deadbee")

    maintenance._assert_running_merged_code(maintenance.Path("/repo"))


def test_unverifiable_remote_does_not_silently_pass_when_drifted(monkeypatch, source_checkout):
    """A failed fetch must not become an excuse to run stale code."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(maintenance, "_git_head_sha", lambda root: "aaaaaaa")
    monkeypatch.setattr(maintenance, "_git_merged_head_sha", lambda root: None)

    # Unknown remote state: cannot prove freshness, so it must not claim success.
    with pytest.raises(maintenance.MaintenanceAbort):
        maintenance._assert_running_merged_code(maintenance.Path("/repo"), strict=True)


def test_explicit_override_allows_drift(monkeypatch, source_checkout):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(maintenance, "_git_head_sha", lambda root: "aaaaaaa")
    monkeypatch.setattr(maintenance, "_git_merged_head_sha", lambda root: "bbbbbbb")
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_ALLOW_STALE", "1")

    maintenance._assert_running_merged_code(maintenance.Path("/repo"))


def test_guard_is_inert_under_pytest(monkeypatch, source_checkout):
    """Production keeps the guard; pytest opts out, matching this repo's convention.

    Without this, every run_maintenance test aborts as STALE simply because a feature
    branch differs from origin/main -- which is what a feature branch IS.
    """
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")
    monkeypatch.setattr(maintenance, "_git_head_sha", lambda root: "aaaaaaa")
    monkeypatch.setattr(maintenance, "_git_merged_head_sha", lambda root: "bbbbbbb")

    maintenance._assert_running_merged_code(maintenance.Path("/repo"))


def test_guard_fires_when_not_under_pytest(monkeypatch, source_checkout):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(maintenance, "_git_head_sha", lambda root: "aaaaaaa")
    monkeypatch.setattr(maintenance, "_git_merged_head_sha", lambda root: "bbbbbbb")

    with pytest.raises(maintenance.MaintenanceAbort):
        maintenance._assert_running_merged_code(maintenance.Path("/repo"))
