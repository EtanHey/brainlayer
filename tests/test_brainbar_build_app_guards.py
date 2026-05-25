"""Tests for BrainBar canonical build/install guards."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _clean_git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        env=_clean_git_env(),
        check=True,
        capture_output=True,
        text=True,
    )


def _git_stdout(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=_clean_git_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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
    dry_run: bool = True,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _clean_git_env()
    env.pop("BRAINBAR_APP_DIR", None)
    env["HOME"] = str(home)
    env["BRAINBAR_CANONICAL_REPO_ROOT"] = str(canonical_root)
    if extra_env:
        env.update(extra_env)
    cmd = ["bash", str(script)]
    if dry_run:
        cmd.append("--dry-run")
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
    )


def _create_fake_bundle(apps_dir: Path, name: str, git_commit: str | None = None) -> Path:
    bundle = apps_dir / name
    contents = bundle / "Contents"
    contents.mkdir(parents=True, exist_ok=True)
    git_commit_xml = ""
    if git_commit is not None:
        git_commit_xml = f"""
    <key>GitCommit</key>
    <string>{git_commit}</string>"""
    (contents / "Info.plist").write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>{git_commit_xml}
</dict>
</plist>
"""
    )
    return bundle


def _prepare_bundle_inputs(repo: Path) -> None:
    bundle_dir = repo / "brain-bar" / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "Info.plist").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
</dict>
</plist>
"""
    )
    for label in ("com.brainlayer.brainbar", "com.brainlayer.brainbar-daemon"):
        (bundle_dir / f"{label}.plist").write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
</dict>
</plist>
"""
        )
    _git(repo, "add", "brain-bar/bundle")
    _commit(repo, "test: add bundle inputs")


def _prepare_fake_build_tools(tmp_path: Path) -> tuple[Path, Path]:
    tool_dir = tmp_path / "fake-tools"
    bin_dir = tmp_path / "fake-swift-bin"
    tool_dir.mkdir()
    bin_dir.mkdir()
    (bin_dir / "BrainBar").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "BrainBarDaemon").write_text("#!/usr/bin/env bash\nexit 0\n")
    os.chmod(bin_dir / "BrainBar", 0o755)
    os.chmod(bin_dir / "BrainBarDaemon", 0o755)
    (tool_dir / "swift").write_text(
        """#!/usr/bin/env bash
if [[ "$*" == *"--show-bin-path"* ]]; then
  printf '%s\n' "$BRAINBAR_FAKE_BIN_DIR"
fi
exit 0
"""
    )
    (tool_dir / "codesign").write_text(
        """#!/usr/bin/env bash
if [[ "$*" == *"-dv"* ]]; then
  printf 'Authority=%s\n' "$BRAINBAR_CODESIGN_IDENTITY"
fi
exit 0
"""
    )
    (tool_dir / "plistbuddy").write_text(
        """#!/usr/bin/env bash
if [[ "$2" == "Print :GitCommit" && -f "$3" ]]; then
  awk '
    found { gsub(/^[[:space:]]*<string>|<\\/string>[[:space:]]*$/, ""); print; exit }
    /<key>GitCommit<\\/key>/ { found=1 }
  ' "$3"
  exit 0
fi
if [[ "$2" == Print* ]]; then
  exit 1
fi
exit 0
"""
    )
    (tool_dir / "launchctl").write_text(
        """#!/usr/bin/env bash
if [[ "$1" == "kickstart" && -n "${BRAINBAR_SOCKET_PATH:-}" && ! -S "$BRAINBAR_SOCKET_PATH" ]]; then
  python3 - "$BRAINBAR_SOCKET_PATH" <<'PY' >/dev/null 2>&1 &
import os
import socket
import sys

path = sys.argv[1]
try:
    os.unlink(path)
except FileNotFoundError:
    pass
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(path)
server.listen(1)
conn, _ = server.accept()
conn.close()
server.close()
PY
  disown
fi
exit 0
"""
    )
    (tool_dir / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n")
    (tool_dir / "killall").write_text("#!/usr/bin/env bash\nexit 0\n")
    for tool in ("swift", "codesign", "plistbuddy", "launchctl", "pgrep", "killall"):
        os.chmod(tool_dir / tool, 0o755)
    return tool_dir, bin_dir


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
    assert "UI LaunchAgent: canonical install" in result.stdout
    assert "Daemon LaunchAgent: canonical install" in result.stdout


def test_canonical_build_removes_only_stale_dev_bundles(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    apps_dir = home / "Applications"
    apps_dir.mkdir(parents=True)
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    remote = tmp_path / "origin.git"
    _git(remote.parent, "init", "--bare", remote.name)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    main_sha = _git_stdout(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "feat/active-dev")
    _git(repo, "branch", "feat/old-dev")
    _git(repo, "tag", "feat-missing")
    worktree = tmp_path / "active-worktree"
    _git(repo, "worktree", "add", str(worktree), "feat/active-dev")

    merged_bundle = _create_fake_bundle(apps_dir, "BrainBar-DEV-feat-merged.app", main_sha)
    missing_branch_bundle = _create_fake_bundle(apps_dir, "BrainBar-DEV-feat-missing.app")
    active_bundle = _create_fake_bundle(apps_dir, "BrainBar-DEV-feat-active-dev.app", main_sha)
    old_age_bundle = _create_fake_bundle(apps_dir, "BrainBar-DEV-feat-old-dev.app")
    old_mtime = 0
    os.utime(old_age_bundle, (old_mtime, old_mtime))
    _prepare_bundle_inputs(repo)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        dry_run=False,
        extra_env={
            "BRAINBAR_CODESIGN_IDENTITY": "Test Identity",
            "BRAINBAR_DEV_STALE_DAYS": "1",
            "BRAINBAR_FAKE_BIN_DIR": str(bin_dir),
            "BRAINBAR_PLIST_BUDDY": str(tool_dir / "plistbuddy"),
            "BRAINBAR_SOCKET_PATH": f"/tmp/brainbar-test-{os.getpid()}.sock",
            "BRAINBAR_SOCKET_WAIT_ATTEMPTS": "1",
            "PATH": f"{tool_dir}{os.pathsep}{os.environ['PATH']}",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "Cleaning stale DEV bundle: BrainBar-DEV-feat-merged.app" in result.stdout
    assert "Cleaning stale DEV bundle: BrainBar-DEV-feat-missing.app" in result.stdout
    assert "Cleaning stale DEV bundle: BrainBar-DEV-feat-old-dev.app" in result.stdout
    assert not merged_bundle.exists()
    assert not missing_branch_bundle.exists()
    assert not old_age_bundle.exists()
    assert active_bundle.exists()


def test_canonical_dry_run_lists_dev_bundle_cleanup_without_removing(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    apps_dir = home / "Applications"
    apps_dir.mkdir(parents=True)
    remote = tmp_path / "origin.git"
    _git(remote.parent, "init", "--bare", remote.name)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    main_sha = _git_stdout(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "feat/active-dev")
    worktree = tmp_path / "active-worktree"
    _git(repo, "worktree", "add", str(worktree), "feat/active-dev")

    bundles = [
        _create_fake_bundle(apps_dir, "BrainBar-DEV-feat-merged.app", main_sha),
        _create_fake_bundle(apps_dir, "BrainBar-DEV-feat-missing.app"),
        _create_fake_bundle(apps_dir, "BrainBar-DEV-feat-active-dev.app"),
    ]

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
    )

    assert result.returncode == 0, result.stderr
    for bundle in bundles:
        assert bundle.exists()
    assert "would clean stale DEV bundle: BrainBar-DEV-feat-merged.app" in result.stdout
    assert "would clean stale DEV bundle: BrainBar-DEV-feat-missing.app" in result.stdout
    assert "Keeping DEV bundle: BrainBar-DEV-feat-active-dev.app" in result.stdout


def test_canonical_dev_cleanup_scans_home_applications_when_app_dir_is_overridden(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    apps_dir = home / "Applications"
    apps_dir.mkdir(parents=True)
    custom_app_dir = tmp_path / "custom" / "BrainBar-Custom.app"
    remote = tmp_path / "origin.git"
    _git(remote.parent, "init", "--bare", remote.name)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    main_sha = _git_stdout(repo, "rev-parse", "HEAD")
    bundle = _create_fake_bundle(apps_dir, "BrainBar-DEV-feat-merged.app", main_sha)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={"BRAINBAR_APP_DIR": str(custom_app_dir)},
    )

    assert result.returncode == 0, result.stderr
    assert str(custom_app_dir) in result.stdout
    assert bundle.name in result.stdout
    assert "would clean stale DEV bundle" in result.stdout
    assert bundle.exists()


def test_canonical_dev_cleanup_preserves_local_branch_named_origin_prefix(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    apps_dir = home / "Applications"
    apps_dir.mkdir(parents=True)
    _git(repo, "branch", "origin/active-dev")
    worktree = tmp_path / "origin-active-worktree"
    _git(repo, "worktree", "add", str(worktree), "origin/active-dev")
    bundle = _create_fake_bundle(apps_dir, "BrainBar-DEV-origin-active-dev.app")

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
    )

    assert result.returncode == 0, result.stderr
    assert "Keeping DEV bundle: BrainBar-DEV-origin-active-dev.app" in result.stdout
    assert bundle.exists()


def test_canonical_dev_cleanup_preserves_detached_worktree_bundle(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    apps_dir = home / "Applications"
    apps_dir.mkdir(parents=True)
    detached_sha = _git_stdout(repo, "rev-parse", "--short", "HEAD")
    worktree = tmp_path / "detached-worktree"
    _git(repo, "worktree", "add", "--detach", str(worktree), "HEAD")
    bundle = _create_fake_bundle(apps_dir, f"BrainBar-DEV-detached-{detached_sha}.app")

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
    )

    assert result.returncode == 0, result.stderr
    assert f"Keeping DEV bundle: BrainBar-DEV-detached-{detached_sha}.app" in result.stdout
    assert bundle.exists()


def test_canonical_dev_cleanup_preserves_checked_out_sanitized_branch_collision(
    tmp_path: Path,
) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    apps_dir = home / "Applications"
    apps_dir.mkdir(parents=True)
    remote = tmp_path / "origin.git"
    _git(remote.parent, "init", "--bare", remote.name)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    main_sha = _git_stdout(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "feat-z-collision")
    _git(repo, "branch", "feat/z/collision")
    worktree = tmp_path / "collision-worktree"
    _git(repo, "worktree", "add", str(worktree), "feat/z/collision")
    bundle = _create_fake_bundle(apps_dir, "BrainBar-DEV-feat-z-collision.app", main_sha)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
    )

    assert result.returncode == 0, result.stderr
    assert "Keeping DEV bundle: BrainBar-DEV-feat-z-collision.app" in result.stdout
    assert bundle.exists()


def test_canonical_dev_cleanup_reads_gnu_stat_mtime(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    apps_dir = home / "Applications"
    apps_dir.mkdir(parents=True)
    _git(repo, "branch", "feat/old-dev")
    bundle = _create_fake_bundle(apps_dir, "BrainBar-DEV-feat-old-dev.app")
    fake_stat_dir = tmp_path / "fake-gnu-stat"
    fake_stat_dir.mkdir()
    (fake_stat_dir / "stat").write_text(
        """#!/usr/bin/env bash
if [[ "$1" == "-c" && "$2" == "%Y" ]]; then
  printf '0\n'
  exit 0
fi
if [[ "$1" == "-f" ]]; then
  printf '  File: "%s"\n' "$3"
  exit 0
fi
exit 1
"""
    )
    os.chmod(fake_stat_dir / "stat", 0o755)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={
            "BRAINBAR_DEV_STALE_DAYS": "1",
            "PATH": f"{fake_stat_dir}{os.pathsep}{os.environ['PATH']}",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "would clean stale DEV bundle: BrainBar-DEV-feat-old-dev.app" in result.stdout
    assert bundle.exists()


def test_build_app_helpers_ignore_parent_git_hook_env(tmp_path: Path, monkeypatch) -> None:
    parent_repo = tmp_path / "parent"
    _init_repo(parent_repo)
    _write_tracked_file(parent_repo, "README.md", "# parent repo\n")
    _commit(parent_repo, "feat: parent commit")

    monkeypatch.setenv("GIT_DIR", str(parent_repo / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(parent_repo))

    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    home.mkdir()

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
    )

    repo_subjects = _git_stdout(repo, "log", "--format=%s").splitlines()
    parent_subjects = _git_stdout(parent_repo, "log", "--format=%s").splitlines()

    assert result.returncode == 0
    assert repo_subjects == ["chore: seed build script"]
    assert parent_subjects == ["feat: parent commit"]


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
    assert "LaunchAgents: skipped for DEV worktree build" in result.stdout


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


def test_run_build_script_strips_parent_brainbar_app_dir_for_test_isolation(
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


def test_build_app_honors_explicit_brainbar_app_dir_for_canonical_dry_run(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    home.mkdir()
    explicit_app_dir = tmp_path / "custom" / "BrainBar-Custom.app"

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={"BRAINBAR_APP_DIR": str(explicit_app_dir)},
    )

    assert result.returncode == 0
    assert str(explicit_app_dir) in result.stdout


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


def test_brainbar_daemon_launchagent_runs_interactive_daemon_binary() -> None:
    plist = Path(__file__).resolve().parents[1] / "brain-bar" / "bundle" / "com.brainlayer.brainbar-daemon.plist"
    content = plist.read_text()

    assert "<string>com.brainlayer.brainbar-daemon</string>" in content
    assert "<string>/Applications/BrainBar.app/Contents/MacOS/BrainBarDaemon</string>" in content
    assert "<key>ProcessType</key>" in content
    assert "<string>Interactive</string>" in content


def test_brainbar_package_declares_separate_ui_and_daemon_products() -> None:
    package = Path(__file__).resolve().parents[1] / "brain-bar" / "Package.swift"
    content = package.read_text()

    assert '.executable(name: "BrainBar", targets: ["BrainBar"])' in content
    assert '.executable(name: "BrainBarDaemon", targets: ["BrainBarDaemon"])' in content
    assert 'path: "Sources/BrainBarDaemon"' in content
