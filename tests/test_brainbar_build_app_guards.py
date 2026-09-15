"""Tests for BrainBar canonical build/install guards."""

from __future__ import annotations

import hashlib
import os
import plistlib
import shutil
import subprocess
from pathlib import Path

import pytest


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
    # Default every test to "Homebrew does not manage BrainBar here" so the
    # brew-managed refusal guard cannot depend on the developer's real machine.
    env["BRAINBAR_BREW_BIN"] = str(repo / "no-such-brew")
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
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIconName</key>
    <string>AppIcon</string>
</dict>
</plist>
"""
    )
    (bundle_dir / "AppIcon.icns").write_bytes(b"fake icns")
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
package_path=""
previous=""
for argument in "$@"; do
  if [[ "$previous" == "--package-path" ]]; then
    package_path="$argument"
    break
  fi
  previous="$argument"
done
if [[ -n "${BRAINBAR_FAKE_SWIFT_HEAD_LOG:-}" && -n "$package_path" ]]; then
  git -C "$package_path" rev-parse HEAD >> "$BRAINBAR_FAKE_SWIFT_HEAD_LOG"
fi
if [[ "${BRAINBAR_FAKE_SWIFT_FAIL:-0}" == "1" ]]; then
  exit 42
fi
if [[ "$*" == *"--show-bin-path"* ]]; then
  printf '%s\n' "$BRAINBAR_FAKE_BIN_DIR"
fi
exit 0
"""
    )
    (tool_dir / "codesign").write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${BRAINBAR_FAKE_CODESIGN_LOG:-/dev/null}"
if [[ "${BRAINBAR_FAKE_CODESIGN_FAIL:-0}" == "1" ]]; then
  exit 42
fi
if [[ "$*" == *"-dv"* ]]; then
  printf 'Authority=%s\n' "${BRAINBAR_CODESIGN_IDENTITY:-Developer ID Application: Etan Heyman (PPN23G925Y)}"
fi
exit 0
"""
    )
    (tool_dir / "xcrun").write_text(
        """#!/usr/bin/env bash
tool="${1:-}"
shift || true
case "$tool" in
  notarytool)
    printf '%s\n' "$*" >> "${BRAINBAR_FAKE_NOTARYTOOL_LOG:-/dev/null}"
    if [[ "${BRAINBAR_FAKE_NOTARYTOOL_FAIL:-0}" == "1" ]]; then
      exit 42
    fi
    ;;
  stapler)
    printf '%s\n' "$*" >> "${BRAINBAR_FAKE_STAPLER_LOG:-/dev/null}"
    ;;
esac
exit 0
"""
    )
    (tool_dir / "spctl").write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${BRAINBAR_FAKE_SPCTL_LOG:-/dev/null}"
if [[ -n "${BRAINBAR_FAKE_RESIDENT_APP:-}" && "$*" == *"${BRAINBAR_FAKE_RESIDENT_APP}"* && "${BRAINBAR_FAKE_RESIDENT_SPCTL_FAIL:-0}" == "1" ]]; then
  exit 1
fi
if [[ "${BRAINBAR_FAKE_SPCTL_FAIL:-0}" == "1" ]]; then
  exit 1
fi
exit 0
"""
    )
    (tool_dir / "ditto").write_text(
        """#!/usr/bin/env bash
out="${@: -1}"
mkdir -p "$(dirname "$out")"
: > "$out"
exit 0
"""
    )
    (tool_dir / "plistbuddy").write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${BRAINBAR_FAKE_PLISTBUDDY_LOG:-/dev/null}"
python3 - "$2" "$3" <<'PY'
import plistlib
import shlex
import sys
from pathlib import Path

command = shlex.split(sys.argv[1])
plist_path = Path(sys.argv[2])
try:
    document = plistlib.loads(plist_path.read_bytes())
except (FileNotFoundError, plistlib.InvalidFileException):
    document = {}

action = command[0]
keys = command[1].lstrip(":").split(":")
parent = document
for key in keys[:-1]:
    if action == "Add":
        parent = parent.setdefault(key, {})
    elif not isinstance(parent, dict) or key not in parent:
        raise SystemExit(1)
    else:
        parent = parent[key]
leaf = keys[-1]

if action == "Print":
    if leaf not in parent:
        if keys == ["CFBundleShortVersionString"]:
            print("1.4.0")
            raise SystemExit(0)
        raise SystemExit(1)
    value = parent[leaf]
    if isinstance(value, bool):
        print(str(value).lower())
    else:
        print(value)
    raise SystemExit(0)

if action == "Delete":
    if leaf not in parent:
        raise SystemExit(1)
    del parent[leaf]
elif action == "Set":
    parent[leaf] = command[2]
elif action == "Add":
    value_type = command[2]
    if value_type == "string":
        value = command[3]
    elif value_type == "bool":
        value = command[3].lower() == "true"
    elif value_type == "dict":
        value = {}
    else:
        raise SystemExit(f"unsupported fake PlistBuddy type: {value_type}")
    parent[leaf] = value
else:
    raise SystemExit(f"unsupported fake PlistBuddy action: {action}")

plist_path.parent.mkdir(parents=True, exist_ok=True)
plist_path.write_bytes(plistlib.dumps(document))
PY
"""
    )
    (tool_dir / "launchctl").write_text(
        """#!/usr/bin/env bash
if [[ "$1" == "kickstart" && -n "${BRAINBAR_SOCKET_PATH:-}" && ! -S "$BRAINBAR_SOCKET_PATH" ]]; then
  ready_file="${BRAINBAR_SOCKET_PATH}.ready"
  rm -f "$ready_file"
  python3 - "$BRAINBAR_SOCKET_PATH" "$ready_file" <<'PY' >/dev/null 2>&1 &
import os
import socket
import sys

path = sys.argv[1]
ready_file = sys.argv[2]
try:
    os.unlink(path)
except FileNotFoundError:
    pass
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(path)
server.listen(1)
with open(ready_file, "w"):
    pass
conn, _ = server.accept()
conn.close()
server.close()
PY
  for _ in {1..50}; do
    [[ -S "$BRAINBAR_SOCKET_PATH" ]] && break
    sleep 0.02
  done
  rm -f "$ready_file"
  disown
fi
exit 0
"""
    )
    (tool_dir / "lsregister").write_text("#!/usr/bin/env bash\nexit 0\n")
    (tool_dir / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n")
    (tool_dir / "killall").write_text("#!/usr/bin/env bash\nexit 0\n")
    for tool in (
        "swift",
        "codesign",
        "xcrun",
        "spctl",
        "ditto",
        "plistbuddy",
        "launchctl",
        "lsregister",
        "pgrep",
        "killall",
    ):
        os.chmod(tool_dir / tool, 0o755)
    return tool_dir, bin_dir


def _install_fake_venv_python(repo: Path, *, import_brainlayer_succeeds: bool) -> Path:
    python_path = repo / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    import_exit = "0" if import_brainlayer_succeeds else "1"
    python_path.write_text(
        f"""#!/usr/bin/env bash
if [[ "$*" == *"import brainlayer"* ]]; then
  exit {import_exit}
fi
exit 0
"""
    )
    os.chmod(python_path, 0o755)
    return python_path


def _fake_build_env(tmp_path: Path, tool_dir: Path, bin_dir: Path) -> dict[str, str]:
    return {
        "BRAINBAR_CODESIGN_IDENTITY": "Test Identity",
        "BRAINBAR_FAKE_BIN_DIR": str(bin_dir),
        "BRAINBAR_LSREGISTER": str(tool_dir / "lsregister"),
        "BRAINBAR_PLIST_BUDDY": str(tool_dir / "plistbuddy"),
        "BRAINBAR_SOCKET_PATH": f"/tmp/brainbar-test-{os.getpid()}-{tmp_path.name}.sock",
        "BRAINBAR_SOCKET_WAIT_ATTEMPTS": "1",
        "PATH": f"{tool_dir}{os.pathsep}{os.environ['PATH']}",
    }


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


def test_canonical_build_aborts_when_brainlayer_package_is_not_importable(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    _prepare_bundle_inputs(repo)
    _install_fake_venv_python(repo, import_brainlayer_succeeds=False)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        dry_run=False,
        extra_args=["--force-dirty"],
        extra_env=_fake_build_env(tmp_path, tool_dir, bin_dir),
    )

    assert result.returncode != 0
    assert "[build-app] ERROR: brainlayer package not installed" in result.stderr
    assert "BrainBar and launchd services no longer use PYTHONPATH for imports." in result.stderr
    assert f"{repo}/.venv/bin/python -m pip install -e ." in result.stderr
    assert not (home / "Library" / "LaunchAgents" / "com.brainlayer.brainbar.plist").exists()


def test_canonical_build_allows_launchagent_install_when_brainlayer_package_is_importable(
    tmp_path: Path,
) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    _prepare_bundle_inputs(repo)
    _install_fake_venv_python(repo, import_brainlayer_succeeds=True)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        dry_run=False,
        extra_args=["--force-dirty"],
        extra_env=_fake_build_env(tmp_path, tool_dir, bin_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "[build-app] brainlayer package is installed" in result.stdout


def test_build_app_embeds_launchagent_templates_in_app_resources(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    _prepare_bundle_inputs(repo)
    _install_fake_venv_python(repo, import_brainlayer_succeeds=True)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        dry_run=False,
        extra_args=["--force-dirty"],
        extra_env=_fake_build_env(tmp_path, tool_dir, bin_dir),
    )

    launchagents_dir = home / "Applications" / "BrainBar.app" / "Contents" / "Resources" / "LaunchAgents"
    resources_dir = home / "Applications" / "BrainBar.app" / "Contents" / "Resources"
    info_plist = home / "Applications" / "BrainBar.app" / "Contents" / "Info.plist"
    assert result.returncode == 0, result.stderr
    assert (resources_dir / "AppIcon.icns").read_bytes() == b"fake icns"
    plist_data = plistlib.loads(info_plist.read_bytes())
    assert plist_data["CFBundleIconFile"] == "AppIcon"
    assert plist_data["CFBundleIconName"] == "AppIcon"
    assert (launchagents_dir / "com.brainlayer.brainbar.plist").is_file()
    assert (launchagents_dir / "com.brainlayer.brainbar-daemon.plist").is_file()


def test_build_app_rejects_invalid_release_version(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-dev-worktree")
    home = tmp_path / "home"
    home.mkdir()
    _prepare_bundle_inputs(repo)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)

    result = _run_build_script(
        repo,
        script,
        canonical_root=tmp_path / "canonical",
        home=home,
        dry_run=False,
        extra_args=["--force-worktree-build"],
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_RELEASE_VERSION": "1.2.3-beta1",
        },
    )

    assert result.returncode != 0
    assert "release version must match X.Y.Z or interim X.Y.Z.N" in result.stderr


def test_build_app_accepts_truthful_four_part_interim_release_version(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-dev-worktree")
    home = tmp_path / "home"
    home.mkdir()
    _prepare_bundle_inputs(repo)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)

    result = _run_build_script(
        repo,
        script,
        canonical_root=tmp_path / "canonical",
        home=home,
        dry_run=False,
        extra_args=["--force-worktree-build"],
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_RELEASE_VERSION": "1.5.2.1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "ReleaseVersion=1.5.2.1" in result.stdout
    assert "BundleShortVersion=1.5.2" in result.stdout
    assert "BundleBuildVersion=" in result.stdout
    assert "BrainLayerReleaseVersion=1.5.2.1" in result.stdout


@pytest.mark.parametrize("tag_name", ["ci-smoke", "1.2.3"])
def test_build_app_ignores_non_release_exact_tags_and_uses_plist_version(tmp_path: Path, tag_name: str) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-dev-worktree")
    home = tmp_path / "home"
    home.mkdir()
    _prepare_bundle_inputs(repo)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)
    _git(repo, "tag", tag_name)

    result = _run_build_script(
        repo,
        script,
        canonical_root=tmp_path / "canonical",
        home=home,
        dry_run=False,
        extra_args=["--force-worktree-build"],
        extra_env=_fake_build_env(tmp_path, tool_dir, bin_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "ReleaseVersion=1.4.0" in result.stdout


def test_build_app_script_is_notarization_ready_for_developer_id() -> None:
    script = (Path(__file__).resolve().parents[1] / "brain-bar" / "build-app.sh").read_text()

    assert "Developer ID Application: Etan Heyman (PPN23G925Y)" in script
    assert "Apple Development: Etan Heyman" not in script
    assert "--options runtime" in script
    assert "--timestamp=none" not in script
    assert "notarytool" in script
    assert "submit" in script
    assert "--wait" in script
    assert "stapler" in script
    assert "staple" in script
    assert "mktemp -t brainbar-notary" not in script
    assert "brainbar-notary.XXXXXX" in script


def test_build_app_runs_hardened_codesign_and_notarization_when_profile_is_available(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-dev-worktree")
    home = tmp_path / "home"
    home.mkdir()
    _prepare_bundle_inputs(repo)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)
    codesign_log = tmp_path / "codesign.log"
    notarytool_log = tmp_path / "notarytool.log"
    stapler_log = tmp_path / "stapler.log"
    spctl_log = tmp_path / "spctl.log"

    result = _run_build_script(
        repo,
        script,
        canonical_root=tmp_path / "canonical",
        home=home,
        dry_run=False,
        extra_args=["--force-worktree-build"],
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_FAKE_CODESIGN_LOG": str(codesign_log),
            "BRAINBAR_FAKE_NOTARYTOOL_LOG": str(notarytool_log),
            "BRAINBAR_FAKE_STAPLER_LOG": str(stapler_log),
            "BRAINBAR_FAKE_SPCTL_LOG": str(spctl_log),
            "BRAINBAR_NOTARY_PROFILE": "brainbar-notary-test",
            "BRAINBAR_CODESIGN_IDENTITY": "Developer ID Application: Etan Heyman (PPN23G925Y)",
        },
    )

    assert result.returncode == 0, result.stderr
    codesign_calls = codesign_log.read_text(encoding="utf-8")
    assert "--sign Developer ID Application: Etan Heyman (PPN23G925Y)" in codesign_calls
    assert "--options runtime" in codesign_calls
    assert "--timestamp " in codesign_calls or codesign_calls.rstrip().endswith("--timestamp")
    assert "--timestamp=none" not in codesign_calls
    assert "--verify --deep --strict --verbose=4" in codesign_calls
    notarytool_call = notarytool_log.read_text(encoding="utf-8")
    notarytool_parts = notarytool_call.split()
    assert notarytool_parts[0] == "submit"
    assert notarytool_parts[1].endswith(".zip")
    assert not notarytool_parts[1].endswith(".app")
    assert "--keychain-profile brainbar-notary-test" in notarytool_call
    assert "--wait" in notarytool_call
    assert stapler_log.read_text(encoding="utf-8").startswith("staple ")
    assert "-a -vvv" in spctl_log.read_text(encoding="utf-8")


def test_build_app_refuses_to_clobber_notarized_resident_with_unnotarized_rebuild(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    resident_apps = tmp_path / "Applications"
    resident_app = resident_apps / "BrainBar.app"
    resident_app.mkdir(parents=True)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)
    spctl_log = tmp_path / "spctl.log"

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_APP_DIR": f"{resident_app}/",
            "BRAINBAR_PROTECTED_APPLICATIONS_DIR": str(resident_apps),
            "BRAINBAR_BREW_BIN": str(_brew_stub(tmp_path, manages=False)),
            "BRAINBAR_FAKE_SPCTL_LOG": str(spctl_log),
        },
    )

    assert result.returncode == 1
    assert f"Refusing to replace notarized {resident_app} with an unnotarized local rebuild" in result.stderr
    assert "BRAINBAR_NOTARY_PROFILE=notary-layers" in result.stderr
    assert "bash scripts/brainlayer-update-brainbar.sh" in result.stderr
    spctl_calls = spctl_log.read_text(encoding="utf-8")
    assert "--assess --type execute" in spctl_calls
    assert str(resident_app) in spctl_calls


def test_build_app_refuses_equivalent_protected_resident_path(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    resident_apps = tmp_path / "Applications"
    resident_app = resident_apps / "BrainBar.app"
    resident_app.mkdir(parents=True)
    linked_apps = tmp_path / "LinkedApplications"
    linked_apps.symlink_to(resident_apps, target_is_directory=True)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)
    spctl_log = tmp_path / "spctl.log"

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_APP_DIR": str(linked_apps / "BrainBar.app"),
            "BRAINBAR_PROTECTED_APPLICATIONS_DIR": str(resident_apps),
            "BRAINBAR_BREW_BIN": str(_brew_stub(tmp_path, manages=False)),
            "BRAINBAR_FAKE_SPCTL_LOG": str(spctl_log),
        },
    )

    assert result.returncode == 1
    assert "Refusing to replace notarized" in result.stderr
    assert str(resident_app) in spctl_log.read_text(encoding="utf-8")


def test_build_app_keeps_notarized_resident_when_notarization_fails(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    resident_apps = tmp_path / "Applications"
    resident_app = resident_apps / "BrainBar.app"
    marker = resident_app / "Contents" / "resident-marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")
    _prepare_bundle_inputs(repo)
    _install_fake_venv_python(repo, import_brainlayer_succeeds=True)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        dry_run=False,
        extra_args=["--force-dirty"],
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_APP_DIR": str(resident_app),
            "BRAINBAR_PROTECTED_APPLICATIONS_DIR": str(resident_apps),
            "BRAINBAR_BREW_BIN": str(_brew_stub(tmp_path, manages=False)),
            "BRAINBAR_NOTARY_PROFILE": "notary-layers",
            "BRAINBAR_FAKE_NOTARYTOOL_FAIL": "1",
        },
    )

    assert result.returncode != 0
    assert marker.read_text(encoding="utf-8") == "keep"
    assert "Protected resident rebuild will stage" in result.stdout


def test_build_app_allows_notarized_resident_rebuild_when_notary_profile_is_set(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    resident_apps = tmp_path / "Applications"
    resident_app = resident_apps / "BrainBar.app"
    resident_app.mkdir(parents=True)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_APP_DIR": f"{resident_app}/",
            "BRAINBAR_PROTECTED_APPLICATIONS_DIR": str(resident_apps),
            "BRAINBAR_BREW_BIN": str(_brew_stub(tmp_path, manages=False)),
            "BRAINBAR_NOTARY_PROFILE": "notary-layers",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "[build-app] Dry run OK" in result.stdout


def test_build_app_allows_dev_app_dir_when_resident_is_notarized(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    resident_apps = tmp_path / "Applications"
    resident_app = resident_apps / "BrainBar.app"
    resident_app.mkdir(parents=True)
    dev_app = tmp_path / "dist" / "BrainBar.app"
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_APP_DIR": str(dev_app),
            "BRAINBAR_PROTECTED_APPLICATIONS_DIR": str(resident_apps),
        },
    )

    assert result.returncode == 0, result.stderr
    assert str(dev_app) in result.stdout


def test_build_app_submits_before_requiring_post_notary_spctl_acceptance(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-dev-worktree")
    home = tmp_path / "home"
    home.mkdir()
    _prepare_bundle_inputs(repo)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)
    notarytool_log = tmp_path / "notarytool.log"
    stapler_log = tmp_path / "stapler.log"
    spctl_log = tmp_path / "spctl.log"

    result = _run_build_script(
        repo,
        script,
        canonical_root=tmp_path / "canonical",
        home=home,
        dry_run=False,
        extra_args=["--force-worktree-build"],
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_FAKE_NOTARYTOOL_LOG": str(notarytool_log),
            "BRAINBAR_FAKE_STAPLER_LOG": str(stapler_log),
            "BRAINBAR_FAKE_SPCTL_LOG": str(spctl_log),
            "BRAINBAR_FAKE_SPCTL_FAIL": "1",
            "BRAINBAR_NOTARY_PROFILE": "brainbar-notary-test",
            "BRAINBAR_CODESIGN_IDENTITY": "Developer ID Application: Etan Heyman (PPN23G925Y)",
        },
    )

    assert result.returncode != 0
    assert notarytool_log.read_text(encoding="utf-8").startswith("submit ")
    assert stapler_log.read_text(encoding="utf-8").startswith("staple ")
    assert spctl_log.read_text(encoding="utf-8").count("-a -vvv") >= 2
    assert "spctl assessment failed after notarization" in result.stderr


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
            "BRAINBAR_LSREGISTER": str(tool_dir / "lsregister"),
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


def test_canonical_dev_cleanup_preserves_branch_bundle_for_detached_worktree_head(
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
    _git(repo, "branch", "feat/detached-active")
    worktree = tmp_path / "detached-active-worktree"
    _git(repo, "worktree", "add", "--detach", str(worktree), "HEAD")
    bundle = _create_fake_bundle(apps_dir, "BrainBar-DEV-feat-detached-active.app", main_sha)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
    )

    assert result.returncode == 0, result.stderr
    assert "Keeping DEV bundle: BrainBar-DEV-feat-detached-active.app" in result.stdout
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


def test_canonical_dev_cleanup_invalid_stale_days_falls_back_to_default(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    apps_dir = home / "Applications"
    apps_dir.mkdir(parents=True)
    _git(repo, "branch", "feat/old-dev")
    bundle = _create_fake_bundle(apps_dir, "BrainBar-DEV-feat-old-dev.app")
    old_mtime = 0
    os.utime(bundle, (old_mtime, old_mtime))

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={"BRAINBAR_DEV_STALE_DAYS": "not-a-number"},
    )

    assert result.returncode == 0, result.stderr
    assert "invalid BRAINBAR_DEV_STALE_DAYS='not-a-number', using 14" in result.stderr
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


@pytest.mark.parametrize(
    "requested_path",
    [
        "home",
        "home-trailing",
        "home-relative",
        "home-symlink",
        "home-case",
        "protected",
        "home-nested",
        "protected-nested",
    ],
)
def test_dev_build_refuses_production_app_path_before_rebuild(tmp_path: Path, requested_path: str) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-worktree", branch="feat/ui-guards")
    home = tmp_path / "home"
    _prepare_bundle_inputs(repo)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)
    production_app = home / "Applications" / "BrainBar.app"
    protected_app = tmp_path / "protected" / "BrainBar.app"
    target_app = protected_app if requested_path in {"protected", "protected-nested"} else production_app
    daemon = target_app / "Contents" / "MacOS" / "BrainBarDaemon"
    daemon.parent.mkdir(parents=True)
    daemon.write_text("production daemon", encoding="utf-8")
    plist_path = target_app / "Contents" / "Info.plist"
    plist_path.write_bytes(plistlib.dumps({"CFBundleIdentifier": "com.brainlayer.brainbar"}))

    if requested_path == "home-trailing":
        requested = f"{production_app}/"
    elif requested_path == "home-relative":
        requested = str(home / "Applications" / ".." / "Applications" / "BrainBar.app")
    elif requested_path == "home-symlink":
        linked_apps = tmp_path / "linked-apps"
        linked_apps.symlink_to(home / "Applications", target_is_directory=True)
        requested = str(linked_apps / "BrainBar.app")
    elif requested_path == "home-case":
        requested = str(home / "Applications" / "Brainbar.app")
    elif requested_path in {"home-nested", "protected-nested"}:
        requested = str(target_app / "Contents" / "Resources" / "BrainBar-DEV-nested.app")
    else:
        requested = str(target_app)
    before_files = sorted(path.relative_to(target_app) for path in target_app.rglob("*"))

    result = _run_build_script(
        repo,
        script,
        canonical_root=tmp_path / "brainlayer-canonical",
        home=home,
        dry_run=False,
        extra_args=["--force-worktree-build", "--force-dirty"],
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_DEV_APP_DIR": requested,
            "BRAINBAR_PROTECTED_APPLICATIONS_DIR": str(tmp_path / "protected"),
        },
    )

    assert result.returncode != 0
    assert "refusing DEV bundle" in result.stderr
    assert plistlib.loads(plist_path.read_bytes())["CFBundleIdentifier"] == "com.brainlayer.brainbar"
    assert daemon.read_text(encoding="utf-8") == "production daemon"
    assert sorted(path.relative_to(target_app) for path in target_app.rglob("*")) == before_files


@pytest.mark.parametrize("production_payload", ["bundle-id", "daemon"])
def test_dev_build_refuses_production_identity_at_dev_named_path(tmp_path: Path, production_payload: str) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-worktree", branch="feat/ui-guards")
    home = tmp_path / "home"
    _prepare_bundle_inputs(repo)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)
    app = home / "Applications" / "BrainBar-DEV-disguised.app"
    plist_path = app / "Contents" / "Info.plist"
    plist_path.parent.mkdir(parents=True)
    plist_path.write_bytes(
        plistlib.dumps(
            {
                "CFBundleIdentifier": (
                    "com.brainlayer.brainbar"
                    if production_payload == "bundle-id"
                    else "com.brainlayer.brainbar.dev.disguised"
                )
            }
        )
    )
    daemon = app / "Contents" / "MacOS" / "BrainBarDaemon"
    if production_payload == "daemon":
        daemon.parent.mkdir(parents=True)
        daemon.write_text("production daemon", encoding="utf-8")
    marker = app / "Contents" / "production-marker"
    marker.write_text("keep", encoding="utf-8")
    original_plist = plist_path.read_bytes()

    result = _run_build_script(
        repo,
        script,
        canonical_root=tmp_path / "brainlayer-canonical",
        home=home,
        dry_run=False,
        extra_args=["--force-worktree-build", "--force-dirty"],
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_DEV_APP_DIR": str(app),
        },
    )

    assert result.returncode != 0
    assert "refusing DEV bundle over an existing" in result.stderr
    assert plist_path.read_bytes() == original_plist
    assert marker.read_text(encoding="utf-8") == "keep"
    if production_payload == "daemon":
        assert daemon.read_text(encoding="utf-8") == "production daemon"


def test_dev_build_refuses_path_reserved_by_production_override(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-worktree", branch="feat/ui-guards")
    home = tmp_path / "home"
    reserved_app = home / "Applications" / "BrainBar-DEV-reserved.app"

    result = _run_build_script(
        repo,
        script,
        canonical_root=tmp_path / "brainlayer-canonical",
        home=home,
        extra_args=["--force-worktree-build"],
        extra_env={
            "BRAINBAR_DEV_APP_DIR": str(reserved_app),
            "BRAINBAR_APP_DIR": str(reserved_app),
        },
    )

    assert result.returncode != 0
    assert "refusing DEV bundle at production app path" in result.stderr


def test_dev_build_fails_closed_when_existing_bundle_identity_cannot_be_read(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-worktree", branch="feat/ui-guards")
    home = tmp_path / "home"
    app = home / "Applications" / "BrainBar-DEV-uninspectable.app"
    plist_path = app / "Contents" / "Info.plist"
    plist_path.parent.mkdir(parents=True)
    plist_path.write_bytes(plistlib.dumps({"CFBundleIdentifier": "com.brainlayer.brainbar"}))

    result = _run_build_script(
        repo,
        script,
        canonical_root=tmp_path / "brainlayer-canonical",
        home=home,
        extra_args=["--force-worktree-build"],
        extra_env={
            "BRAINBAR_DEV_APP_DIR": str(app),
            "BRAINBAR_PLIST_BUDDY": str(tmp_path / "missing-plistbuddy"),
        },
    )

    assert result.returncode != 0
    assert "existing bundle identifier cannot be inspected" in result.stderr
    assert plistlib.loads(plist_path.read_bytes())["CFBundleIdentifier"] == "com.brainlayer.brainbar"


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


def test_canonical_bundle_identifier_remains_production_identity() -> None:
    info_plist = Path(__file__).resolve().parents[1] / "brain-bar" / "bundle" / "Info.plist"

    plist_data = plistlib.loads(info_plist.read_bytes())

    assert plist_data["CFBundleIdentifier"] == "com.brainlayer.brainbar"
    assert "BrainBarDevPreview" not in plist_data


def test_dev_build_stamps_unique_preview_identity_without_daemon_payload(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-worktree", branch="feat/UI_Guards")
    home = tmp_path / "home"
    home.mkdir()
    _prepare_bundle_inputs(repo)
    repo_head = _git_stdout(repo, "rev-parse", "HEAD")
    bogus_source_commit = "f" * 40
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)
    plistbuddy_log = tmp_path / "plistbuddy.log"
    preview_app = home / "Applications" / "BrainBar DEV" / "BrainBar DEV · feat-UI_Guards.app"

    result = _run_build_script(
        repo,
        script,
        canonical_root=tmp_path / "canonical",
        home=home,
        dry_run=False,
        extra_args=["--force-worktree-build"],
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_DEV_APP_DIR": str(preview_app),
            "BRAINBAR_FAKE_PLISTBUDDY_LOG": str(plistbuddy_log),
            "BRAINBAR_DEV_SOURCE_COMMIT": bogus_source_commit,
        },
    )

    assert result.returncode == 0, result.stderr
    assert f"Done: {preview_app}" in result.stdout
    plist_calls = plistbuddy_log.read_text(encoding="utf-8")
    assert 'CFBundleIdentifier string "com.brainlayer.brainbar.dev.feat-ui-guards-' in plist_calls
    assert "Add :BrainBarDevPreview bool true" in plist_calls
    assert 'BrainBarDevBranch string "feat/UI_Guards"' in plist_calls
    git_commit_calls = [line for line in plist_calls.splitlines() if ":GitCommit" in line]
    assert any(repo_head in line for line in git_commit_calls)
    assert not any(bogus_source_commit in line for line in git_commit_calls)
    assert not (preview_app / "Contents" / "MacOS" / "BrainBarDaemon").exists()
    assert not (preview_app / "Contents" / "Resources" / "LaunchAgents").exists()


def test_dev_preview_wrapper_builds_verifies_and_cleans_only_dev_bundles(tmp_path: Path) -> None:
    repo, build_script = _prepare_build_repo(
        tmp_path,
        "brainlayer-worktree",
        branch="feat/a",
    )
    home = tmp_path / "home"
    home.mkdir()
    _prepare_bundle_inputs(repo)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)

    source_wrapper = Path(__file__).resolve().parents[1] / "brain-bar" / "Scripts" / "dev-preview.sh"
    wrapper = repo / "brain-bar" / "Scripts" / "dev-preview.sh"
    wrapper.parent.mkdir(parents=True)
    shutil.copy2(source_wrapper, wrapper)

    open_log = tmp_path / "open.log"
    open_stub = tmp_path / "open"
    open_stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$BRAINBAR_TEST_OPEN_LOG"\n')
    open_stub.chmod(0o755)
    preview_root = home / "Applications" / "BrainBar DEV"
    slash_hash = hashlib.sha256(b"feat/a").hexdigest()[:8]
    dash_hash = hashlib.sha256(b"feat-a").hexdigest()[:8]
    slash_app = preview_root / f"BrainBar DEV · feat-a-{slash_hash}.app"
    dash_app = preview_root / f"BrainBar DEV · feat-a-{dash_hash}.app"
    env = {
        **_clean_git_env(),
        **_fake_build_env(tmp_path, tool_dir, bin_dir),
        "HOME": str(home),
        "BRAINBAR_CANONICAL_REPO_ROOT": str(tmp_path / "canonical"),
        "BRAINBAR_BREW_BIN": str(repo / "no-such-brew"),
        "BRAINBAR_DEV_PREVIEW_ROOT": str(preview_root),
        "BRAINBAR_DEV_OPEN_BIN": str(open_stub),
        "BRAINBAR_TEST_OPEN_LOG": str(open_log),
        "BRAINBAR_DEV_TRASH_DIR": str(tmp_path / "trash"),
    }

    built = subprocess.run(
        ["/bin/bash", str(wrapper), "feat/a"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert built.returncode == 0, built.stdout + built.stderr
    assert "PREVIEW\tfeat/a\t" in built.stdout
    assert open_log.read_text().strip() == f"-n {slash_app}"

    marker = slash_app / "Contents" / "last-good-build"
    marker.write_text("keep", encoding="utf-8")
    old_binary = slash_app / "Contents" / "MacOS" / "BrainBar"
    old_binary.write_text("known working preview", encoding="utf-8")
    failed_env = {**env, "BRAINBAR_FAKE_CODESIGN_FAIL": "1"}
    failed_rebuild = subprocess.run(
        ["/bin/bash", str(wrapper), "feat/a"],
        cwd=repo,
        env=failed_env,
        capture_output=True,
        text=True,
    )
    assert failed_rebuild.returncode != 0
    assert marker.read_text(encoding="utf-8") == "keep"
    assert old_binary.read_text(encoding="utf-8") == "known working preview"
    assert open_log.read_text().count("\n") == 1

    duplicate_app = preview_root / "BrainBar DEV old-generation.app"
    shutil.copytree(slash_app, duplicate_app)
    duplicate = subprocess.run(
        ["/bin/bash", str(wrapper), "feat/a"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert duplicate.returncode != 0
    assert "appears in 2 DEV bundles" in duplicate.stderr
    assert open_log.read_text().count("\n") == 1
    shutil.rmtree(duplicate_app)

    _git(repo, "checkout", "-b", "feat-a")
    collision_build = subprocess.run(
        ["/bin/bash", str(wrapper), str(repo)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert collision_build.returncode == 0, collision_build.stdout + collision_build.stderr
    assert slash_app.is_dir()
    assert dash_app.is_dir()
    assert slash_app != dash_app

    plist_path = dash_app / "Contents" / "Info.plist"
    plist_data = plistlib.loads(plist_path.read_bytes())
    plist_data["CFBundleIdentifier"] = "com.brainlayer.brainbar"
    plist_path.write_bytes(plistlib.dumps(plist_data))
    build_script.write_text("#!/usr/bin/env bash\n# BrainBarDevHarnessCommit\nexit 0\n")
    build_script.chmod(0o755)
    refused = subprocess.run(
        ["/bin/bash", str(wrapper), str(repo)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert refused.returncode != 0
    assert "unsafe or stale DEV bundle stamp" in refused.stderr
    assert open_log.read_text().count("\n") == 2

    legacy = home / "Applications" / "BrainBar-DEV-legacy.app"
    production = home / "Applications" / "BrainBar.app"
    backup = home / "Applications" / "BrainBar.app.bak-safe"
    shutil.copytree(slash_app, legacy)
    legacy_plist = legacy / "Contents" / "Info.plist"
    legacy_data = plistlib.loads(legacy_plist.read_bytes())
    legacy_data["CFBundleIdentifier"] = "com.brainlayer.brainbar"
    legacy_data.pop("BrainBarDevPreview", None)
    legacy_plist.write_bytes(plistlib.dumps(legacy_data))
    for app in (production, backup):
        (app / "Contents" / "MacOS").mkdir(parents=True)
        (app / "Contents" / "MacOS" / "BrainBarDaemon").write_text("production daemon\n")
        (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": "com.brainlayer.brainbar"}))
    identity_named = preview_root / "preview-from-an-older-naming-generation.app"
    shutil.copytree(slash_app, identity_named)
    unrecognized = preview_root / "Unrelated.app"
    unrecognized.mkdir(parents=True)
    production_copy = preview_root / "Production-copy.app"
    shutil.copytree(production, production_copy)

    cleaned = subprocess.run(
        ["/bin/bash", str(wrapper), "--clean"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert cleaned.returncode == 0, cleaned.stdout + cleaned.stderr
    assert not slash_app.exists()
    assert not dash_app.exists()
    assert not legacy.exists()
    assert not identity_named.exists()
    assert production.is_dir()
    assert backup.is_dir()
    assert unrecognized.is_dir()
    assert production_copy.is_dir()
    trash_dir = tmp_path / "trash"
    for app in (slash_app, dash_app, legacy, identity_named):
        assert (trash_dir / app.name).is_dir()


def test_dev_preview_wrapper_builds_from_target_head_not_harness_head(tmp_path: Path) -> None:
    current_build_script = Path(__file__).resolve().parents[1] / "brain-bar" / "build-app.sh"
    current_wrapper = Path(__file__).resolve().parents[1] / "brain-bar" / "Scripts" / "dev-preview.sh"
    repo, build_script = _prepare_build_repo(tmp_path, "brainlayer-harness", branch="harness")
    _prepare_bundle_inputs(repo)
    build_script.write_text(
        current_build_script.read_text().replace("BrainBarDevHarnessCommit", "OldPreviewHarnessCommit")
    )
    _git(repo, "add", "brain-bar/build-app.sh")
    _commit(repo, "test: old target build script")

    _git(repo, "checkout", "-b", "target")
    _write_tracked_file(repo, "brain-bar/Sources/feature-source.txt", "target-only source\n")
    _commit(repo, "feat: target source")
    target_sha = _git_stdout(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "harness")
    build_script.write_text(current_build_script.read_text())
    wrapper = repo / "brain-bar" / "Scripts" / "dev-preview.sh"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(current_wrapper.read_text())
    _git(repo, "add", "brain-bar/build-app.sh", "brain-bar/Scripts/dev-preview.sh")
    _commit(repo, "feat: add preview harness")
    harness_sha = _git_stdout(repo, "rev-parse", "HEAD")
    target_worktree = tmp_path / "target-worktree"
    _git(repo, "worktree", "add", str(target_worktree), "target")

    home = tmp_path / "home"
    home.mkdir()
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)
    open_stub = tmp_path / "open"
    open_stub.write_text("#!/usr/bin/env bash\nexit 0\n")
    open_stub.chmod(0o755)
    preview_root = home / "Applications" / "BrainBar DEV"
    swift_head_log = tmp_path / "swift-heads.log"
    env = {
        **_clean_git_env(),
        **_fake_build_env(tmp_path, tool_dir, bin_dir),
        "HOME": str(home),
        "BRAINBAR_CANONICAL_REPO_ROOT": str(tmp_path / "canonical"),
        "BRAINBAR_BREW_BIN": str(repo / "no-such-brew"),
        "BRAINBAR_DEV_PREVIEW_ROOT": str(preview_root),
        "BRAINBAR_DEV_OPEN_BIN": str(open_stub),
        "BRAINBAR_FAKE_SWIFT_HEAD_LOG": str(swift_head_log),
    }

    result = subprocess.run(
        ["/bin/bash", str(wrapper), "target"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    branch_hash = hashlib.sha256(b"target").hexdigest()[:8]
    plist_path = preview_root / f"BrainBar DEV · target-{branch_hash}.app" / "Contents" / "Info.plist"
    plist_data = plistlib.loads(plist_path.read_bytes())
    assert target_sha != harness_sha
    assert plist_data["GitCommit"] == target_sha
    assert plist_data["BrainBarDevHarnessCommit"] == harness_sha
    compiled_heads = swift_head_log.read_text(encoding="utf-8").splitlines()
    assert compiled_heads
    assert set(compiled_heads) == {target_sha}


def test_dev_preview_wrapper_refuses_canonical_root_before_invoking_build(tmp_path: Path) -> None:
    repo, build_script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    source_wrapper = Path(__file__).resolve().parents[1] / "brain-bar" / "Scripts" / "dev-preview.sh"
    wrapper = repo / "brain-bar" / "Scripts" / "dev-preview.sh"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_wrapper, wrapper)
    teardown_log = tmp_path / "production-teardown.log"
    socket_stand_in = tmp_path / "brainbar.sock"
    socket_stand_in.write_text("fleet socket stand-in\n")
    build_script.write_text(
        "#!/usr/bin/env bash\n"
        'printf "killall BrainBar\\nbootout LaunchAgent\\n" >> "$BRAINBAR_TEST_TEARDOWN_LOG"\n'
        'rm -f "$BRAINBAR_SOCKET_PATH"\n'
        "exit 0\n"
    )
    build_script.chmod(0o755)
    symlink_root = tmp_path / "canonical-link"
    symlink_root.symlink_to(repo, target_is_directory=True)
    env = {
        **_clean_git_env(),
        "HOME": str(tmp_path / "home"),
        "BRAINBAR_CANONICAL_REPO_ROOT": str(symlink_root),
        "BRAINBAR_TEST_TEARDOWN_LOG": str(teardown_log),
        "BRAINBAR_SOCKET_PATH": str(socket_stand_in),
    }

    result = subprocess.run(
        ["/bin/bash", str(wrapper), str(symlink_root)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing canonical repo root" in result.stderr
    assert not teardown_log.exists(), "wrapper must refuse before invoking the target build script"
    assert socket_stand_in.exists(), "wrapper refusal must preserve the fleet socket"


def test_build_app_rejects_dev_intent_at_canonical_root_before_teardown(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    home.mkdir()
    _prepare_bundle_inputs(repo)
    tool_dir, bin_dir = _prepare_fake_build_tools(tmp_path)
    teardown_log = tmp_path / "production-teardown.log"
    socket_stand_in = tmp_path / "brainbar.sock"
    socket_stand_in.write_text("fleet socket stand-in\n")
    for tool in ("killall", "launchctl"):
        stub = tool_dir / tool
        stub.write_text(f'#!/usr/bin/env bash\nprintf "{tool} %s\\n" "$*" >> "$BRAINBAR_TEST_TEARDOWN_LOG"\nexit 0\n')
        stub.chmod(0o755)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        dry_run=False,
        extra_args=["--force-worktree-build", "--force-dirty"],
        extra_env={
            **_fake_build_env(tmp_path, tool_dir, bin_dir),
            "BRAINBAR_DEV_APP_DIR": str(tmp_path / "BrainBar DEV.app"),
            "BRAINBAR_TEST_TEARDOWN_LOG": str(teardown_log),
            "BRAINBAR_SOCKET_PATH": str(socket_stand_in),
        },
    )

    assert result.returncode != 0
    assert "refusing DEV preview intent from the canonical repo root" in result.stderr
    assert not teardown_log.exists(), "DEV intent must fail before bootout or killall"
    assert socket_stand_in.exists(), "DEV intent refusal must preserve the fleet socket"


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


# --- drift-proof contract rule 7: stop the drift at its source ------------------------------


def _brew_stub(
    tmp_path: Path,
    *,
    manages: bool,
    log: Path | None = None,
    query_error: bool = False,
) -> Path:
    stub = tmp_path / ("brew-manages" if manages else "brew-clean")
    log_line = f"printf '%s\\n' \"$*\" >> {log}\n" if log else ""
    list_output = "printf 'brainbar\\n'" if manages else "true"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        + log_line
        + f"""if [[ "$*" == "list --cask" ]]; then
  {"exit 42" if query_error else f"{list_output}; exit 0"}
fi
if [[ "$*" == "list --cask brainbar" ]]; then
  {"exit 42" if query_error else f"exit {0 if manages else 1}"}
fi
exit 1
""",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def test_build_app_refuses_to_overwrite_a_brew_managed_resident_app(tmp_path: Path) -> None:
    """VoiceBar's root cause: building in place over a brew-managed app creates silent drift."""
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    resident_apps = tmp_path / "Applications"
    resident_app = resident_apps / "BrainBar.app"
    resident_app.mkdir(parents=True)
    brew_log = tmp_path / "brew.log"

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={
            "BRAINBAR_APP_DIR": str(resident_app),
            "BRAINBAR_PROTECTED_APPLICATIONS_DIR": str(resident_apps),
            "BRAINBAR_BREW_BIN": str(_brew_stub(tmp_path, manages=True, log=brew_log)),
        },
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"refusing to build over the Homebrew-managed app at {resident_app}" in result.stderr
    assert "bash scripts/brainlayer-update-brainbar.sh" in result.stderr
    assert "BRAINBAR_APP_DIR=" in result.stderr
    assert "list --cask" in brew_log.read_text(encoding="utf-8")


def test_build_app_allows_the_resident_path_when_brew_does_not_manage_it(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    resident_apps = tmp_path / "Applications"
    resident_apps.mkdir(parents=True)
    resident_app = resident_apps / "BrainBar.app"

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={
            "BRAINBAR_APP_DIR": str(resident_app),
            "BRAINBAR_PROTECTED_APPLICATIONS_DIR": str(resident_apps),
            "BRAINBAR_BREW_BIN": str(_brew_stub(tmp_path, manages=False)),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dry run OK" in result.stdout
    assert "refusing to build over the Homebrew-managed app" not in result.stderr


def test_build_app_does_not_consult_brew_for_non_resident_paths(tmp_path: Path) -> None:
    """A ~/Applications build is never brew-managed, so it must not be blocked."""
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    resident_apps = tmp_path / "Applications"
    resident_apps.mkdir(parents=True)
    brew_log = tmp_path / "brew.log"

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={
            "BRAINBAR_PROTECTED_APPLICATIONS_DIR": str(resident_apps),
            "BRAINBAR_BREW_BIN": str(_brew_stub(tmp_path, manages=True, log=brew_log)),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert str(home / "Applications" / "BrainBar.app") in result.stdout
    assert not brew_log.exists(), "consulted brew for a path Homebrew can never own"


def test_build_app_fails_closed_when_brew_inventory_query_errors(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    resident_apps = tmp_path / "Applications"
    resident_app = resident_apps / "BrainBar.app"
    resident_app.mkdir(parents=True)

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={
            "BRAINBAR_APP_DIR": str(resident_app),
            "BRAINBAR_PROTECTED_APPLICATIONS_DIR": str(resident_apps),
            "BRAINBAR_BREW_BIN": str(_brew_stub(tmp_path, manages=False, query_error=True)),
        },
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "could not determine whether Homebrew manages" in result.stderr


def test_build_app_fails_closed_for_unusable_explicit_brew_path(tmp_path: Path) -> None:
    repo, script = _prepare_build_repo(tmp_path, "brainlayer-canonical")
    home = tmp_path / "home"
    resident_apps = tmp_path / "Applications"
    resident_app = resident_apps / "BrainBar.app"
    resident_app.mkdir(parents=True)
    missing_brew = tmp_path / "missing-brew"

    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=home,
        extra_env={
            "BRAINBAR_APP_DIR": str(resident_app),
            "BRAINBAR_PROTECTED_APPLICATIONS_DIR": str(resident_apps),
            "BRAINBAR_BREW_BIN": str(missing_brew),
        },
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert str(missing_brew) in result.stderr
    assert "could not determine whether Homebrew manages" in result.stderr
