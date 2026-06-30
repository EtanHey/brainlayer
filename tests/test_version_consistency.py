"""BrainBar release metadata consistency guard."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "brainlayer-version-check.sh"


def _clean_git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _copy_version_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    fixture_root = tmp_path / "brainlayer"
    tap_root = tmp_path / "homebrew-layers"
    package_version = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]

    for rel_path in (
        "pyproject.toml",
        "src/brainlayer/__init__.py",
        "server.json",
        "brain-bar/bundle/Info.plist",
    ):
        source = REPO_ROOT / rel_path
        target = fixture_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    cask_target = tap_root / "Casks" / "brainbar.rb"
    cask_target.parent.mkdir(parents=True, exist_ok=True)
    cask_target.write_text(
        f"""cask "brainbar" do
  version "{package_version}"
  sha256 "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

  url "https://example.invalid/BrainBar.zip"
  name "BrainBar"
  app "BrainBar.app"
end
""",
        encoding="utf-8",
    )
    return fixture_root, tap_root, package_version


def _run(
    fixture_root: Path,
    tap_root: Path,
    *,
    git_tag: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **_clean_git_env(),
        "BRAINLAYER_VERSION_CHECK_REPO_ROOT": str(fixture_root),
        "BRAINLAYER_VERSION_CHECK_TAP_ROOT": str(tap_root),
        **(extra_env or {}),
    }
    if git_tag is not None:
        env["BRAINLAYER_VERSION_CHECK_GIT_TAG"] = git_tag
    return subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env, timeout=30)


def _repo_git_env() -> dict[str, str]:
    git_dir = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--git-dir"],
        text=True,
        env=_clean_git_env(),
    ).strip()
    git_dir_path = Path(git_dir)
    if not git_dir_path.is_absolute():
        git_dir_path = REPO_ROOT / git_dir_path
    return {
        "GIT_DIR": str(git_dir_path),
        "GIT_WORK_TREE": str(REPO_ROOT),
        "GIT_INDEX_FILE": str(git_dir_path / "index"),
        "GIT_PREFIX": "hooks/",
        "GIT_CONFIG_COUNT": "1",
    }


def _init_git_repo(path: Path, tag: str) -> None:
    env = _clean_git_env()
    subprocess.run(["git", "-C", str(path), "init"], check=True, capture_output=True, text=True, env=env)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True, text=True, env=env)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=BrainLayer Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    subprocess.run(["git", "-C", str(path), "tag", tag], check=True, capture_output=True, text=True, env=env)


def test_version_check_passes_when_release_metadata_matches(tmp_path: Path) -> None:
    fixture_root, tap_root, package_version = _copy_version_fixture(tmp_path)

    result = _run(fixture_root, tap_root, git_tag=f"v{package_version}")

    assert result.returncode == 0, result.stderr
    assert f"PASS: BrainLayer/BrainBar {package_version} release metadata is consistent" in result.stdout


def test_version_check_fails_loudly_when_cask_version_drifts(tmp_path: Path) -> None:
    fixture_root, tap_root, package_version = _copy_version_fixture(tmp_path)
    cask = tap_root / "Casks" / "brainbar.rb"
    cask.write_text(cask.read_text(encoding="utf-8").replace(package_version, "0.0.0", 1), encoding="utf-8")

    result = _run(fixture_root, tap_root, git_tag=f"v{package_version}")

    assert result.returncode == 1
    assert f"Casks/brainbar.rb version is '0.0.0', expected '{package_version}'" in result.stderr


def test_version_check_fails_loudly_when_checked_in_info_plist_drifts(tmp_path: Path) -> None:
    fixture_root, tap_root, package_version = _copy_version_fixture(tmp_path)
    plist = fixture_root / "brain-bar" / "bundle" / "Info.plist"
    plist.write_text(
        plist.read_text(encoding="utf-8").replace(
            f"<key>CFBundleShortVersionString</key>\n    <string>{package_version}</string>",
            "<key>CFBundleShortVersionString</key>\n    <string>0.0.0</string>",
        ),
        encoding="utf-8",
    )

    result = _run(fixture_root, tap_root, git_tag=f"v{package_version}")

    assert result.returncode == 1
    assert f"Info.plist CFBundleShortVersionString is '0.0.0', expected '{package_version}'" in result.stderr


def test_version_check_fails_loudly_when_server_manifest_drifts(tmp_path: Path) -> None:
    fixture_root, tap_root, package_version = _copy_version_fixture(tmp_path)
    server_path = fixture_root / "server.json"
    server = json.loads(server_path.read_text(encoding="utf-8"))
    server["packages"][0]["version"] = "0.0.0"
    server_path.write_text(json.dumps(server), encoding="utf-8")

    result = _run(fixture_root, tap_root, git_tag=f"v{package_version}")

    assert result.returncode == 1
    assert f"server.json packages[0].version is '0.0.0', expected '{package_version}'" in result.stderr


def test_version_check_fails_loudly_when_latest_git_tag_drifts(tmp_path: Path) -> None:
    fixture_root, tap_root, _package_version = _copy_version_fixture(tmp_path)

    result = _run(fixture_root, tap_root, git_tag="v0.0.0")

    assert result.returncode == 1
    assert "latest git tag is 'v0.0.0'" in result.stderr


def test_version_check_reports_controlled_error_outside_git_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_root, tap_root, _package_version = _copy_version_fixture(tmp_path)
    repo_git_path = REPO_ROOT / ".git"
    if repo_git_path.is_file():
        git_dir = repo_git_path.read_text(encoding="utf-8").strip().removeprefix("gitdir: ")
    else:
        git_dir = str(repo_git_path)
    if not Path(git_dir).is_absolute():
        git_dir = str(repo_git_path.parent / git_dir)
    monkeypatch.setenv("GIT_DIR", git_dir)
    monkeypatch.setenv("GIT_WORK_TREE", str(REPO_ROOT))

    result = _run(fixture_root, tap_root)

    assert result.returncode == 1
    assert "latest git tag could not be determined" in result.stderr
    assert "fatal:" not in result.stderr


def test_version_check_ignores_parent_git_env_outside_git_checkout(tmp_path: Path) -> None:
    fixture_root, tap_root, _package_version = _copy_version_fixture(tmp_path)

    result = _run(fixture_root, tap_root, extra_env=_repo_git_env())

    assert result.returncode == 1
    assert "latest git tag could not be determined" in result.stderr
    assert "fatal:" not in result.stderr


def test_version_check_ignores_parent_git_env_inside_valid_checkout(tmp_path: Path) -> None:
    fixture_root, tap_root, package_version = _copy_version_fixture(tmp_path)
    _init_git_repo(fixture_root, f"v{package_version}")

    result = _run(fixture_root, tap_root, extra_env=_repo_git_env())

    assert result.returncode == 0, result.stderr
    assert f"PASS: BrainLayer/BrainBar {package_version} release metadata is consistent" in result.stdout


def test_version_check_scrubs_all_git_local_env_vars_declared_by_git() -> None:
    git_local_vars = subprocess.check_output(
        ["git", "rev-parse", "--local-env-vars"],
        text=True,
        env=_clean_git_env(),
    ).splitlines()
    script = SCRIPT.read_text(encoding="utf-8")

    missing = [name for name in git_local_vars if name not in script]
    assert missing == []
