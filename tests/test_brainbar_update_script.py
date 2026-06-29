"""BrainBar cask update and dedupe script contracts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPDATE_SCRIPT = REPO_ROOT / "scripts" / "brainlayer-update-brainbar.sh"
DEDUPE_SCRIPT = REPO_ROOT / "scripts" / "brainlayer-dedupe-brainbar.sh"


def _run_update(env: dict[str, str] | None = None, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(UPDATE_SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=30,
    )


def test_update_brainbar_reinstalls_existing_homebrew_cask_in_dry_run() -> None:
    result = _run_update({"BRAINLAYER_UPDATE_TEST_BREW_CASK_INSTALLED": "1"}, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "BRAINBAR APP UPDATE: brew reinstall --cask etanhey/layers/brainbar" in result.stdout
    assert "+ brew reinstall --cask etanhey/layers/brainbar" in result.stdout


def test_update_brainbar_installs_homebrew_cask_when_not_installed_in_dry_run() -> None:
    result = _run_update({"BRAINLAYER_UPDATE_TEST_BREW_CASK_INSTALLED": "0"}, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "BRAINBAR APP UPDATE: brew install --cask etanhey/layers/brainbar" in result.stdout
    assert "+ brew install --cask etanhey/layers/brainbar" in result.stdout


def test_update_brainbar_non_dry_run_can_be_command_stubbed_for_tests() -> None:
    result = _run_update(
        {
            "BRAINLAYER_UPDATE_TEST_BREW_CASK_INSTALLED": "1",
            "BRAINLAYER_UPDATE_DRY_RUN_COMMANDS": "1",
        }
    )

    assert result.returncode == 0, result.stderr
    assert "+ brew reinstall --cask etanhey/layers/brainbar" in result.stdout


def test_update_brainbar_uses_configured_cask_token_for_detection(tmp_path: Path) -> None:
    log = tmp_path / "brew.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    brew = bin_dir / "brew"
    brew.write_text(
        f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "{log}"
exit 1
""",
        encoding="utf-8",
    )
    brew.chmod(0o755)

    result = _run_update(
        {
            "BRAINLAYER_UPDATE_BRAINBAR_CASK_TOKEN": "custom/tap/custombrainbar",
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        },
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert "list --cask custombrainbar" in log.read_text(encoding="utf-8")
    assert "BRAINBAR APP UPDATE: brew install --cask custom/tap/custombrainbar" in result.stdout


def test_update_brainbar_documents_recovery_no_sudo_path() -> None:
    script = UPDATE_SCRIPT.read_text(encoding="utf-8")

    assert "recovery-no-sudo" in script
    assert "Contents/Resources/LaunchAgents" in script
    assert "com.brainlayer.brainbar-daemon" in script
    assert "com.brainlayer.brainbar" in script
    assert "brew reinstall --cask etanhey/layers/brainbar" in script


def test_dedupe_brainbar_dry_run_makes_no_filesystem_changes(tmp_path: Path) -> None:
    home = tmp_path / "home"
    canonical_app = tmp_path / "Applications" / "BrainBar.app"
    stray_app = home / "Applications" / "BrainBar.app"
    wrong_bundle_app = tmp_path / "wrong-bundle" / "BrainBar.app"
    (canonical_app / "Contents").mkdir(parents=True)
    (stray_app / "Contents").mkdir(parents=True)
    (wrong_bundle_app / "Contents").mkdir(parents=True)
    (canonical_app / "Contents" / "marker").write_text("canonical", encoding="utf-8")
    (stray_app / "Contents" / "marker").write_text("stray", encoding="utf-8")
    (wrong_bundle_app / "Contents" / "marker").write_text("wrong", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("spctl", "xcrun"):
        path = bin_dir / tool
        path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    defaults = bin_dir / "defaults"
    defaults.write_text(
        """#!/usr/bin/env bash
case "$2" in
  *wrong-bundle*) printf 'com.example.not-brainbar\n' ;;
  *) printf 'com.brainlayer.brainbar\n' ;;
esac
""",
        encoding="utf-8",
    )
    defaults.chmod(0o755)

    result = subprocess.run(
        ["bash", str(DEDUPE_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HOME": str(home),
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "BRAINLAYER_DEDUPE_BRAINBAR_CANONICAL_APP": str(canonical_app),
            "BRAINLAYER_DEDUPE_BRAINBAR_SEARCH_ROOTS": str(tmp_path),
        },
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "DRY-RUN complete. Re-run with --apply to execute. Nothing was changed." in result.stdout
    assert (canonical_app / "Contents" / "marker").read_text(encoding="utf-8") == "canonical"
    assert (stray_app / "Contents" / "marker").read_text(encoding="utf-8") == "stray"
    assert str(wrong_bundle_app) not in result.stdout
    assert not (home / ".brainlayer" / "brainbar-dedupe-backup").exists()


def test_dedupe_brainbar_rejects_wrong_canonical_bundle_id(tmp_path: Path) -> None:
    home = tmp_path / "home"
    canonical_app = tmp_path / "Applications" / "BrainBar.app"
    (canonical_app / "Contents").mkdir(parents=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("spctl", "xcrun"):
        path = bin_dir / tool
        path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    defaults = bin_dir / "defaults"
    defaults.write_text("#!/usr/bin/env bash\nprintf 'com.example.not-brainbar\\n'\n", encoding="utf-8")
    defaults.chmod(0o755)

    result = subprocess.run(
        ["bash", str(DEDUPE_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HOME": str(home),
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "BRAINLAYER_DEDUPE_BRAINBAR_CANONICAL_APP": str(canonical_app),
            "BRAINLAYER_DEDUPE_BRAINBAR_SEARCH_ROOTS": str(tmp_path),
        },
        timeout=30,
    )

    assert result.returncode == 1
    assert "Expected bundle id: com.brainlayer.brainbar; found: com.example.not-brainbar" in result.stdout


def test_dedupe_brainbar_script_is_dry_run_safe_and_preserves_canonical_app() -> None:
    script = DEDUPE_SCRIPT.read_text(encoding="utf-8")

    assert "SAFE BY DEFAULT" in script
    assert "--apply" in script
    assert "/Applications/BrainBar.app" in script
    assert "xcrun stapler validate" in script
    assert "BACKUP_DIR=" in script
    assert "keep canonical user LaunchAgent" in script
    assert 'rm -rf "$CANONICAL_APP"' not in script
    assert 'run mv "$bundle" "$DEST_BACKUP/bundles/$(echo "$bundle" | tr \'/ \' \'__\')" || true' not in script
    assert 'run mv "$file" "$DEST_BACKUP/LaunchAgents/" || true' not in script
