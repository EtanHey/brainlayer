"""BrainBar cask update and dedupe script contracts.

The update script implements the drift-proof contract ratified 2026-08-19
(collab/2026-08-19-drift-proof-mac-sync.md). Every external tool is stubbed here so
the tests exercise the real decision logic without touching Homebrew or /Applications.
"""

from __future__ import annotations

import json
import os
import plistlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPDATE_SCRIPT = REPO_ROOT / "scripts" / "brainlayer-update-brainbar.sh"
DEDUPE_SCRIPT = REPO_ROOT / "scripts" / "brainlayer-dedupe-brainbar.sh"

CASK_TOKEN = "etanhey/layers/brainbar"


def _write_exec(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _make_app(path: Path, version: str) -> Path:
    contents = path / "Contents"
    contents.mkdir(parents=True, exist_ok=True)
    with (contents / "Info.plist").open("wb") as handle:
        plistlib.dump({"CFBundleShortVersionString": version}, handle)
    return path


class _Harness:
    """A fully stubbed world for one update-script run."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.home = tmp_path / "home"
        self.home.mkdir(parents=True, exist_ok=True)
        self.prefix = tmp_path / "brew-prefix"
        self.caskroom = self.prefix / "Caskroom" / "brainbar"
        self.quarantine = tmp_path / "quarantine"
        self.app_path = tmp_path / "Applications" / "BrainBar.app"
        self.brew_log = tmp_path / "brew.log"
        self.git_log = tmp_path / "git.log"
        self.sudo_log = tmp_path / "sudo.log"
        self.bin_dir = tmp_path / "bin"
        self.tap_dir = tmp_path / "tap"
        (self.tap_dir / ".git").mkdir(parents=True, exist_ok=True)

    def install_stubs(self, *, offered: str, registered: str | None) -> None:
        info = json.dumps({"casks": [{"version": offered}]})
        listed_cmd = "printf '%s\\n' " + repr(f"brainbar {registered}") if registered else "true"
        _write_exec(
            self.bin_dir / "brew",
            f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> {self.brew_log!s}
case "$1 $2" in
  "--prefix ") printf '%s\\n' {self.prefix!s}; exit 0 ;;
  "--repository ") printf '%s\\n' {self.tmp_path!s}/brew-repo; exit 0 ;;
esac
case "$*" in
  "list --versions --cask brainbar") {listed_cmd}; exit 0 ;;
  "info --cask --json=v2 {CASK_TOKEN}") printf '%s\\n' {info!r}; exit 0 ;;
  "list --versions brainlayer") printf 'brainlayer 0.1.0\\n'; exit 0 ;;
esac
exit 0
""",
        )
        _write_exec(
            self.bin_dir / "git",
            f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> {self.git_log!s}
exit 0
""",
        )
        _write_exec(self.bin_dir / "launchctl", "#!/usr/bin/env bash\nexit 0\n")
        _write_exec(
            self.bin_dir / "defaults",
            """#!/usr/bin/env bash
# usage: defaults read <plist> <key>
exec python3 - "$2" "$3" <<'PY'
import plistlib, sys
try:
    with open(sys.argv[1], "rb") as handle:
        value = plistlib.load(handle)[sys.argv[2]]
except Exception:
    raise SystemExit(1)
print(value)
PY
""",
        )
        _write_exec(
            self.bin_dir / "sudo",
            f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> {self.sudo_log!s}
exit 1
""",
        )

    def env(self, **extra: str) -> dict[str, str]:
        env = {
            **os.environ,
            "HOME": str(self.home),
            "PATH": f"{self.bin_dir}{os.pathsep}{os.environ['PATH']}",
            "BRAINLAYER_UPDATE_BREW_BIN": str(self.bin_dir / "brew"),
            "BRAINLAYER_UPDATE_GIT_BIN": str(self.bin_dir / "git"),
            "BRAINLAYER_UPDATE_LAUNCHCTL_BIN": str(self.bin_dir / "launchctl"),
            "BRAINLAYER_UPDATE_DEFAULTS_BIN": str(self.bin_dir / "defaults"),
            "BRAINLAYER_UPDATE_BRAINBAR_APP": str(self.app_path),
            "BRAINLAYER_UPDATE_TAP_DIR": str(self.tap_dir),
            "BRAINLAYER_UPDATE_QUARANTINE_DIR": str(self.quarantine),
            "BRAINLAYER_UPDATE_SKIP_VERIFY": "1",
        }
        env.update(extra)
        return env

    def run(self, *args: str, **extra_env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(UPDATE_SCRIPT), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=self.env(**extra_env),
            timeout=30,
        )

    @property
    def brew_calls(self) -> str:
        return self.brew_log.read_text(encoding="utf-8") if self.brew_log.exists() else ""

    @property
    def git_calls(self) -> str:
        return self.git_log.read_text(encoding="utf-8") if self.git_log.exists() else ""

    @property
    def sudo_calls(self) -> str:
        return self.sudo_log.read_text(encoding="utf-8") if self.sudo_log.exists() else ""


# --- rule 1: detect drift before acting ---------------------------------------------------


def test_unmanaged_app_is_reported_as_drift(tmp_path: Path) -> None:
    """Today's VoiceBar disease: a real app brew has never heard of."""
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered=None)
    _make_app(h.app_path, "1.5.8")

    result = h.run("--dry-run")

    assert result.returncode == 0, result.stderr
    assert "drift:       unmanaged" in result.stdout
    assert "app version: 1.5.8" in result.stdout
    assert "registered:  <not registered with brew>" in result.stdout


def test_stale_ledger_is_reported_as_drift(tmp_path: Path) -> None:
    """Brew's ledger said 2.1.10 while /Applications held 2.2.5."""
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered="1.5.2")
    _make_app(h.app_path, "1.5.8")

    result = h.run("--dry-run")

    assert result.returncode == 0, result.stderr
    assert "drift:       stale-ledger" in result.stdout


def test_in_sync_and_current_reports_no_drift(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered="1.5.8")
    _make_app(h.app_path, "1.5.8")

    result = h.run("--dry-run")

    assert result.returncode == 0, result.stderr
    assert "drift:       none" in result.stdout


def test_missing_install_is_reported_as_missing(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered=None)

    result = h.run("--dry-run")

    assert result.returncode == 0, result.stderr
    assert "drift:       missing" in result.stdout


# --- rule 2: on drift, never `brew upgrade`/`reinstall`; clear the ledger and force-adopt ---


def test_drift_clears_stale_caskroom_then_force_adopts(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered="1.5.2")
    _make_app(h.app_path, "1.5.8")
    (h.caskroom / "1.5.2").mkdir(parents=True)
    (h.caskroom / "1.5.2" / "marker").write_text("stale", encoding="utf-8")

    result = h.run()

    assert result.returncode == 0, result.stderr
    assert not h.caskroom.exists(), "stale Caskroom registration was not cleared"
    quarantined = list(h.quarantine.glob("*/brainbar/1.5.2/marker"))
    assert quarantined, "quarantine is not reversible — the old registration was destroyed"
    assert quarantined[0].read_text(encoding="utf-8") == "stale"
    assert f"install --cask --force {CASK_TOKEN}" in h.brew_calls


def test_update_never_runs_brew_upgrade_or_reinstall(tmp_path: Path) -> None:
    """upgrade/reinstall both execute the OLD saved cask's uninstall recipe."""
    for registered in (None, "1.5.2", "1.5.8"):
        h = _Harness(tmp_path / f"case-{registered}")
        h.install_stubs(offered="1.5.8", registered=registered)
        _make_app(h.app_path, "1.5.8")

        result = h.run()

        assert result.returncode == 0, result.stderr
        assert "upgrade" not in h.brew_calls, f"brew upgrade ran (registered={registered})"
        assert "reinstall" not in h.brew_calls, f"brew reinstall ran (registered={registered})"


def test_no_drift_is_a_no_op(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered="1.5.8")
    _make_app(h.app_path, "1.5.8")

    result = h.run()

    assert result.returncode == 0, result.stderr
    assert "Already canonical at 1.5.8 — nothing to do." in result.stdout
    assert "install --cask" not in h.brew_calls


def test_missing_install_uses_force_adopt(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered=None)

    result = h.run()

    assert result.returncode == 0, result.stderr
    assert f"install --cask --force {CASK_TOKEN}" in h.brew_calls


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered="1.5.2")
    _make_app(h.app_path, "1.5.8")
    (h.caskroom / "1.5.2").mkdir(parents=True)

    result = h.run("--dry-run")

    assert result.returncode == 0, result.stderr
    assert h.caskroom.exists(), "dry run moved the Caskroom registration"
    assert not h.quarantine.exists()
    assert "Dry run complete. Nothing was changed." in result.stdout


# --- rule 3: never require sudo/TTY; stop BEFORE destroying anything -----------------------


def test_root_owned_path_stops_before_touching_anything(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered="1.5.2")
    _make_app(h.app_path, "1.5.8")
    (h.caskroom / "1.5.2").mkdir(parents=True)

    # `stat` is what the guard consults; stub it to report a root-owned app.
    _write_exec(
        h.bin_dir / "stat",
        f"""#!/usr/bin/env bash
if [ "$3" = "{h.app_path}" ]; then printf '0\n'; else printf '%s\n' "$(id -u)"; fi
""",
    )

    result = h.run()

    assert result.returncode == 3, result.stdout + result.stderr
    assert "would shell out to sudo and abort without a TTY" in result.stderr
    assert "Nothing has been changed." in result.stderr
    assert h.caskroom.exists(), "the guard fired AFTER destroying the registration"
    assert "install --cask" not in h.brew_calls


def test_script_never_invokes_sudo(tmp_path: Path) -> None:
    """Behavioral, not grep: a logging `sudo` stub must never be called, from any state."""
    for label, registered, installed in (
        ("unmanaged", None, "1.5.8"),
        ("stale-ledger", "1.5.2", "1.5.8"),
        ("current", "1.5.8", "1.5.8"),
        ("missing", None, None),
    ):
        h = _Harness(tmp_path / f"sudo-{label}")
        h.install_stubs(offered="1.5.8", registered=registered)
        if installed:
            _make_app(h.app_path, installed)
        if registered:
            (h.caskroom / registered).mkdir(parents=True)

        result = h.run()

        assert result.returncode == 0, f"{label}: {result.stdout}{result.stderr}"
        assert h.sudo_calls == "", f"{label}: script shelled out to sudo: {h.sudo_calls}"


# --- rule 4: refresh the tap explicitly ----------------------------------------------------


def test_tap_is_pulled_explicitly_with_a_named_remote_and_branch(tmp_path: Path) -> None:
    """`brew update` does not refresh this tap, and it has no upstream branch."""
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered="1.5.8")
    _make_app(h.app_path, "1.5.8")

    result = h.run()

    assert result.returncode == 0, result.stderr
    assert f"-C {h.tap_dir} pull --ff-only origin main" in h.git_calls


def test_skip_tap_update_flag_suppresses_the_pull(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered="1.5.8")
    _make_app(h.app_path, "1.5.8")

    result = h.run("--skip-tap-update")

    assert result.returncode == 0, result.stderr
    assert h.git_calls == ""


# --- rule 5: absolute brew path ------------------------------------------------------------


def test_script_defaults_to_the_absolute_brew_path() -> None:
    script = UPDATE_SCRIPT.read_text(encoding="utf-8")
    assert 'BREW_BIN="${BRAINLAYER_UPDATE_BREW_BIN:-/opt/homebrew/bin/brew}"' in script


def test_missing_brew_fails_loudly(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered=None)

    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    result = h.run(
        BRAINLAYER_UPDATE_BREW_BIN=str(tmp_path / "nope" / "brew"),
        PATH=f"{empty_bin}{os.pathsep}/usr/bin{os.pathsep}/bin",
    )

    assert result.returncode == 127, result.stdout + result.stderr
    assert "Homebrew not found" in result.stderr


# --- rule 6: verify at the end, fail loudly ------------------------------------------------


def test_verify_only_fails_loudly_when_not_green(tmp_path: Path) -> None:
    h = _Harness(tmp_path)
    h.install_stubs(offered="1.5.8", registered="1.5.2")
    _make_app(h.app_path, "1.5.2")

    result = h.run("--verify-only", BRAINLAYER_UPDATE_SKIP_VERIFY="0")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "[FAIL] app version: 1.5.2 (expected 1.5.8)" in result.stdout
    assert "[FAIL] cask version: 1.5.2 (expected 1.5.8)" in result.stdout
    assert "BrainBar is NOT green" in result.stderr


def test_verify_checks_every_contracted_signal() -> None:
    script = UPDATE_SCRIPT.read_text(encoding="utf-8")
    for signal in (
        '"app version"',
        '"cask version"',
        '"brainlayer formula"',
        '"BrainBar process"',
        '"launchd $label"',
        '"socket"',
    ):
        assert signal in script, f"verification does not cover {signal}"


# --- documentation contract -----------------------------------------------------------------


def test_update_brainbar_documents_recovery_no_sudo_path() -> None:
    script = UPDATE_SCRIPT.read_text(encoding="utf-8")

    assert "recovery-no-sudo" in script
    assert "Contents/Resources/LaunchAgents" in script
    assert "com.brainlayer.brainbar-daemon" in script
    assert "com.brainlayer.brainbar" in script


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
