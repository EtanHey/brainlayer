import _ctypes
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "release-verify-signatures.sh"
INSTALL_SH = REPO_ROOT / "scripts" / "launchd" / "install.sh"


def _loadable_stdlib_extension() -> Path:
    # Some CPython builds make _ctypes built-in; use their lib-dynload extensions instead.
    if source := getattr(_ctypes, "__file__", None):
        return Path(source)
    candidates = sorted((Path(sysconfig.get_path("stdlib")) / "lib-dynload").glob("*.so"))
    assert candidates, "no standard-library native extension in this Python"
    return candidates[0]


def _write_fake_codesign(path: Path, invalid_suffix: str) -> Path:
    """Fake codesign: fail only for paths ending in ``invalid_suffix``."""
    path.write_text(
        "#!/usr/bin/env bash\n"
        f'[[ "${{@: -1}}" != *{invalid_suffix} ]] || {{ '
        'echo "${@: -1}: code object is not signed at all" >&2; exit 1; }\n'
    )
    path.chmod(0o755)
    return path


def _run(script: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script), *args],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_reports_invalid_native_signature_and_fails(tmp_path: Path) -> None:
    venv = tmp_path / "keg" / "libexec" / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    native_dir = venv / "native"
    native_dir.mkdir(parents=True)
    (native_dir / "valid.so").touch()
    (native_dir / "invalid.dylib").touch()
    codesign = _write_fake_codesign(tmp_path / "codesign", "invalid.dylib")

    result = _run(SCRIPT, str(tmp_path / "keg"), env={"BRAINLAYER_CODESIGN_BIN": str(codesign)})

    assert result.returncode == 1
    assert "valid: 1" in result.stdout
    assert "invalid: 1" in result.stdout
    assert "invalid.dylib: code object is not signed at all" in result.stdout


def test_empty_native_tree_fails_instead_of_passing(tmp_path: Path) -> None:
    venv = tmp_path / "keg" / "libexec" / "venv"
    (venv / "nothing").mkdir(parents=True)
    (venv / "bin").mkdir()
    (venv / "bin" / "python").symlink_to(sys.executable)
    codesign = _write_fake_codesign(tmp_path / "codesign", "never-matches")

    result = _run(SCRIPT, str(tmp_path / "keg"), env={"BRAINLAYER_CODESIGN_BIN": str(codesign)})

    assert result.returncode == 1
    assert "valid: 0" in result.stdout
    assert "invalid: 0" in result.stdout
    assert "ERROR: no native extensions found under" in result.stderr


def test_symlinked_native_root_outside_keg_is_rejected(tmp_path: Path) -> None:
    keg = tmp_path / "keg"
    (keg / "libexec").mkdir(parents=True)
    outside_venv = tmp_path / "outside-venv"
    (outside_venv / "bin").mkdir(parents=True)
    (outside_venv / "bin" / "python").symlink_to(sys.executable)
    native_dir = outside_venv / "native"
    native_dir.mkdir()
    shutil.copy(_loadable_stdlib_extension(), native_dir / "working.so")
    (keg / "libexec" / "venv").symlink_to(outside_venv, target_is_directory=True)
    codesign = _write_fake_codesign(tmp_path / "codesign", "never-matches")

    result = _run(SCRIPT, str(keg), env={"BRAINLAYER_CODESIGN_BIN": str(codesign)})

    assert result.returncode == 2
    assert "ERROR: native extension root escapes keg:" in result.stderr


def test_native_symlink_outside_keg_cannot_hide_unloadable_file(tmp_path: Path) -> None:
    keg = tmp_path / "keg"
    venv = keg / "libexec" / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    native_dir = venv / "lib" / "python3.13" / "site-packages"
    native_dir.mkdir(parents=True)
    shutil.copy(_loadable_stdlib_extension(), native_dir / "working.so")
    outside = tmp_path / "broken.so"
    outside.write_text("not a native library")
    (native_dir / "hidden.so").symlink_to(outside)
    codesign = _write_fake_codesign(tmp_path / "codesign", "never-matches")

    result = _run(SCRIPT, str(keg), env={"BRAINLAYER_CODESIGN_BIN": str(codesign)})

    assert result.returncode == 2
    assert "ERROR: symlink in native library tree:" in result.stderr


def test_bin_directory_symlink_cannot_hide_unloadable_file(tmp_path: Path) -> None:
    keg = tmp_path / "keg"
    venv = keg / "libexec" / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    native_dir = venv / "lib" / "python3.13" / "site-packages"
    native_dir.mkdir(parents=True)
    shutil.copy(_loadable_stdlib_extension(), native_dir / "working.so")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "hidden.so").write_text("not a native library")
    (venv / "bin" / "extensions").symlink_to(outside, target_is_directory=True)
    codesign = _write_fake_codesign(tmp_path / "codesign", "never-matches")

    result = _run(SCRIPT, str(keg), env={"BRAINLAYER_CODESIGN_BIN": str(codesign)})

    assert result.returncode == 2
    assert "ERROR: symlink in native library tree:" in result.stderr
    assert "bin/extensions" in result.stderr


def test_signed_but_unloadable_native_file_fails(tmp_path: Path) -> None:
    keg = tmp_path / "keg"
    venv = keg / "libexec" / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    native_dir = venv / "native"
    native_dir.mkdir()
    (native_dir / "broken.so").write_text("not a native library")
    codesign = _write_fake_codesign(tmp_path / "codesign", "never-matches")

    result = _run(SCRIPT, str(keg), env={"BRAINLAYER_CODESIGN_BIN": str(codesign)})

    assert result.returncode == 1
    assert "valid-signature: 1" in result.stdout
    assert "loadable: 0" in result.stdout
    assert "allowed-optional: 0" in result.stdout
    assert "LOAD_FAILED native/broken.so:" in result.stdout


def test_only_numba_omppool_is_allowed_to_fail_loading(tmp_path: Path) -> None:
    keg = tmp_path / "keg"
    venv = keg / "libexec" / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    native_dir = venv / "lib" / "python3.13" / "site-packages" / "numba" / "np" / "ufunc"
    native_dir.mkdir(parents=True)
    (native_dir / "omppool.cpython-313-darwin.so").write_text("optional but unloadable")
    codesign = _write_fake_codesign(tmp_path / "codesign", "never-matches")

    result = _run(SCRIPT, str(keg), env={"BRAINLAYER_CODESIGN_BIN": str(codesign)})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "valid-signature: 1" in result.stdout
    assert "loadable: 0" in result.stdout
    assert "allowed-optional: 1" in result.stdout
    assert (
        "ALLOWED_OPTIONAL lib/python3.13/site-packages/numba/np/ufunc/omppool.cpython-313-darwin.so:" in result.stdout
    )


def test_omppool_name_outside_numba_path_still_fails(tmp_path: Path) -> None:
    keg = tmp_path / "keg"
    venv = keg / "libexec" / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    native_dir = venv / "native"
    native_dir.mkdir()
    (native_dir / "omppool.so").write_text("not numba's optional extension")
    codesign = _write_fake_codesign(tmp_path / "codesign", "never-matches")

    result = _run(SCRIPT, str(keg), env={"BRAINLAYER_CODESIGN_BIN": str(codesign)})

    assert result.returncode == 1
    assert "LOAD_FAILED native/omppool.so:" in result.stdout
    assert "allowed-optional: 0" in result.stdout


def test_loadable_native_file_is_counted(tmp_path: Path) -> None:
    keg = tmp_path / "keg"
    venv = keg / "libexec" / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    (venv / "bin" / "python3").symlink_to("python")
    (venv / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}").symlink_to("python3")
    native_dir = venv / "native"
    native_dir.mkdir()
    shutil.copy(_loadable_stdlib_extension(), native_dir / "working.so")
    codesign = _write_fake_codesign(tmp_path / "codesign", "never-matches")

    result = _run(SCRIPT, str(keg), env={"BRAINLAYER_CODESIGN_BIN": str(codesign)})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "valid-signature: 1" in result.stdout
    assert "loadable: 1" in result.stdout
    assert "allowed-optional: 0" in result.stdout


def test_packaged_layout_is_wired_into_wheel() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()

    assert '"scripts/release-verify-signatures.sh" = "brainlayer/launchd/release-verify-signatures.sh"' in pyproject


def _first_unsigned_macho_copy(dest: Path) -> Path | None:
    """Copy one real native extension from the running interpreter and strip its signature."""
    dynload = Path(sysconfig.get_path("stdlib")) / "lib-dynload"
    for source in sorted(dynload.glob("*.so")):
        target = dest / source.name
        shutil.copy(source, target)
        stripped = subprocess.run(
            ["codesign", "--remove-signature", str(target)], capture_output=True, text=True, check=False
        )
        if stripped.returncode == 0:
            return target
        target.unlink()
    return None


NEEDS_CODESIGN = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("codesign") is None, reason="needs macOS codesign"
)


def _fake_packaged_keg(tmp_path: Path) -> tuple[Path, Path]:
    """Fake keg laid out like a brew install: install.sh lives under site-packages, keg root is an ancestor.

    Returns ``(launchd_dir, unsigned_macho)``; the unsigned Mach-O sits in a dot-dir like ``PIL/.dylibs``.
    """
    keg = tmp_path / "keg"
    python_bin = keg / "libexec" / "venv" / "bin"
    python_bin.mkdir(parents=True)
    (python_bin / "python").symlink_to(sys.executable)
    site_packages = keg / "libexec" / "venv" / "lib" / "python3.13" / "site-packages"
    launchd_dir = site_packages / "brainlayer" / "launchd"
    launchd_dir.mkdir(parents=True)
    shutil.copy(INSTALL_SH, launchd_dir / "install.sh")
    shutil.copy(SCRIPT, launchd_dir / "release-verify-signatures.sh")
    native_dir = site_packages / "PIL" / ".dylibs"
    native_dir.mkdir(parents=True)
    unsigned = _first_unsigned_macho_copy(native_dir)
    assert unsigned is not None, "no strippable Mach-O in lib-dynload"
    return launchd_dir, unsigned


def _install_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {"HOME": str(tmp_path), "BRAINLAYER_BIN": "/usr/bin/true", "PYTHON_BIN": "/usr/bin/true", **extra}


@NEEDS_CODESIGN
def test_packaged_install_sh_runs_signature_gate_and_fails_on_unsigned_keg(tmp_path: Path) -> None:
    launchd_dir, unsigned = _fake_packaged_keg(tmp_path)

    result = _run(launchd_dir / "install.sh", "all", env=_install_env(tmp_path))

    assert result.returncode != 0
    assert (
        f"INVALID lib/python3.13/site-packages/PIL/.dylibs/{unsigned.name}: code object is not signed at all"
        in result.stdout
    )
    assert "invalid: 1" in result.stdout
    assert not (tmp_path / "Library" / "LaunchAgents").exists(), "gate must abort before any plist is installed"


@NEEDS_CODESIGN
def test_packaged_install_sh_load_is_gated_before_any_launchctl_call(tmp_path: Path) -> None:
    """`install.sh load <name>` bootstraps a service from the keg, so it must not bypass the gate (#748 P1)."""
    launchd_dir, unsigned = _fake_packaged_keg(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_calls = tmp_path / "launchctl.calls"
    (fake_bin / "launchctl").write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{launchctl_calls}"\n')
    (fake_bin / "launchctl").chmod(0o755)

    result = _run(
        launchd_dir / "install.sh",
        "load",
        "drain",
        env=_install_env(tmp_path, PATH=f"{fake_bin}:{os.environ['PATH']}"),
    )

    assert result.returncode != 0
    assert f"INVALID lib/python3.13/site-packages/PIL/.dylibs/{unsigned.name}" in result.stdout
    assert "invalid: 1" in result.stdout
    assert not launchctl_calls.exists(), "gate must fail before install.sh touches launchctl"


def test_install_sh_gate_bypass_is_only_for_teardown_actions() -> None:
    """Static guard for the #748 P1: only remove/unload skip the gate; load (which bootstraps) must not."""
    install = INSTALL_SH.read_text(encoding="utf-8")
    gate_case = install.split("find_release_verify_script()", 1)[1].split("esac", 1)[0]

    assert 'case "$BRAINLAYER_INSTALL_ACTION" in\n    remove|unload)\n        ;;\n    *)\n' in gate_case


def test_source_checkout_install_sh_skips_gate_without_keg(tmp_path: Path) -> None:
    """Source checkout: scripts/launchd/install.sh has no libexec/venv ancestor -> gate is a clean skip."""
    checkout = tmp_path / "checkout"
    launchd_dir = checkout / "scripts" / "launchd"
    launchd_dir.mkdir(parents=True)
    shutil.copy(INSTALL_SH, launchd_dir / "install.sh")
    shutil.copy(SCRIPT, checkout / "scripts" / "release-verify-signatures.sh")
    install_source = INSTALL_SH.read_text(encoding="utf-8")
    marker = (
        'case "$BRAINLAYER_INSTALL_ACTION" in\n    remove|unload|load)\n        ;;\n    *)\n        if [ "$(uname -s)"'
    )
    assert marker in install_source, "install.sh gate-block marker moved; update this test"
    gate_only = install_source.split(marker, 1)[0]
    # The slice must carry the whole signature-gate block, so the skip branch below really executes it.
    assert '"$BRAINLAYER_RELEASE_VERIFY" "$BRAINLAYER_KEG"' in gate_only
    assert gate_only.rstrip().endswith("esac"), "gate block must be complete through its closing esac"
    harness = launchd_dir / "install.sh"
    harness.write_text(gate_only + '\necho "GATE_SKIPPED"\n', encoding="utf-8")
    harness.chmod(0o755)

    result = _run(
        harness, "all", env={"HOME": str(tmp_path), "BRAINLAYER_BIN": "/usr/bin/true", "PYTHON_BIN": "/usr/bin/true"}
    )

    assert result.returncode == 0, result.stderr
    assert "GATE_SKIPPED" in result.stdout
    assert "INVALID" not in result.stdout


def test_release_runbook_and_installer_name_signature_gate() -> None:
    """Static, platform-independent guard: the gate stays wired even where the darwin test is skipped."""
    agents = (REPO_ROOT / "AGENTS.md").read_text()
    install = INSTALL_SH.read_text(encoding="utf-8")

    assert "scripts/release-verify-signatures.sh" in agents
    assert "find_brainlayer_keg()" in install
    assert '"$BRAINLAYER_RELEASE_VERIFY" "$BRAINLAYER_KEG"' in install
