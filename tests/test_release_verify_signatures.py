import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "release-verify-signatures.sh"


def test_reports_invalid_native_signature_and_fails(tmp_path: Path) -> None:
    native_dir = tmp_path / "keg" / "libexec" / "venv" / "native"
    native_dir.mkdir(parents=True)
    (native_dir / "valid.so").touch()
    (native_dir / "invalid.dylib").touch()
    codesign = tmp_path / "codesign"
    codesign.write_text(
        "#!/usr/bin/env bash\n"
        "[[ \"${@: -1}\" != *invalid.dylib ]] || { "
        "echo \"${@: -1}: code object is not signed at all\" >&2; exit 1; }\n"
    )
    codesign.chmod(0o755)

    result = subprocess.run(
        [SCRIPT, tmp_path / "keg"],
        env={**os.environ, "BRAINLAYER_CODESIGN_BIN": str(codesign)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "valid: 1" in result.stdout
    assert "invalid: 1" in result.stdout
    assert "invalid.dylib: code object is not signed at all" in result.stdout


def test_launchd_install_and_release_runbook_require_signature_gate() -> None:
    install = (REPO_ROOT / "scripts" / "launchd" / "install.sh").read_text()
    agents = (REPO_ROOT / "AGENTS.md").read_text()

    assert 'release-verify-signatures.sh" "$BRAINLAYER_DIR"' in install
    assert "scripts/release-verify-signatures.sh" in agents
