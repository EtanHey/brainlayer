from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import get_type_hints

from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_brainlayer_cli_entrypoint_imports_typer_app() -> None:
    from brainlayer.cli import app

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "brainlayer" in result.stdout


def test_launchd_templates_are_declared_as_package_data() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = payload["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert force_include["scripts/launchd"] == "brainlayer/launchd"

    plist_paths = sorted((REPO_ROOT / "scripts" / "launchd").glob("com.brainlayer.*.plist"))
    assert len(plist_paths) >= 10
    for path in plist_paths:
        plist = plistlib.loads(path.read_bytes())
        assert plist["Label"].startswith("com.brainlayer.")


def test_setup_invokes_launchd_install_script_with_env_file(tmp_path: Path) -> None:
    from brainlayer.setup import install_launchd

    launchd_dir = tmp_path / "launchd"
    launchd_dir.mkdir()
    install_script = launchd_dir / "install.sh"
    marker = tmp_path / "called.txt"
    install_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'printf "%s\\n" "$BRAINLAYER_ENV_FILE" "$1" > "$CALL_MARKER"',
            ]
        ),
        encoding="utf-8",
    )
    install_script.chmod(0o755)
    env_file = tmp_path / "brainlayer.env"

    install_launchd("watch", env_file=env_file, launchd_dir=launchd_dir, extra_env={"CALL_MARKER": str(marker)})

    assert marker.read_text(encoding="utf-8").splitlines() == [str(env_file), "watch"]


def test_setup_command_writes_op_backed_env_without_plaintext_and_can_skip_launchd(tmp_path: Path) -> None:
    from brainlayer.cli import app

    env_file = tmp_path / "brainlayer.env"
    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--no-launchd",
            "--env-file",
            str(env_file),
            "--google-api-key-op-ref",
            "op://Private/Google AI/Gemini API key",
        ],
    )

    assert result.exit_code == 0, result.stdout
    content = env_file.read_text(encoding="utf-8")
    assert "GOOGLE_API_KEY=\"$(op read 'op://Private/Google AI/Gemini API key')\"" in content
    assert "AIza" not in content
    assert oct(env_file.stat().st_mode & 0o777) == "0o600"


def test_setup_command_env_file_annotation_accepts_none() -> None:
    from brainlayer.cli import setup

    assert get_type_hints(setup)["env_file"] == Path | None


def test_setup_command_reports_launchd_failure_without_traceback(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers
    from brainlayer.cli import app

    def fail_install(*args, **kwargs):
        raise FileNotFoundError("missing install.sh")

    monkeypatch.setattr(setup_helpers, "install_launchd", fail_install)
    env_file = tmp_path / "brainlayer.env"

    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--env-file",
            str(env_file),
            "--google-api-key-op-ref",
            "op://Private/Google AI/Gemini API key",
        ],
    )

    assert result.exit_code == 1
    assert "BrainLayer setup failed: missing install.sh" in result.stdout
    assert "Traceback" not in result.stdout


def test_config_loader_prefers_process_env_then_user_env_and_ignores_repo_root_dotenv(
    tmp_path: Path, monkeypatch
) -> None:
    from brainlayer.config import load_brainlayer_env

    user_env = tmp_path / "brainlayer.env"
    user_env.write_text(
        "\n".join(
            [
                "BRAINLAYER_FROM_USER=ok",
                "BRAINLAYER_EXISTING=from-file",
                "GOOGLE_API_KEY=\"$(op read 'op://Private/Google AI/Gemini API key')\"",
            ]
        ),
        encoding="utf-8",
    )
    repo_env = tmp_path / ".env"
    repo_env.write_text("BRAINLAYER_FROM_REPO=deprecated\n", encoding="utf-8")
    monkeypatch.setenv("BRAINLAYER_EXISTING", "from-process")
    monkeypatch.delenv("BRAINLAYER_FROM_USER", raising=False)
    monkeypatch.delenv("BRAINLAYER_FROM_REPO", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    loaded = load_brainlayer_env(user_env, repo_env_path=repo_env)

    assert loaded == {"BRAINLAYER_FROM_USER": "ok"}
    assert os.environ["BRAINLAYER_FROM_USER"] == "ok"
    assert os.environ["BRAINLAYER_EXISTING"] == "from-process"
    assert "BRAINLAYER_FROM_REPO" not in os.environ
    assert "GOOGLE_API_KEY" not in os.environ


def test_wheel_contains_cli_and_launchd_templates(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "dist"
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheel = next(wheel_dir.glob("brainlayer-*.whl"))
    listing = subprocess.run(
        [sys.executable, "-m", "zipfile", "-l", str(wheel)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert "brainlayer/cli/__init__.py" in listing
    assert "brainlayer/cli_new.py" in listing
    assert "brainlayer/launchd/install.sh" in listing
    assert "brainlayer/launchd/com.brainlayer.enrichment.plist" in listing
