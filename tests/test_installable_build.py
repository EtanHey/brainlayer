from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import get_type_hints

from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[1]


def _plist_args(name: str) -> list[str]:
    plist_path = REPO_ROOT / "scripts" / "launchd" / f"com.brainlayer.{name}.plist"
    return plistlib.loads(plist_path.read_bytes())["ProgramArguments"]


def _write_full_launchd_env(env_file: Path) -> None:
    env_file.write_text(
        "\n".join(
            [
                "GOOGLE_API_KEY=test-key",
                "BRAINLAYER_ENRICH_ENABLED=1",
                "BRAINLAYER_ENRICH_MODE=realtime",
                "BRAINLAYER_ENRICH_PROVIDER=google",
                "BRAINLAYER_ENRICH_BACKEND=gemini",
                "BRAINLAYER_ENRICH_RATE=1",
                "BRAINLAYER_ENRICH_CONCURRENCY=1",
                "BRAINLAYER_MAX_COMMIT_BATCH=1",
                "BRAINLAYER_GEMINI_SERVICE_TIER=flex",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)


def test_brainlayer_cli_entrypoint_imports_typer_app(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    from brainlayer.cli import app

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "brainlayer" in result.stdout


def test_brainlayer_cli_exposes_transport_commands(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    from brainlayer.cli import app

    for args in (["drain", "--help"], ["status", "--help"]):
        result = CliRunner().invoke(app, args)

        assert result.exit_code == 0, result.stdout


def test_launchd_templates_are_declared_as_package_data() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = payload["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert force_include["scripts/launchd"] == "brainlayer/launchd"

    plist_paths = sorted((REPO_ROOT / "scripts" / "launchd").glob("com.brainlayer.*.plist"))
    assert len(plist_paths) >= 10
    for path in plist_paths:
        plist = plistlib.loads(path.read_bytes())
        assert plist["Label"].startswith("com.brainlayer.")


def test_setup_invokes_launchd_install_script_with_env_file(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers
    from brainlayer.setup import install_launchd

    launchd_dir = tmp_path / "launchd"
    launchd_dir.mkdir()
    fake_brainlayer = tmp_path / "tool" / "bin" / "brainlayer"
    fake_brainlayer.parent.mkdir(parents=True)
    fake_brainlayer.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    fake_brainlayer.chmod(0o755)
    monkeypatch.setattr(setup_helpers.sys, "argv", [str(fake_brainlayer)])
    install_script = launchd_dir / "install.sh"
    marker = tmp_path / "called.txt"
    install_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'printf "%s\\n" "$BRAINLAYER_ENV_FILE" "$PYTHON_BIN" "$BRAINLAYER_PYTHON" "$BRAINLAYER_BIN" "$1" > "$CALL_MARKER"',
            ]
        ),
        encoding="utf-8",
    )
    install_script.chmod(0o755)
    env_file = tmp_path / "brainlayer.env"

    install_launchd("watch", env_file=env_file, launchd_dir=launchd_dir, extra_env={"CALL_MARKER": str(marker)})

    assert marker.read_text(encoding="utf-8").splitlines() == [
        str(env_file),
        sys.executable,
        sys.executable,
        str(fake_brainlayer),
        "watch",
    ]


def test_install_launchd_times_out_with_clear_error(tmp_path: Path) -> None:
    from brainlayer.setup import install_launchd

    launchd_dir = tmp_path / "launchd"
    launchd_dir.mkdir()
    install_script = launchd_dir / "install.sh"
    install_script.write_text("#!/usr/bin/env bash\nsleep 5\n", encoding="utf-8")
    install_script.chmod(0o755)

    try:
        install_launchd("watch", launchd_dir=launchd_dir, timeout_seconds=0.01)
    except TimeoutError as exc:
        assert "launchd installer timed out after 0.01s" in str(exc)
    else:
        raise AssertionError("install_launchd did not time out")


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


def test_launchd_env_runner_makes_homebrew_op_available_before_loading_env() -> None:
    runner = REPO_ROOT / "scripts" / "launchd" / "brainlayer-env-run.sh"
    content = runner.read_text(encoding="utf-8")

    path_export_index = content.index("export PATH=")
    load_index = content.index("load_simple_env_file")
    assert "/opt/homebrew/bin" in content[path_export_index:load_index]
    assert path_export_index < load_index


def test_setup_command_does_not_install_launchd_by_default(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers
    from brainlayer.cli import app

    def fail_install(*args, **kwargs):
        raise AssertionError("launchd should be opt-in")

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

    assert result.exit_code == 0, result.stdout
    assert env_file.exists()


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
            "--launchd",
            "--env-file",
            str(env_file),
            "--google-api-key-op-ref",
            "op://Private/Google AI/Gemini API key",
        ],
    )

    assert result.exit_code == 1
    assert "BrainLayer setup failed: missing install.sh" in result.stdout
    assert "Traceback" not in result.stdout


def test_setup_command_reports_env_write_failure_without_traceback(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers
    from brainlayer.cli import app

    def fail_env(*args, **kwargs):
        raise ValueError("empty op reference")

    monkeypatch.setattr(setup_helpers, "ensure_brainlayer_env", fail_env)

    result = CliRunner().invoke(app, ["setup", "--no-launchd", "--env-file", str(tmp_path / "brainlayer.env")])

    assert result.exit_code == 1
    assert "BrainLayer setup failed: empty op reference" in result.stdout
    assert "Traceback" not in result.stdout


def test_init_command_reports_launchd_failure_without_traceback(monkeypatch) -> None:
    import brainlayer.cli as cli
    import brainlayer.cli.wizard as wizard
    import brainlayer.setup as setup_helpers
    from brainlayer.cli import app

    class Config:
        gemini_env_file = Path("/tmp/brainlayer.env")

    monkeypatch.setattr(wizard, "run_wizard", lambda: Config())
    monkeypatch.setattr(
        setup_helpers,
        "install_launchd",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing install.sh")),
    )
    monkeypatch.setattr(cli, "install_launchd_agents", setup_helpers.install_launchd, raising=False)

    result = CliRunner().invoke(app, ["init", "--install-launchd"])

    assert result.exit_code == 1
    assert "BrainLayer init failed: missing install.sh" in result.stdout
    assert "Traceback" not in result.stdout


def test_config_loader_prefers_process_env_then_user_env_and_ignores_repo_root_dotenv(
    tmp_path: Path, monkeypatch
) -> None:
    import brainlayer.config as config
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

    def fake_run(args, **kwargs):
        assert args == ["op", "read", "op://Private/Google AI/Gemini API key"]
        return subprocess.CompletedProcess(args, 0, stdout="resolved-secret\n", stderr="")

    monkeypatch.setattr(config.subprocess, "run", fake_run)

    loaded = load_brainlayer_env(user_env, repo_env_path=repo_env)

    assert loaded == {"BRAINLAYER_FROM_USER": "ok", "GOOGLE_API_KEY": "resolved-secret"}
    assert os.environ["BRAINLAYER_FROM_USER"] == "ok"
    assert os.environ["BRAINLAYER_EXISTING"] == "from-process"
    assert "BRAINLAYER_FROM_REPO" not in os.environ
    assert os.environ["GOOGLE_API_KEY"] == "resolved-secret"


def test_config_loader_ignores_shell_substitution_that_is_not_op_read(tmp_path: Path, monkeypatch) -> None:
    from brainlayer.config import load_brainlayer_env

    user_env = tmp_path / "brainlayer.env"
    user_env.write_text('GOOGLE_API_KEY="$(cat /tmp/key)"\n', encoding="utf-8")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert load_brainlayer_env(user_env) == {}
    assert "GOOGLE_API_KEY" not in os.environ


def test_config_loader_does_not_resolve_op_when_process_env_already_has_key(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.config as config
    from brainlayer.config import load_brainlayer_env

    user_env = tmp_path / "brainlayer.env"
    user_env.write_text("GOOGLE_API_KEY=\"$(op read 'op://Private/Google AI/Gemini API key')\"\n", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_API_KEY", "from-process")

    def fail_run(*args, **kwargs):
        raise AssertionError("op read should not run when process env has GOOGLE_API_KEY")

    monkeypatch.setattr(config.subprocess, "run", fail_run)

    assert load_brainlayer_env(user_env) == {}
    assert os.environ["GOOGLE_API_KEY"] == "from-process"


def test_config_loader_ignores_unreadable_user_env(monkeypatch, tmp_path: Path) -> None:
    from brainlayer.config import load_brainlayer_env

    user_env = tmp_path / "brainlayer.env"
    user_env.write_text("BRAINLAYER_FROM_USER=ok\n", encoding="utf-8")

    original_read_text = Path.read_text

    def fail_read_text(path: Path, *args, **kwargs):
        if path == user_env:
            raise PermissionError("blocked")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    assert load_brainlayer_env(user_env, repo_env_path=tmp_path / ".env") == {}


def test_launchd_repo_only_wrappers_are_not_used_by_packaged_plists() -> None:
    assert _plist_args("drain")[:3] == ["__BRAINLAYER_ENV_RUN__", "__PYTHON_BIN__", "-m"]
    assert _plist_args("drain")[3] == "brainlayer.drain"
    assert _plist_args("maintenance-nightly")[:3] == ["__BRAINLAYER_ENV_RUN__", "__PYTHON_BIN__", "-m"]
    assert _plist_args("maintenance-nightly")[3] == "brainlayer.maintenance"
    assert _plist_args("maintenance-weekly")[:3] == ["__BRAINLAYER_ENV_RUN__", "__PYTHON_BIN__", "-m"]
    assert _plist_args("maintenance-weekly")[3] == "brainlayer.maintenance"


def test_drain_module_supports_python_m_execution() -> None:
    content = (REPO_ROOT / "src" / "brainlayer" / "drain.py").read_text(encoding="utf-8")

    assert 'if __name__ == "__main__":' in content
    assert "raise SystemExit(main())" in content


def test_launchd_installer_preflights_all_before_loading_without_google_key(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=1\n", encoding="utf-8")
    env_file.chmod(0o600)
    home = tmp_path / "home"
    home.mkdir()

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "all"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "did not provide GOOGLE_API_KEY" in result.stdout
    assert not launchctl_log.exists()
    assert not list((home / "Library" / "LaunchAgents").glob("com.brainlayer.*.plist"))


def test_launchd_installer_renders_brainlayer_python_override(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)
    brainlayer_python = tmp_path / "tool" / "bin" / "python"

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "backup"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": "/usr/bin/python3",
            "BRAINLAYER_PYTHON": str(brainlayer_python),
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rendered = home / "Library" / "LaunchAgents" / "com.brainlayer.backup-daily.plist"
    assert f"<string>{brainlayer_python}</string>" in rendered.read_text(encoding="utf-8")
    assert "__BRAINLAYER_PYTHON__" not in rendered.read_text(encoding="utf-8")


def test_launchd_installer_renders_launchd_dir_for_maintenance_resume(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "maintenance"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_LAUNCHD_DIR": str(launchd_dir),
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rendered = home / "Library" / "LaunchAgents" / "com.brainlayer.maintenance-nightly.plist"
    content = rendered.read_text(encoding="utf-8")
    assert f"<string>{launchd_dir}</string>" in content
    assert "__BRAINLAYER_LAUNCHD_DIR__" not in content


def test_launchd_installer_renders_homebrew_opt_symlink_instead_of_cellar_version(tmp_path: Path) -> None:
    fake_homebrew = tmp_path / "opt" / "homebrew"
    cellar_root = fake_homebrew / "Cellar" / "brainlayer" / "9.9.9"
    opt_root = fake_homebrew / "opt" / "brainlayer"
    launchd_dir = cellar_root / "libexec" / "lib" / "python3.12" / "site-packages" / "brainlayer" / "launchd"
    shutil.copytree(REPO_ROOT / "scripts" / "launchd", launchd_dir)
    opt_root.parent.mkdir(parents=True)
    opt_root.symlink_to(cellar_root)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    brainlayer_bin = cellar_root / "libexec" / "bin" / "brainlayer"
    brainlayer_bin.parent.mkdir(parents=True)
    brainlayer_bin.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    brainlayer_bin.chmod(0o755)
    python_bin = cellar_root / "libexec" / "bin" / "python3"
    python_bin.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    python_bin.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "maintenance-nightly"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": str(brainlayer_bin),
            "PYTHON_BIN": str(python_bin),
            "BRAINLAYER_PYTHON": str(python_bin),
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rendered = home / "Library" / "LaunchAgents" / "com.brainlayer.maintenance-nightly.plist"
    content = rendered.read_text(encoding="utf-8")
    assert f"{fake_homebrew}/Cellar/brainlayer/9.9.9" not in content
    assert f"<string>{opt_root}/libexec/bin/python3</string>" in content
    assert f"<string>{opt_root}/libexec/lib/python3.12/site-packages</string>" in content
    assert f"<string>{opt_root}/libexec/lib/python3.12/site-packages/brainlayer/launchd</string>" in content


def test_launchd_installer_attempts_remaining_services_after_bootstrap_error(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
                'if [ "$1" = "bootstrap" ] && [[ "$3" == *"maintenance-nightly.plist" ]]; then',
                '  printf "%s\\n" "bootstrap failed" >&2',
                "  exit 5",
                "fi",
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "maintenance"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert result.returncode != 0
    assert any(
        command.startswith("bootstrap ") and command.endswith("com.brainlayer.maintenance-nightly.plist")
        for command in commands
    )
    assert any(
        command.startswith("bootstrap ") and command.endswith("com.brainlayer.maintenance-weekly.plist")
        for command in commands
    )


def test_launchd_installer_does_not_load_services_when_env_runner_install_fails(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "launchd"
    shutil.copytree(REPO_ROOT / "scripts" / "launchd", launchd_dir)
    (launchd_dir / "brainlayer-env-run.sh").unlink()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "maintenance"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "brainlayer-env-run.sh" in result.stdout
    assert not launchctl_log.exists()
    assert not list((home / "Library" / "LaunchAgents").glob("com.brainlayer.*.plist"))


def test_launchd_installer_does_not_load_service_when_env_runner_copy_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_install = fake_bin / "install"
    fake_install.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "install failed" >&2',
                "exit 13",
            ]
        ),
        encoding="utf-8",
    )
    fake_install.chmod(0o755)
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "maintenance-nightly"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "install failed" in result.stderr
    assert not launchctl_log.exists()


def test_launchd_installer_does_not_load_service_when_plist_render_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sed = fake_bin / "sed"
    fake_sed.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "sed failed" >&2',
                "exit 7",
            ]
        ),
        encoding="utf-8",
    )
    fake_sed.chmod(0o755)
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "maintenance-nightly"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "sed failed" in result.stderr
    assert not launchctl_log.exists()


def test_launchd_enable_missing_service_is_retried_after_bootstrap(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    bootstrap_state = tmp_path / "bootstrapped"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
                'if [ "$1" = "enable" ] && [ ! -f "$FAKE_BOOTSTRAPPED" ]; then',
                '  printf "%s\\n" "Could not find service" >&2',
                "  exit 113",
                "fi",
                'if [ "$1" = "bootstrap" ]; then',
                '  touch "$FAKE_BOOTSTRAPPED"',
                "fi",
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "maintenance-nightly"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
            "FAKE_BOOTSTRAPPED": str(bootstrap_state),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert result.returncode == 0, result.stdout + result.stderr
    assert [command.split()[0] for command in commands] == ["bootout", "enable", "bootstrap", "enable", "print"]


def test_launchd_all_does_not_load_backup_jobs_when_wrapper_render_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sed = fake_bin / "sed"
    fake_sed.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'last="${@: -1}"',
                'case "$last" in',
                '  */backup-daily.sh|*/jsonl-backup.sh) printf "%s\\n" "wrapper render failed" >&2; exit 7 ;;',
                "esac",
                'exec /usr/bin/sed "$@"',
            ]
        ),
        encoding="utf-8",
    )
    fake_sed.chmod(0o755)
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    _write_full_launchd_env(env_file)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "all"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert result.returncode != 0
    assert "wrapper render failed" in result.stderr
    assert not any(command.startswith("bootstrap ") and "backup-daily.plist" in command for command in commands)
    assert not any(command.startswith("bootstrap ") and "jsonl-backup.plist" in command for command in commands)


def test_launchd_all_preserves_legacy_enrich_when_replacement_batch_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
                'if [ "$1" = "bootstrap" ] && [[ "$3" == *"com.brainlayer.enrichment.plist" ]]; then',
                '  printf "%s\\n" "replacement bootstrap failed" >&2',
                "  exit 5",
                "fi",
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    _write_full_launchd_env(env_file)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "all"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert result.returncode != 0
    assert "replacement bootstrap failed" in result.stderr
    assert not any(command.startswith("unload ") and "com.brainlayer.enrich.plist" in command for command in commands)


def test_wheel_contains_cli_and_launchd_templates(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "dist"
    pip_available = (
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        == 0
    )
    build_command = (
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-cache-dir",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(REPO_ROOT),
        ]
        if pip_available
        else ["uv", "build", "--no-cache", "--wheel", "--out-dir", str(wheel_dir)]
    )

    assert pip_available or shutil.which("uv"), "wheel build test requires either pip or uv"
    result = subprocess.run(
        build_command,
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
