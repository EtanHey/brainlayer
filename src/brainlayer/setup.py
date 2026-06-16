"""Installable setup helpers for BrainLayer."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

from .cli.wizard import DEFAULT_BRAINLAYER_CONFIG, write_gemini_env_file

DEFAULT_GOOGLE_API_KEY_OP_REF = "op://Private/Google AI/Gemini API key"


def get_default_env_file() -> Path:
    """Return the standard per-user BrainLayer env file."""
    return Path.home() / ".config" / "brainlayer" / "brainlayer.env"


def get_launchd_dir() -> Path:
    """Return the packaged launchd template directory, falling back to source checkout."""
    packaged = resources.files("brainlayer").joinpath("launchd")
    if packaged.is_dir():
        return Path(str(packaged))

    source = Path(__file__).resolve().parents[2] / "scripts" / "launchd"
    if source.is_dir():
        return source

    raise FileNotFoundError("BrainLayer launchd templates were not found")


def ensure_brainlayer_env(
    env_file: Path | None = None,
    *,
    google_api_key_op_ref: str | None = None,
    overwrite_google_key: bool = False,
) -> Path:
    """Create or update brainlayer.env with defaults and an op-backed Google key."""
    target = env_file or get_default_env_file()

    if google_api_key_op_ref:
        write_gemini_env_file(
            target,
            google_api_key=google_api_key_op_ref,
            secret_source="1password",
            overwrite=overwrite_google_key,
            enrichment_env=DEFAULT_BRAINLAYER_CONFIG,
        )
        return target

    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BrainLayer private config.",
        "# Add a 1Password-backed key before enabling cloud enrichment with:",
        f"# brainlayer setup --google-api-key-op-ref '{DEFAULT_GOOGLE_API_KEY_OP_REF}'",
        "",
    ]
    lines.extend(f"{key}={value}" for key, value in DEFAULT_BRAINLAYER_CONFIG.items())
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.chmod(0o600)
    return target


def install_launchd(
    target: str = "all",
    *,
    env_file: Path | None = None,
    launchd_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
    timeout_seconds: float = 120,
) -> None:
    """Run the packaged launchd installer for a single target or all agents."""
    template_dir = launchd_dir or get_launchd_dir()
    install_script = template_dir / "install.sh"
    if not install_script.exists():
        raise FileNotFoundError(f"launchd installer not found: {install_script}")

    run_env = os.environ.copy()
    if extra_env:
        run_env.update(extra_env)
    run_env.setdefault("PYTHON_BIN", sys.executable)
    run_env.setdefault("BRAINLAYER_PYTHON", sys.executable)
    if env_file is not None:
        run_env["BRAINLAYER_ENV_FILE"] = str(env_file)

    try:
        subprocess.run([str(install_script), target], env=run_env, check=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"launchd installer timed out after {timeout_seconds:g}s") from exc
