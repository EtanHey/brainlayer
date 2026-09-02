"""``brainlayer.__build_sha__`` — stamped by the release build, ``None`` in a source tree (#749 keg mode)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_PY = REPO_ROOT / "src" / "brainlayer" / "__init__.py"


def _import_build_sha(tmp_path: Path, stamp: str | None) -> str:
    # Import a bare copy of the real __init__.py so a stale local _build.py cannot leak in. No build, no network.
    package = tmp_path / "brainlayer"
    package.mkdir()
    shutil.copy2(INIT_PY, package / "__init__.py")
    if stamp is not None:
        (package / "_build.py").write_text(f'BUILD_SHA = "{stamp}"\n', encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "PYTHONSAFEPATH": "1"}
    return subprocess.run(
        [sys.executable, "-c", "import brainlayer; print(repr(brainlayer.__build_sha__))"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        cwd=tmp_path,
    ).stdout.strip()


def test_build_sha_is_none_without_stamp(tmp_path: Path) -> None:
    assert _import_build_sha(tmp_path, None) == "None"


def test_build_sha_reads_stamp_module(tmp_path: Path) -> None:
    assert _import_build_sha(tmp_path, "a" * 40) == repr("a" * 40)


def test_stamp_module_is_gitignored_but_shipped() -> None:
    rc = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q", "src/brainlayer/_build.py"]).returncode
    assert rc == 0, "src/brainlayer/_build.py must be gitignored (never committed)"
    assert '"src/brainlayer/_build.py"' in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"), (
        "hatchling drops VCS-ignored files unless pyproject lists the stamp under [tool.hatch.build] artifacts"
    )
