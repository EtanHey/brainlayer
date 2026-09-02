"""``brainlayer.__build_sha__`` — stamped by the release build, ``None`` in a source tree (#749 keg mode)."""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_PY = REPO_ROOT / "src" / "brainlayer" / "__init__.py"
STAMP_REL = "src/brainlayer/_build.py"


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


def _pattern_hits_stamp(pattern: str) -> bool:
    pattern = pattern.rstrip("/")
    return (
        fnmatch.fnmatch(STAMP_REL, pattern)
        or fnmatch.fnmatch(STAMP_REL, f"{pattern}/**")
        or STAMP_REL.startswith(f"{pattern}/")
    )


def _hatch_build_config() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["hatch"]["build"]


def test_stamp_module_is_gitignored_but_shipped() -> None:
    # hatchling drops VCS-ignored files from a target unless `artifacts` names them for that target;
    # `[tool.hatch.build] artifacts` is the global default a target inherits unless it sets its own.
    rc = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q", STAMP_REL]).returncode
    assert rc == 0, f"{STAMP_REL} must be gitignored (never committed)"

    build = _hatch_build_config()
    targets = build.get("targets", {})
    assert STAMP_REL in build.get("artifacts", []), (
        f"[tool.hatch.build] artifacts must list {STAMP_REL} or hatchling drops the gitignored stamp"
    )

    for target_name in ("wheel", "sdist"):
        target = targets.get(target_name, {})
        if "artifacts" in target:
            assert STAMP_REL in target["artifacts"], (
                f"[tool.hatch.build.targets.{target_name}] overrides artifacts without {STAMP_REL}, "
                "so the stamp is dropped from that target"
            )
        hit_by = [pattern for pattern in target.get("exclude", []) if _pattern_hits_stamp(pattern)]
        assert hit_by == [], f"{target_name} exclude patterns {hit_by} drop {STAMP_REL}"

    # The wheel carries src/brainlayer as the brainlayer package, so the stamp lands at brainlayer/_build.py.
    wheel_packages = targets.get("wheel", {}).get("packages", [])
    assert "src/brainlayer" in wheel_packages, (
        f"wheel packages {wheel_packages} must carry src/brainlayer for {STAMP_REL} to ship as brainlayer/_build.py"
    )

    # The sdist is only-include based: an included path must cover the stamp.
    sdist_only_include = targets.get("sdist", {}).get("only-include", [])
    covered = [
        entry for entry in sdist_only_include if STAMP_REL == entry or STAMP_REL.startswith(f"{entry.rstrip('/')}/")
    ]
    assert covered, f"sdist only-include {sdist_only_include} does not cover {STAMP_REL}"
