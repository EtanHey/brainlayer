"""Fail-closed import sweep for an installed BrainLayer environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ImportTarget:
    name: str
    kind: str
    value: str

    @classmethod
    def hook(cls, path: Path) -> ImportTarget:
        resolved = path.resolve()
        return cls(name=f"hook:{resolved.name}", kind="hook", value=str(resolved))


@dataclass(frozen=True)
class ImportResult:
    target: ImportTarget
    status: str
    detail: str = ""


def discover_wheel_modules(wheel: Path, package: str = "brainlayer") -> list[ImportTarget]:
    """Return every Python module named by the immutable wheel manifest."""
    prefix = f"{package}/"
    with zipfile.ZipFile(wheel) as archive:
        paths = sorted(name for name in archive.namelist() if name.startswith(prefix) and name.endswith(".py"))
    targets: list[ImportTarget] = []
    seen: set[str] = set()
    for path in paths:
        parts = list(Path(path).relative_to(package).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join((package, *parts))
        if name in seen:
            raise ValueError(f"duplicate module in wheel: {name}")
        seen.add(name)
        targets.append(ImportTarget(name, "module", name))
    if not targets:
        raise ValueError(f"wheel contains no {package} Python modules: {wheel}")
    return targets


def discover_hook_targets(hooks_dir: Path) -> list[ImportTarget]:
    resolved = hooks_dir.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"hooks directory not found: {resolved}")
    return [ImportTarget.hook(path) for path in sorted(resolved.glob("*.py"))]


def stage_hook_targets(hooks_dirs: Sequence[Path], staging_root: Path) -> list[ImportTarget]:
    staging_root.mkdir(parents=True, exist_ok=True)
    staged_targets: list[ImportTarget] = []
    for index, hooks_dir in enumerate(hooks_dirs):
        staged = staging_root / f"hooks-{index}"
        staged.mkdir()
        for target in discover_hook_targets(hooks_dir):
            shutil.copy2(target.value, staged / Path(target.value).name)
        staged_targets.extend(discover_hook_targets(staged))
    return staged_targets


def _child_code(kind: str) -> str:
    guard = (
        "import socket; deny=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network disabled during import')); "
        "socket.create_connection=deny; socket.socket.connect=deny; "
    )
    if kind == "module":
        return guard + (
            "import importlib,pathlib,sys; "
            "module=importlib.import_module(sys.argv[1]); "
            "path=getattr(module,'__file__',None); root=pathlib.Path(sys.argv[2]).resolve(); "
            "resolved=pathlib.Path(path).resolve() if path else None; "
            "assert resolved is not None and resolved.is_relative_to(root), "
            "f'{sys.argv[1]} resolved outside installed artifact root: {resolved} (expected {root})'"
        )
    return guard + (
        "import pathlib,runpy,sys; "
        "sys.path.insert(0,str(pathlib.Path(sys.argv[1]).resolve().parent)); "
        "runpy.run_path(sys.argv[1], run_name='__brainlayer_import_sweep__')"
    )


def _child_env(sandbox: Path) -> dict[str, str]:
    inherited = {"PATH", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    env = {key: value for key, value in os.environ.items() if key in inherited}
    isolated_paths = "HOME XDG_CACHE_HOME MPLCONFIGDIR BRAINLAYER_DB BRAINLAYER_OBSERVABILITY_PATH"
    isolated_paths += " BRAINLAYER_OBSERVABILITY_TRACE_PATH BRAINLAYER_OBSERVABILITY_INPUT_ROOT"
    isolated_paths += " BRAINLAYER_JSONL_BACKUP_LOG_PATH BRAINLAYER_BACKUP_LOG_PATH"
    for key in isolated_paths.split():
        env[key] = str(sandbox / key.lower())
    flags = "BRAINLAYER_FORBID_EMBEDDING_MODEL BRAINLAYER_HOOKS_DISABLED BRAINLAYER_IMPORT_SWEEP"
    flags += " CLAUDE_NON_INTERACTIVE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE PYTHONDONTWRITEBYTECODE"
    env.update(dict.fromkeys(flags.split(), "1"))
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def run_target(
    target: ImportTarget, *, timeout_seconds: float, sandbox_root: Path, package_root: Path | None = None
) -> ImportResult:
    digest = hashlib.sha256(f"{target.kind}:{target.value}".encode()).hexdigest()[:16]
    sandbox = sandbox_root / digest
    sandbox.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-I", "-c", _child_code(target.kind), target.value]
    if target.kind == "module":
        if package_root is None:
            raise ValueError("package_root is required for module imports")
        command.append(str(package_root.resolve()))
    try:
        completed = subprocess.run(
            command,
            cwd=sandbox,
            env=_child_env(sandbox),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ImportResult(target, "timed_out", f"exceeded {timeout_seconds:g}s")
    if completed.returncode == 0:
        return ImportResult(target, "passed")
    detail = (completed.stderr or completed.stdout).strip()
    return ImportResult(target, "failed", detail or f"child exited {completed.returncode}")


def _installed_package_root(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    locations = list(spec.submodule_search_locations or []) if spec is not None else []
    if len(locations) != 1:
        raise RuntimeError(f"expected one installed {package} package root, found {locations}")
    root = Path(locations[0]).resolve()
    if not root.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"{package} resolved outside active environment {sys.prefix}: {root}")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path, help="built wheel whose manifest defines coverage")
    parser.add_argument("--hooks-dir", action="append", type=Path, default=[])
    parser.add_argument("--timeout", type=float, default=60.0, help="per-import timeout in seconds")
    parser.add_argument("--jobs", type=int, default=4, help="maximum concurrent isolated imports")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.jobs <= 0:
        parser.error("--timeout and --jobs must be positive")

    try:
        package_root = _installed_package_root("brainlayer")
        module_targets = discover_wheel_modules(args.wheel)
    except (FileNotFoundError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="brainlayer-import-sweep-") as temporary:
        temp_root = Path(temporary)
        try:
            hook_targets = stage_hook_targets(args.hooks_dir, temp_root)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        targets = [*module_targets, *hook_targets]
        run = lambda target: run_target(  # noqa: E731 - local closure keeps executor inputs explicit
            target, timeout_seconds=args.timeout, sandbox_root=temp_root / "sandboxes", package_root=package_root
        )
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(run, targets))
    unsuccessful = [result for result in results if result.status != "passed"]
    for result in unsuccessful:
        print(f"{result.status.upper()} {result.target.name}\n{result.detail}", file=sys.stderr)
    failed = sum(result.status == "failed" for result in results)
    timed_out = sum(result.status == "timed_out" for result in results)
    print(f"interpreter: {sys.executable}")
    print(f"artifact root: {package_root}")
    print(f"wheel: {args.wheel.resolve()}")
    print(f"inventory: {len(module_targets)} wheel modules, {len(hook_targets)} staged hooks")
    print(f"import sweep: {len(results) - len(unsuccessful)} passed, {failed} failed, {timed_out} timed out")
    return 1 if unsuccessful else 0


if __name__ == "__main__":
    raise SystemExit(main())
