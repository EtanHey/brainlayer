#!/usr/bin/env python3
"""Select PR sweeps and require a full exact-SHA batch receipt before release."""

import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tomllib
from functools import lru_cache
from importlib.machinery import PathFinder
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = {
    "pyproject.toml",
    "tests/conftest.py",
    "scripts/run_tests.sh",
    ".github/workflows/ci.yml",
    "scripts/ci/crash_sweep_gate.py",
    "tests/test_ci_crash_sweeps.py",
    "tests/test_crash_sweep_gate.py",
    "tests/test_crash_sweep_dependencies.py",
}


def import_roots(sources, config):
    """Use the repo root, src layout and declared pytest/conftest Python paths."""
    configured = config.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("pythonpath", [])
    roots = {".", "src", *(configured.split() if isinstance(configured, str) else configured)}
    for path, text in sources.items():
        if not path.endswith("conftest.py"):
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            target = ast.unparse(node.func)
            if target in {"sys.path.insert", "sys.path.append"} or target.endswith(".syspath_prepend"):
                value = node.args[1 if target == "sys.path.insert" else 0]
                if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                    raise RuntimeError(f"Unresolved import root in {path}: {ast.unparse(value)}")
                roots.add(value.value)
    if any(Path(value).is_absolute() or ".." in Path(value).parts for value in roots):
        raise RuntimeError("Unresolved import root outside repository")
    return sorted(str(Path(value)) for value in roots)


@lru_cache(maxsize=None)
def external_module(name):
    """Resolve external specs without importing or executing third-party package initializers."""
    if name in sys.stdlib_module_names or sys.modules.get(name) is not None:
        return True
    search = sys.path
    for index, part in enumerate(name.split(".")):
        prefix = ".".join(name.split(".")[: index + 1])
        spec = PathFinder.find_spec(prefix, search)
        if spec is None:
            return False
        search = spec.submodule_search_locations
        if search is None and prefix != name:
            return False
    return True


def import_closure(sources, roots=(".", "src"), declared=()):
    """Walk local imports without executing code, including literal child-process probes."""
    modules = {}
    for path in sources:
        for root in roots:
            if root == "." or path.startswith(root + "/"):
                relative = path if root == "." else path[len(root) + 1 :]
                module = relative.removesuffix(".py").replace("/", ".")
                modules[module.removesuffix(".__init__")] = path
    namespaces = {name.rsplit(".", 1)[0] for name in modules if "." in name}
    namespaces |= {".".join(name.split(".")[:index]) for name in modules for index in range(1, len(name.split(".")))}

    local_tops = {name.split(".")[0] for name in modules}

    def resolve(name, path):
        if name not in modules and name not in namespaces:
            third_party = any(name == prefix or name.startswith(prefix + ".") for prefix in declared)
            if name.split(".")[0] in local_tops or not (third_party or external_module(name)):
                raise RuntimeError(f"Unresolved import {name!r} in {path}")
        return name

    # Parse non-test packages lazily: unrelated scripts need not be executable/importable test code.
    trees = {path: ast.parse(text, filename=path) for path, text in sources.items() if path.startswith("tests/")}
    pending = [
        path
        for path, tree in trees.items()
        if path.startswith("tests/")
        and any(isinstance(node, ast.Attribute) and node.attr == "crash_sweep" for node in ast.walk(tree))
    ]
    if not pending:
        raise RuntimeError("No crash_sweep-marked test files found")
    reached = set()
    while pending:
        path = pending.pop()
        if path in reached:
            continue
        reached.add(path)
        module = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
        package = module.removesuffix(".__init__") if path.endswith("/__init__.py") else module.rpartition(".")[0]
        queue, literals = [trees[path] if path in trees else ast.parse(sources[path], filename=path)], set()
        while queue:
            for node in ast.walk(queue.pop()):
                names = []
                if isinstance(node, ast.Import):
                    names = [resolve(alias.name, path) for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    prefix = node.module or ""
                    if node.level:
                        parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                        prefix = ".".join(parent + ([prefix] if prefix else []))
                    names = [resolve(prefix, path)]
                    for alias in node.names:
                        child = prefix + "." + alias.name
                        if child in modules or child in namespaces:
                            names.append(child)
                        elif prefix in namespaces and prefix not in modules and alias.name != "*":
                            names.append(resolve(child, path))
                elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                    target = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    if target in {"import_module", "__import__"} and isinstance(node.args[0].value, str):
                        names = [resolve(node.args[0].value, path)]
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value not in literals:
                    literals.add(node.value)
                    if node.value in modules:
                        names.append(node.value)  # Includes literal argv ["-m", "local.module"].
                    # AIDEV-NOTE: path-launched files (supervisor_guard -> supervisor_child) are
                    # outside this import closure; these are module edges, not arbitrary file paths.
                    if "import " in node.value:
                        try:
                            probe = ast.parse(node.value)
                            if any(
                                (
                                    isinstance(item, ast.Import)
                                    and any(alias.name.split(".")[0] in local_tops for alias in item.names)
                                )
                                or (
                                    isinstance(item, ast.ImportFrom)
                                    and (item.level or (item.module or "").split(".")[0] in local_tops)
                                )
                                for item in ast.walk(probe)
                            ):
                                queue.append(probe)
                        except SyntaxError:
                            pass  # Ordinary text strings are not executable Python probes.
                for name in names:
                    while name:
                        if name in modules:
                            pending.append(modules[name])
                        name = name.rpartition(".")[0]  # Python executes package initializers too.
    return reached | HARNESS


@lru_cache(maxsize=8)
def sweep_dependencies(root=ROOT, revision=None):
    """Derive imports in the checkout or an immutable base snapshot (for deleted/renamed imports)."""
    if revision is None:
        sources = {}
        for directory, folders, files in os.walk(root):
            folders[:] = [
                name
                for name in folders
                if not name.startswith(".")
                and name
                not in {
                    "node_modules",
                    "venv",
                    "env",
                    "build",
                    "dist",
                    "__pycache__",
                    "docs.local",
                }
            ]
            for name in files:
                if name.endswith(".py"):
                    path = Path(directory) / name
                    sources[path.relative_to(root).as_posix()] = path.read_text()
        metadata = (root / "pyproject.toml").read_text() if (root / "pyproject.toml").is_file() else ""
    else:
        archive = subprocess.check_output(["git", "archive", revision, "*.py", "pyproject.toml"], cwd=root, timeout=60)
        with tarfile.open(fileobj=io.BytesIO(archive)) as snapshot:
            contents = {
                member.name: snapshot.extractfile(member).read().decode() for member in snapshot if member.isfile()
            }
        metadata = contents.pop("pyproject.toml", "")
        sources = {name: text for name, text in contents.items() if name.endswith(".py")}
    config = tomllib.loads(metadata)
    roots = import_roots(sources, config)
    if revision is None:
        # Explicit roots may be hidden directories pruned during the general repository walk.
        for local_root in roots:
            if local_root not in {".", "src"}:
                for path in (root / local_root).rglob("*.py"):
                    sources[path.relative_to(root).as_posix()] = path.read_text()
    # Generated local modules have explicit build declarations, e.g. the optional _build.py stamp.
    for artifact in config.get("tool", {}).get("hatch", {}).get("build", {}).get("artifacts", []):
        if artifact.endswith(".py") and not any(char in artifact for char in "*?["):
            sources.setdefault(artifact, "")
    # Optional dependency imports are identifiable even in the cheap CI selector's minimal environment.
    project = config.get("project", {})
    requirements = [
        *project.get("dependencies", []),
        *[value for group in project.get("optional-dependencies", {}).values() for value in group],
    ]
    aliases = {
        "pyyaml": ("yaml",),
        "scikit-learn": ("sklearn",),
        # sentence-transformers declares torch; the selector must not install/load model runtimes.
        "sentence-transformers": ("sentence_transformers", "torch"),
    }
    declared = set()
    for requirement in requirements:
        distribution = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement.strip()).group().lower().replace("_", "-")
        declared.update(aliases.get(distribution, (distribution.replace("-", "_"),)))
    return import_closure(sources, roots, declared)


def required(event, files, labels, root=ROOT, dependencies=None):
    if event != "pull_request" or "crash-sweep" in labels:
        return True
    dependencies = sweep_dependencies(root) if dependencies is None else dependencies
    return any(path in dependencies or path.startswith(("scripts/ci/", "scripts/launchd/")) for path in files)


def api(endpoint):
    result = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", endpoint],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def verify_release(repo, sha):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError("release requires a full commit SHA")
    pages = api(f"repos/{repo}/actions/workflows/ci.yml/runs?head_sha={sha}&status=success&per_page=100")
    for page in pages:
        for run in page["workflow_runs"]:
            if not (
                run["head_sha"] == sha
                and run["conclusion"] == "success"
                and (run["event"] == "workflow_dispatch" or (run["event"] == "push" and run["head_branch"] == "main"))
            ):
                continue
            jobs = api(f"repos/{repo}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100")
            for job_page in jobs:
                for job in job_page["jobs"]:
                    if (
                        job["name"] == "crash sweeps"
                        and job["head_sha"] == sha
                        and job["conclusion"] == "success"
                        and any(
                            step["name"] == "Exhaustive crash sweeps" and step["conclusion"] == "success"
                            for step in job.get("steps", [])
                        )
                    ):
                        print(f"Full crash sweep passed at {sha}: {job['html_url']}")
                        return
    raise RuntimeError(f"No successful full crash sweep for exact batch SHA {sha}; run CI on that ref before release")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("required")
    release = commands.add_parser("verify-release")
    release.add_argument("repo")
    release.add_argument("sha")
    release.add_argument("--tag", action="store_true", help="peel a local tag object to its commit")
    args = parser.parse_args()
    if args.command == "verify-release":
        sha = (
            subprocess.check_output(["git", "rev-parse", f"{args.sha}^{{commit}}"], text=True, timeout=60).strip()
            if args.tag
            else args.sha
        )
        verify_release(args.repo, sha)
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    files, labels, dependencies = [], [], None
    if os.environ["GITHUB_EVENT_NAME"] == "pull_request":
        pr = event["pull_request"]
        files = subprocess.check_output(
            ["git", "diff", "--name-only", "-z", "--no-renames", f"{pr['base']['sha']}...{pr['head']['sha']}", "--"],
            text=True,
            timeout=60,
        ).split("\0")
        labels = [label["name"] for label in pr["labels"]]
        dependencies = sweep_dependencies(Path.cwd()) | sweep_dependencies(Path.cwd(), pr["base"]["sha"])
    value = str(required(os.environ["GITHUB_EVENT_NAME"], files, labels, dependencies=dependencies)).lower()
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"required={value}\n")
    print(f"Full crash sweep required={value}")


if __name__ == "__main__":
    main()
