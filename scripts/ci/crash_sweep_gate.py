#!/usr/bin/env python3
"""Select PR sweeps and require a full exact-SHA batch receipt before release."""

import argparse
import ast
import io
import json
import os
import re
import subprocess
import tarfile
from functools import lru_cache
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


def import_closure(sources):
    """Walk local imports without executing code, including literal child-process probes."""
    modules = {}
    for path in sources:
        module = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
        modules[module.removesuffix(".__init__")] = path
    trees = {path: ast.parse(text, filename=path) for path, text in sources.items()}
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
        queue, literals = [trees[path]], set()
        while queue:
            for node in ast.walk(queue.pop()):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    prefix = node.module or ""
                    if node.level:
                        parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                        prefix = ".".join(parent + ([prefix] if prefix else []))
                    names = [prefix, *[prefix + "." + alias.name for alias in node.names]]
                elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                    target = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    if target in {"import_module", "__import__"} and isinstance(node.args[0].value, str):
                        names = [node.args[0].value]
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value not in literals:
                    literals.add(node.value)
                    if "import " in node.value and ("brainlayer" in node.value or "tests." in node.value):
                        try:
                            queue.append(ast.parse(node.value))
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
        sources = {
            path.relative_to(root).as_posix(): path.read_text()
            for folder in ("src/brainlayer", "tests")
            for path in (root / folder).rglob("*.py")
        }
    else:
        archive = subprocess.check_output(["git", "archive", revision, "src/brainlayer", "tests"], cwd=root, timeout=60)
        with tarfile.open(fileobj=io.BytesIO(archive)) as snapshot:
            sources = {
                member.name: snapshot.extractfile(member).read().decode()
                for member in snapshot
                if member.isfile() and member.name.endswith(".py")
            }
    return import_closure(sources)


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
            ["git", "diff", "--name-only", "--no-renames", f"{pr['base']['sha']}...{pr['head']['sha']}", "--"],
            text=True,
            timeout=60,
        ).splitlines()
        labels = [label["name"] for label in pr["labels"]]
        dependencies = sweep_dependencies(Path.cwd()) | sweep_dependencies(Path.cwd(), pr["base"]["sha"])
    value = str(required(os.environ["GITHUB_EVENT_NAME"], files, labels, dependencies=dependencies)).lower()
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"required={value}\n")
    print(f"Full crash sweep required={value}")


if __name__ == "__main__":
    main()
