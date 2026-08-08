"""Deploy provenance helpers for long-running BrainLayer daemons."""

from __future__ import annotations

import json
import os
import plistlib
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from .paths import get_db_path

DEFAULT_DEPLOY_DRIFT_LABELS = (
    "com.mcplayer.brainlayer-proxy",
    "com.brainlayer.enrichment",
    "com.brainlayer.drain",
    "com.brainlayer.watch",
)

DEPLOY_DRIFT_IGNORED_EXACT_PATHS = {
    "README.md",
    "src/brainlayer/deploy_drift.py",
    "src/brainlayer/doctor.py",
    "src/brainlayer/reembed_backfill.py",
    "scripts/reembed_backfill_loop.py",
}

DEPLOY_DRIFT_IGNORED_PREFIXES = (
    "docs/",
    "tests/",
)

BRAINLAYER_LABEL_BY_SERVICE = {
    "enrichment": "com.brainlayer.enrichment",
    "drain": "com.brainlayer.drain",
    "watch": "com.brainlayer.watch",
}


class DeployProvenanceError(RuntimeError):
    """Raised when daemon provenance cannot be tied to the launchd plist source."""

    def __init__(self, label: str, plist_path: Path, message: str) -> None:
        super().__init__(message)
        self.label = label
        self.plist_path = plist_path


@dataclass(frozen=True)
class DeployDriftFinding:
    label: str
    repo_root: str | None
    provenance_path: str
    drift_status: str
    identity_kind: str
    launch_commit: str | None = None
    deployed_commit: str | None = None
    launch_version: str | None = None
    deployed_version: str | None = None

    def to_context(self) -> dict[str, str | None]:
        return asdict(self)


def default_deploy_provenance_dir() -> Path:
    return get_db_path().expanduser().parent / "daemon-provenance"


def provenance_path_for_label(provenance_dir: Path, label: str) -> Path:
    return provenance_dir.expanduser() / f"{label}.json"


def _clean_git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git_stdout(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root.expanduser()), *args],
            text=True,
            capture_output=True,
            env=_clean_git_env(),
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_head(repo_root: Path) -> str | None:
    return _git_stdout(repo_root, "rev-parse", "HEAD")


def git_root_for_path(path: Path) -> Path | None:
    candidate = path.expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    root = _git_stdout(candidate, "rev-parse", "--show-toplevel")
    return Path(root) if root else None


def _is_brainlayer_git_checkout(path: Path) -> bool:
    git_root = git_root_for_path(path)
    resolved = path.expanduser().resolve()
    if git_root is None or git_root.resolve() != resolved:
        return False
    package_marker = resolved / "src" / "brainlayer" / "__init__.py"
    try:
        metadata = tomllib.loads((resolved / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    project = metadata.get("project")
    return package_marker.is_file() and isinstance(project, dict) and project.get("name") == "brainlayer"


def _artifact_version() -> str:
    from . import __version__

    return __version__


def commit_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root.expanduser()), "merge-base", "--is-ancestor", ancestor, descendant],
            text=True,
            capture_output=True,
            env=_clean_git_env(),
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def changed_files_between(repo_root: Path, old_commit: str, new_commit: str) -> set[str] | None:
    diff = _git_stdout(repo_root, "diff", "--name-only", f"{old_commit}..{new_commit}")
    if diff is None:
        return None
    return {line.strip() for line in diff.splitlines() if line.strip()}


def deploy_drift_changes_require_redeploy(changed_files: set[str] | None) -> bool:
    if changed_files is None:
        return True
    return any(
        path not in DEPLOY_DRIFT_IGNORED_EXACT_PATHS
        and not any(path.startswith(prefix) for prefix in DEPLOY_DRIFT_IGNORED_PREFIXES)
        for path in changed_files
    )


def detect_deploy_drift(label: str, provenance_dir: Path) -> DeployDriftFinding | None:
    provenance_path = provenance_path_for_label(provenance_dir, label)
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    repo_root_value = payload.get("repo_root")
    repo_root = Path(repo_root_value).expanduser() if isinstance(repo_root_value, str) else None
    launch_commit = payload.get("launch_commit")
    if repo_root is not None and _is_brainlayer_git_checkout(repo_root) and isinstance(launch_commit, str):
        deployed_commit = git_head(repo_root)
        if not deployed_commit or deployed_commit == launch_commit:
            return None
        changed_files = changed_files_between(repo_root, launch_commit, deployed_commit)
        if not deploy_drift_changes_require_redeploy(changed_files):
            return None
        drift_status = "older" if commit_is_ancestor(repo_root, launch_commit, deployed_commit) else "diverged"
        return DeployDriftFinding(
            label=label,
            repo_root=str(repo_root),
            launch_commit=launch_commit,
            deployed_commit=deployed_commit,
            provenance_path=str(provenance_path),
            drift_status=drift_status,
            identity_kind="git_commit",
        )

    launch_version = payload.get("artifact_version")
    deployed_version = _artifact_version()
    if not isinstance(launch_version, str) and repo_root is not None and isinstance(launch_commit, str):
        return DeployDriftFinding(
            label=label,
            repo_root=str(repo_root) if repo_root is not None else None,
            provenance_path=str(provenance_path),
            drift_status="release_identity_missing",
            identity_kind="release_version",
            deployed_version=deployed_version,
        )
    if not isinstance(launch_version, str):
        return None
    if launch_version == deployed_version:
        return None
    return DeployDriftFinding(
        label=label,
        repo_root=str(repo_root) if repo_root is not None else None,
        provenance_path=str(provenance_path),
        drift_status="version_mismatch",
        identity_kind="release_version",
        launch_version=launch_version,
        deployed_version=deployed_version,
    )


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, resolved)


def write_daemon_launch_provenance(
    *,
    label: str,
    repo_root: Path | None = None,
    provenance_dir: Path | None = None,
    auto_detect_repo_root: bool = True,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    resolved_repo = repo_root
    if resolved_repo is None and auto_detect_repo_root:
        resolved_repo = _repo_root_from_env_or_cwd()
    launch_commit = (
        git_head(resolved_repo) if resolved_repo is not None and _is_brainlayer_git_checkout(resolved_repo) else None
    )
    path = provenance_path_for_label(provenance_dir or default_deploy_provenance_dir(), label)
    payload: dict[str, object] = {
        "label": label,
        "launched_at": now_fn().isoformat(),
        "pid": os.getpid(),
        "artifact_version": _artifact_version(),
    }
    if resolved_repo is not None:
        payload["repo_root"] = str(resolved_repo)
    if launch_commit is not None:
        payload["launch_commit"] = launch_commit
    _atomic_write_json(path, payload)
    return path


def record_launch_from_environment() -> Path | None:
    service = os.environ.get("BRAINLAYER_LAUNCHD_SERVICE", "")
    label = BRAINLAYER_LABEL_BY_SERVICE.get(service)
    if not label:
        return None
    return write_daemon_launch_provenance(label=label)


def repo_root_from_launchd_plist(plist_path: Path) -> Path | None:
    try:
        with plist_path.expanduser().open("rb") as handle:
            payload = plistlib.load(handle)
    except (FileNotFoundError, OSError, plistlib.InvalidFileException):
        return None
    if not isinstance(payload, dict):
        return None

    env = payload.get("EnvironmentVariables")
    if isinstance(env, dict):
        repo_root = env.get("BRAINLAYER_REPO_ROOT")
        if isinstance(repo_root, str) and repo_root:
            return Path(repo_root).expanduser()

    working_directory = payload.get("WorkingDirectory")
    if isinstance(working_directory, str) and working_directory:
        root = git_root_for_path(Path(working_directory))
        if root is not None:
            return root

    program_args = payload.get("ProgramArguments")
    if isinstance(program_args, list):
        for item in program_args:
            if not isinstance(item, str) or not item.startswith("/"):
                continue
            root = git_root_for_path(Path(item))
            if root is not None:
                return root
    return None


def record_deploy_provenance_for_label(*, label: str, plist_path: Path, provenance_dir: Path) -> Path:
    try:
        with plist_path.expanduser().open("rb") as handle:
            payload = plistlib.load(handle)
    except (FileNotFoundError, OSError, plistlib.InvalidFileException) as exc:
        raise DeployProvenanceError(label, plist_path, f"could not read launchd plist {plist_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DeployProvenanceError(label, plist_path, f"could not read launchd plist {plist_path}: root is not a dict")
    repo_root = repo_root_from_launchd_plist(plist_path)
    return write_daemon_launch_provenance(
        label=label,
        repo_root=repo_root,
        provenance_dir=provenance_dir,
        auto_detect_repo_root=False,
    )


def brainbar_changed_for_deploy(provenance_dir: Path, *, repo_root: Path | None = None) -> bool:
    resolved_repo = repo_root or git_root_for_path(Path(__file__).resolve())
    if resolved_repo is None:
        return False
    head = git_head(resolved_repo)
    if head is None:
        return False

    previous_commits: list[str] = []
    for path in provenance_dir.expanduser().glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("repo_root") != str(resolved_repo):
            continue
        launch_commit = payload.get("launch_commit")
        if isinstance(launch_commit, str) and launch_commit:
            previous_commits.append(launch_commit)

    if previous_commits:
        changed_files: set[str] = set()
        for commit in previous_commits:
            diff = _git_stdout(resolved_repo, "diff", "--name-only", f"{commit}..{head}") or ""
            changed_files.update(line.strip() for line in diff.splitlines() if line.strip())
    else:
        diff_tree = _git_stdout(resolved_repo, "diff-tree", "--no-commit-id", "--name-only", "-r", head) or ""
        changed_files = {line.strip() for line in diff_tree.splitlines() if line.strip()}

    return any(path == "brain-bar" or path.startswith("brain-bar/") for path in changed_files)


def _repo_root_from_env_or_cwd() -> Path | None:
    env_root = os.environ.get("BRAINLAYER_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return git_root_for_path(Path.cwd())
