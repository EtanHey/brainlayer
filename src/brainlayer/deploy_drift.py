"""Deploy provenance helpers for long-running BrainLayer daemons."""

from __future__ import annotations

import json
import os
import plistlib
import subprocess
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
    repo_root: str
    launch_commit: str
    deployed_commit: str
    provenance_path: str
    drift_status: str

    def to_context(self) -> dict[str, str]:
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


def detect_deploy_drift(label: str, provenance_dir: Path) -> DeployDriftFinding | None:
    provenance_path = provenance_path_for_label(provenance_dir, label)
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    repo_root_value = payload.get("repo_root")
    launch_commit = payload.get("launch_commit")
    if not isinstance(repo_root_value, str) or not isinstance(launch_commit, str):
        return None
    repo_root = Path(repo_root_value).expanduser()
    deployed_commit = git_head(repo_root)
    if not deployed_commit or deployed_commit == launch_commit:
        return None
    drift_status = "older" if commit_is_ancestor(repo_root, launch_commit, deployed_commit) else "diverged"
    return DeployDriftFinding(
        label=label,
        repo_root=str(repo_root),
        launch_commit=launch_commit,
        deployed_commit=deployed_commit,
        provenance_path=str(provenance_path),
        drift_status=drift_status,
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
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    resolved_repo = repo_root or _repo_root_from_env_or_cwd()
    launch_commit = git_head(resolved_repo) if resolved_repo is not None else None
    path = provenance_path_for_label(provenance_dir or default_deploy_provenance_dir(), label)
    payload: dict[str, object] = {
        "label": label,
        "launched_at": now_fn().isoformat(),
        "pid": os.getpid(),
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
    repo_root = repo_root_from_launchd_plist(plist_path)
    if repo_root is None:
        raise DeployProvenanceError(
            label,
            plist_path,
            f"could not resolve repo root from launchd plist for {label}: {plist_path.expanduser()}",
        )
    return write_daemon_launch_provenance(
        label=label,
        repo_root=repo_root,
        provenance_dir=provenance_dir,
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
