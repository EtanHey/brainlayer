"""Durable, offline-only settings backup scheduler.

This process intentionally never contacts the BrainBar socket.  It can run when
the app is absent only after Settings has persisted a connection-observed source
binding.  A missing or changed binding is a visible refusal, never a default DB.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
from typing import Any

from . import backup_run_once, config, jsonl_backup

_BINDING_KEY = "BRAINLAYER_BACKUP_SCHEDULED_OFFLINE_SOURCE_BINDING"
_STATE_VERSION = 1
_FAILURE_BACKOFF_BASE_SECONDS = 60
_FAILURE_BACKOFF_MAX_SECONDS = 3_600
_SCHEDULE_STATUSES = {"disabled", "not_due", "local_verified", "refused", "failed", "busy"}


class ScheduledBackupRefusal(backup_run_once.BackupRunOnceError):
    pass


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_binding(config_path: Path) -> dict[str, Any]:
    raw = config.configured_brainlayer_literal_value(_BINDING_KEY, config_path)
    if not raw:
        raise ScheduledBackupRefusal("scheduled offline source is not configured")
    try:
        binding = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScheduledBackupRefusal("scheduled offline source binding is malformed") from exc
    if not isinstance(binding, dict) or binding.get("version") != 1:
        raise ScheduledBackupRefusal("scheduled offline source binding is unsupported")
    path = binding.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ScheduledBackupRefusal("scheduled offline source binding has no absolute path")
    try:
        st_dev, st_ino = int(binding["st_dev"]), int(binding["st_ino"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScheduledBackupRefusal("scheduled offline source binding has invalid identity") from exc
    source_open_generation = binding.get("source_open_generation")
    if not isinstance(source_open_generation, str) or not source_open_generation.strip():
        raise ScheduledBackupRefusal("scheduled offline source binding has invalid source generation")
    return {
        "version": 1,
        "path": str(Path(path)),
        "st_dev": st_dev,
        "st_ino": st_ino,
        "source_open_generation": source_open_generation,
    }


def _require_bound_source(binding: dict[str, Any]) -> tuple[Path, tuple[int, int]]:
    source = Path(binding["path"])
    try:
        actual = backup_run_once._identity(source)
    except OSError as exc:
        raise ScheduledBackupRefusal("scheduled offline source changed") from exc
    expected = (binding["st_dev"], binding["st_ino"])
    if tuple(actual.values()) != expected:
        raise ScheduledBackupRefusal("scheduled offline source changed")
    return source, expected


def _roots(config_path: Path) -> list[jsonl_backup.BackupSourceRoot]:
    raw = config.configured_brainlayer_literal_value("BRAINLAYER_BACKUP_TRANSCRIPT_ROOT_SPECS", config_path)
    try:
        values = json.loads(raw or "[]")
        if not isinstance(values, list):
            raise TypeError("root specs must be a list")
        roots = [
            jsonl_backup.BackupSourceRoot(
                Path(value["path"]),
                tuple(value["include_globs"]),
                tuple(value.get("exclude_globs", ())),
                str(value.get("source_class", "transcript")),
            )
            for value in values
        ]
    except (TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ScheduledBackupRefusal("scheduled transcript roots are malformed") from exc
    if not roots:
        raise ScheduledBackupRefusal("scheduled transcript roots are not configured")
    return roots


def _validate_timestamp(value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise ScheduledBackupRefusal(f"scheduler state has invalid {field}")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ScheduledBackupRefusal(f"scheduler state has invalid {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ScheduledBackupRefusal(f"scheduler state {field} must be timezone-aware")


def _validate_scope(scope: Any) -> None:
    if not isinstance(scope, dict) or scope.get("version") != 1:
        raise ScheduledBackupRefusal("scheduler state has invalid schedule scope")
    for label in ("source", "destination"):
        identity = scope.get(label)
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise ScheduledBackupRefusal("scheduler state has invalid schedule scope")
        if not Path(identity["path"]).is_absolute():
            raise ScheduledBackupRefusal("scheduler state has invalid schedule scope")
        if any(type(identity.get(key)) is not int for key in ("st_dev", "st_ino")):
            raise ScheduledBackupRefusal("scheduler state has invalid schedule scope")
    roots = scope.get("roots")
    if not isinstance(roots, list):
        raise ScheduledBackupRefusal("scheduler state has invalid schedule scope")
    for root in roots:
        if not isinstance(root, dict) or not isinstance(root.get("path"), str):
            raise ScheduledBackupRefusal("scheduler state has invalid schedule scope")
        if not Path(root["path"]).is_absolute():
            raise ScheduledBackupRefusal("scheduler state has invalid schedule scope")
        if not isinstance(root.get("source_class"), str):
            raise ScheduledBackupRefusal("scheduler state has invalid schedule scope")
        if any(not isinstance(root.get(key), list) for key in ("include_globs", "exclude_globs")):
            raise ScheduledBackupRefusal("scheduler state has invalid schedule scope")
        if any(not isinstance(pattern, str) for key in ("include_globs", "exclude_globs") for pattern in root[key]):
            raise ScheduledBackupRefusal("scheduler state has invalid schedule scope")


def _load_previous_state(state_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ScheduledBackupRefusal("scheduler state is unreadable") from exc
    except json.JSONDecodeError as exc:
        raise ScheduledBackupRefusal("scheduler state is malformed") from exc
    if not isinstance(raw, dict):
        raise ScheduledBackupRefusal("scheduler state is malformed")
    version = raw.get("version")
    if version is not None and (type(version) is not int or version != _STATE_VERSION):
        raise ScheduledBackupRefusal("scheduler state is unsupported")
    status = raw.get("status")
    if status is not None and (not isinstance(status, str) or status not in _SCHEDULE_STATUSES):
        raise ScheduledBackupRefusal("scheduler state has invalid status")
    for field in ("next_due", "retry_after"):
        if field in raw:
            _validate_timestamp(raw[field], field)
    if "failure_count" in raw and (type(raw["failure_count"]) is not int or raw["failure_count"] < 0):
        raise ScheduledBackupRefusal("scheduler state has invalid failure_count")
    if "schedule_scope" in raw:
        _validate_scope(raw["schedule_scope"])
    return raw


def _schedule_scope(
    source: Path,
    source_identity: tuple[int, int],
    destination: Path,
    destination_identity: tuple[int, int],
    roots: list[jsonl_backup.BackupSourceRoot],
) -> dict[str, Any]:
    def path_text(path: Path) -> str:
        return str(path.expanduser().resolve())

    return {
        "version": _STATE_VERSION,
        "source": {"path": path_text(source), "st_dev": source_identity[0], "st_ino": source_identity[1]},
        "destination": {
            "path": path_text(destination),
            "st_dev": destination_identity[0],
            "st_ino": destination_identity[1],
        },
        "roots": [
            {
                "path": path_text(root.path),
                "include_globs": list(root.include_globs),
                "exclude_globs": list(root.exclude_globs),
                "source_class": root.source_class,
            }
            for root in roots
        ],
    }


def _status(status: str, *, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"version": _STATE_VERSION, "status": status}
    if reason is not None:
        result["reason"] = reason
    result.update(extra)
    return result


def run_scheduled_backup(
    *, config_path: Path, state_path: Path, interval_seconds: float = 86_400, now: dt.datetime | None = None
) -> dict[str, Any]:
    now = now or dt.datetime.now(dt.UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        return _status("refused", reason="scheduler clock must be timezone-aware")
    enabled = config.configured_brainlayer_env_value("BRAINLAYER_BACKUP_ENABLED", config_path)
    if enabled not in {"1", "true", "yes", "on"}:
        result = _status("disabled", reason="backup_disabled")
        _write_json(state_path, result)
        return result

    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return _status("busy", reason="scheduler_busy")
        try:
            try:
                previous = _load_previous_state(state_path)
            except ScheduledBackupRefusal as exc:
                # Preserve malformed state for diagnosis; a refusal is visible to
                # the caller but must not turn corrupt bytes into a valid state.
                return _status("refused", reason=str(exc))

            schedule_scope: dict[str, Any] | None = None
            try:
                binding = _load_binding(config_path)
                source, identity = _require_bound_source(binding)
                try:
                    destination, destination_identity = backup_run_once._managed_backup_config(config_path)
                except backup_run_once.BackupRunOnceError as exc:
                    raise ScheduledBackupRefusal(str(exc)) from exc
                try:
                    actual_destination_identity = backup_run_once._identity(destination)
                except OSError as exc:
                    raise ScheduledBackupRefusal("managed backup destination changed") from exc
                if tuple(actual_destination_identity.values()) != destination_identity:
                    raise ScheduledBackupRefusal("managed backup destination changed")
                roots = _roots(config_path)
                try:
                    roots = backup_run_once._validate_roots(roots)
                except backup_run_once.BackupRunOnceError as exc:
                    raise ScheduledBackupRefusal(str(exc)) from exc
                schedule_scope = _schedule_scope(source, identity, destination, destination_identity, roots)
            except ScheduledBackupRefusal as exc:
                outcome = _status("refused", reason=str(exc))
            else:
                # A due time or failure backoff is valid only for the exact
                # source/destination/root scope that produced it. The generation
                # marker remains receipt provenance and is deliberately excluded.
                same_scope = previous.get("schedule_scope") == schedule_scope
                if same_scope:
                    next_due = previous.get("next_due")
                    if isinstance(next_due, str) and dt.datetime.fromisoformat(next_due) > now:
                        extra = {"next_due": next_due}
                        if previous.get("status") == "failed":
                            extra["reason"] = "failure_backoff"
                            extra["failure_count"] = previous.get("failure_count", 0)
                            extra["retry_after"] = previous.get("retry_after", next_due)
                        return _status("not_due", **extra)
                try:
                    result = backup_run_once.run_backup_once(
                        destination=destination,
                        transcript_roots=roots,
                        offline_db_path=source,
                        offline_source_open_generation=binding["source_open_generation"],
                        expected_offline_source_identity=identity,
                        expected_destination_identity=destination_identity,
                    )
                    # run_backup_once checked the saved identity before and after the
                    # actual SQLite snapshot. Recheck here only to classify a later
                    # path replacement as changed rather than silently refreshing it.
                    _require_bound_source(binding)
                    receipt = destination / result["generation"] / "receipt.complete.json"
                    outcome = _status(
                        "local_verified",
                        receipt=str(receipt),
                        next_due=(now + dt.timedelta(seconds=interval_seconds)).isoformat(),
                        saved_source=binding,
                        schedule_scope=schedule_scope,
                    )
                except ScheduledBackupRefusal as exc:
                    outcome = _status("refused", reason=str(exc))
                except Exception as exc:
                    prior_failures = previous.get("failure_count", 0) if same_scope else 0
                    failure_count = prior_failures + 1 if type(prior_failures) is int and prior_failures >= 0 else 1
                    delay = min(
                        _FAILURE_BACKOFF_MAX_SECONDS, _FAILURE_BACKOFF_BASE_SECONDS * (2 ** min(failure_count - 1, 6))
                    )
                    retry_after = now + dt.timedelta(seconds=delay)
                    outcome = _status(
                        "failed",
                        error_type=type(exc).__name__,
                        error=str(exc),
                        failure_count=failure_count,
                        retry_after=retry_after.isoformat(),
                        next_due=retry_after.isoformat(),
                        schedule_scope=schedule_scope,
                    )
            _write_json(state_path, outcome)
            return outcome
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one durable offline BrainLayer backup if due.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=86_400)
    args = parser.parse_args(argv)
    result = run_scheduled_backup(
        config_path=args.config.expanduser(),
        state_path=args.state_path.expanduser(),
        interval_seconds=args.interval_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"disabled", "not_due", "local_verified", "refused", "busy"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
