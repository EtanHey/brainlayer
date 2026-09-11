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
    return {"version": 1, "path": str(Path(path)), "st_dev": st_dev, "st_ino": st_ino}


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
        roots = [
            jsonl_backup.BackupSourceRoot(
                Path(value["path"]), tuple(value["include_globs"]),
                tuple(value.get("exclude_globs", ())), str(value.get("source_class", "transcript")),
            )
            for value in values
        ]
    except (TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ScheduledBackupRefusal("scheduled transcript roots are malformed") from exc
    if not roots:
        raise ScheduledBackupRefusal("scheduled transcript roots are not configured")
    return roots


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
            previous: dict[str, Any] = {}
            try:
                previous = json.loads(state_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                pass
            except (OSError, json.JSONDecodeError):
                previous = {}
            next_due = previous.get("next_due")
            if isinstance(next_due, str):
                try:
                    if dt.datetime.fromisoformat(next_due) > now:
                        return _status("not_due", next_due=next_due)
                except ValueError:
                    pass
            try:
                binding = _load_binding(config_path)
                source, identity = _require_bound_source(binding)
                destination, destination_identity = backup_run_once._managed_backup_config(config_path)
                result = backup_run_once.run_backup_once(
                    destination=destination,
                    transcript_roots=_roots(config_path),
                    offline_db_path=source,
                    expected_offline_source_identity=identity,
                    expected_destination_identity=destination_identity,
                )
                # run_backup_once checked the saved identity before and after the
                # actual SQLite snapshot. Recheck here only to classify a later
                # path replacement as changed rather than silently refreshing it.
                _require_bound_source(binding)
                receipt = destination / result["generation"] / "receipt.complete.json"
                outcome = _status(
                    "local_verified", receipt=str(receipt), next_due=(now + dt.timedelta(seconds=interval_seconds)).isoformat(),
                    saved_source=binding,
                )
            except ScheduledBackupRefusal as exc:
                outcome = _status("refused", reason=str(exc))
            except Exception as exc:
                outcome = _status("failed", error_type=type(exc).__name__, error=str(exc))
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
        config_path=args.config.expanduser(), state_path=args.state_path.expanduser(), interval_seconds=args.interval_seconds
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"disabled", "not_due", "local_verified", "refused", "busy"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
