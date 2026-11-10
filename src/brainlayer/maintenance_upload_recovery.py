"""Select a pending upload without opening the database or creating a snapshot."""

import json
import os
from dataclasses import replace
from pathlib import Path

from . import backup_daily as backup
from .drive_credentials import _atomic_write
from .paths import get_db_path
from .supervisor_jobs import UPLOAD_RECOVERY_TIMEOUT

UPLOAD_TARGET_ENV = "BRAINLAYER_BACKUP_UPLOAD_SNAPSHOT"


def pending_upload_step(plan, directory):
    db = get_db_path()
    log = backup._backup_log_path(None, db_path=db)
    staging = Path(os.environ.get("BRAINLAYER_BACKUP_STAGING_DIR", str(backup.DEFAULT_STAGING_DIR))).resolve()
    machine = backup.resolve_machine_id()
    folder = os.environ.get("BRAINLAYER_BACKUP_DRIVE_FOLDER", os.environ.get("BRAINLAYER_BACKUP_DRIVE_PATH"))
    context = dict(
        db=str(db),
        machine_id=machine,
        drive_folder=folder or "/".join(backup.default_drive_folder_parts(machine)),
        backup_log_provenance=backup._backup_log_provenance(),
    )
    receipts = backup._read_backup_receipts(log, strict=True) if log.exists() else []
    latest = {r["snapshot"]: r for r in receipts if isinstance(r.get("snapshot"), str)}
    state_path = directory / "upload-recovery.json"
    attempts = json.loads(state_path.read_text()) if state_path.exists() else {}
    if not isinstance(attempts, dict) or any(
        not isinstance(entry, dict)
        or set(entry) != {"attempts", "bytes_confirmed"}
        or type(entry["attempts"]) is not int
        or not 0 <= entry["attempts"] <= 2
        or type(entry["bytes_confirmed"]) is not int
        or entry["bytes_confirmed"] < 0
        for entry in attempts.values()
    ):
        raise ValueError("Invalid upload recovery attempts")
    exhausted = False
    for receipt in reversed(list(latest.values())):
        path = Path(receipt["snapshot"])
        artifact = receipt.get("local_artifact")
        if (
            any(receipt.get(k) != v for k, v in context.items())
            or receipt.get("local_verified") is not True
            or (receipt.get("uploaded") is True and receipt.get("verified") is True)
            or receipt.get("local_removed")
            or receipt.get("upload_state") in {"abandoned", "superseded"}
            or path.parent.resolve() != staging
            or path.is_symlink()
            or not path.is_file()
            or not isinstance(artifact, dict)
            or not isinstance(artifact.get("sha256"), str)
            or artifact.get("size") != path.stat().st_size
        ):
            continue
        # Identity is verified again by the upload child under its staging lock. The selector
        # uses the durable verification, never spends minutes hashing on the calendar heartbeat.
        key = json.dumps([str(path.resolve()), artifact["sha256"]])
        committed = receipt.get("bytes_confirmed", 0)
        previous = attempts.get(key, dict(attempts=0, bytes_confirmed=0))
        if (
            type(committed) is not int
            or not 0 <= committed <= path.stat().st_size
            or previous["bytes_confirmed"] > path.stat().st_size
        ):
            raise ValueError("Invalid committed upload bytes")
        # A reservation counts until durable Drive progress proves that attempt advanced.
        # Compare before testing exhaustion so a second advancing attempt can unlock a third.
        count = 0 if committed > previous["bytes_confirmed"] else previous["attempts"]
        if count >= 2:
            exhausted = True
            continue
        step = next(s for s in plan.candidates if s.name == "backup-daily")
        return (
            replace(
                step,
                timeout=UPLOAD_RECOVERY_TIMEOUT,
                env=step.env
                | {
                    "BRAINLAYER_BACKUP_PHASE": "upload",
                    UPLOAD_TARGET_ENV: path.name,
                    "BRAINLAYER_BACKUP_UPLOAD_SHA256": artifact["sha256"],
                    "BRAINLAYER_BACKUP_TIMEOUT_SECONDS": "3600",
                },
            ),
            (state_path, attempts, key, dict(attempts=count + 1, bytes_confirmed=committed)),
            "pending-upload",
        )
    return None, None, "attempts-exhausted" if exhausted else "no-pending-local-snapshot"


def reserve_upload_attempt(reservation):
    path, attempts, key, entry = reservation
    attempts[key] = entry
    _atomic_write(path, attempts)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return entry["attempts"]
