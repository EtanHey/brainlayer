"""Fresh outcome evidence for health-check tests: what a healthy Mac's state files look like."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


def backup_receipt(created: datetime, **extra: object) -> dict:
    return {
        "attempted_at": created.isoformat(),
        "backup_log_provenance": "real",
        "snapshot": f"/backups/{created.date().isoformat()}.db.zst",
        "snapshot_created_at": created.isoformat(),
        "uploaded": True,
        "verified": True,
        "local_verified": True,
        "local_integrity": "ok",
        **extra,
    }


def write_evidence(directory: Path, now: datetime, *, backup_age: timedelta = timedelta(hours=1)) -> dict[str, Path]:
    """Verified local+Drive backup, verified transcript archive, Drive token, live watcher, store round trip.

    Returns the HealthCheckConfig path overrides that point at them.
    """
    paths = {
        "backup_log_path": directory / "backup-daily.log",
        "jsonl_backup_log_path": directory / "jsonl-backup.log",
        "drive_token_path": directory / "drive-tokens.json",
        "watcher_health_path": directory / "watcher-health.json",
        "store_roundtrip_path": directory / "store-roundtrip.json",
        "backup_staging_dir": directory / "backups",
    }
    paths["backup_staging_dir"].mkdir(exist_ok=True)
    paths["backup_log_path"].write_text(json.dumps(backup_receipt(now - backup_age)) + "\n")
    if not paths["jsonl_backup_log_path"].exists():
        attempt = {"attempted_at": (now - timedelta(hours=1)).isoformat(), "status": "no-op", "verified": True}
        paths["jsonl_backup_log_path"].write_text(json.dumps(attempt) + "\n")
    paths["drive_token_path"].write_text(
        json.dumps({"refresh_token": "fixture", "consented_at": (now - timedelta(days=1)).isoformat()})
    )
    paths["watcher_health_path"].write_text(json.dumps({"updated_at": now.isoformat(), "poll_count": 1}))
    paths["store_roundtrip_path"].write_text(
        json.dumps({"checked_at": (now - timedelta(hours=1)).isoformat(), "ok": True, "reason": "vector_retrieved"})
    )
    return paths
