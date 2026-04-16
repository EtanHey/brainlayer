#!/usr/bin/env python3
"""Poll Gemini Batch progress for BrainLayer re-enrichment and log checkpoints."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import brainlayer.cloud_backfill as cloud_backfill
from brainlayer.vector_store import VectorStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _checkpoint_counts(db_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    conn = cloud_backfill._open_checkpoint_conn(db_path)
    try:
        cloud_backfill._ensure_checkpoint_table_in_conn(conn)
        rows = conn.cursor().execute(
            f"""
            SELECT status, COUNT(*)
            FROM {cloud_backfill.CHECKPOINT_TABLE}
            GROUP BY status
            """
        )
        counts = {status: count for status, count in rows}
    finally:
        conn.close()
    return counts


def _preview_stats(db_path: Path) -> dict[str, float]:
    store = VectorStore(db_path)
    try:
        return cloud_backfill.get_reenrichment_stats(store)
    finally:
        store.close()


def _append_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _notify_completion(message: str) -> None:
    notify_cmd = shutil.which("notify")
    if not notify_cmd:
        print(f"{_utc_now()} notify command not found; completion message: {message}")
        return

    try:
        subprocess.run([notify_cmd, message], check=False)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"{_utc_now()} notify failed: {exc}")


def monitor_loop(
    db_path: Path,
    *,
    interval_seconds: int,
    log_path: Path,
    once: bool,
    notify: bool,
) -> int:
    while True:
        summary = cloud_backfill.process_pending_jobs_once(db_path)
        checkpoint_counts = _checkpoint_counts(db_path)
        preview = _preview_stats(db_path)
        line = (
            f"{_utc_now()} checked={summary['checked']} imported_jobs={summary['imported_jobs']} "
            f"failed_jobs={summary['failed_jobs']} pending_jobs={summary['still_pending']} "
            f"chunks_ok={summary['success']} chunks_failed={summary['failed']} "
            f"chunks_skipped={summary['skipped']} previewed={preview['previewed']}/{preview['eligible']} "
            f"remaining={preview['remaining']} statuses={checkpoint_counts}"
        )
        print(line)
        _append_log(log_path, line)

        if summary["checked"] == 0 and checkpoint_counts.get("submitted", 0) == 0:
            message = (
                "BrainLayer batch re-enrichment complete: "
                f"previewed {preview['previewed']}/{preview['eligible']}, "
                f"remaining {preview['remaining']}, failed jobs {checkpoint_counts.get('failed', 0)}"
            )
            print(f"{_utc_now()} {message}")
            _append_log(log_path, f"{_utc_now()} {message}")
            if notify:
                _notify_completion(message)
            return 0

        if once:
            return 0

        time.sleep(interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor BrainLayer Gemini Batch re-enrichment progress")
    parser.add_argument("--db", type=Path, default=cloud_backfill.DEFAULT_DB_PATH, help="BrainLayer DB path")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=1800,
        help="Polling interval in seconds (default: 1800 / 30 minutes)",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("/tmp/brainlayer-batch-monitor.log"),
        help="Append-only progress log path",
    )
    parser.add_argument("--once", action="store_true", help="Run a single poll/import pass and exit")
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Do not invoke the local notify command on completion",
    )
    args = parser.parse_args()

    return monitor_loop(
        args.db,
        interval_seconds=args.interval_seconds,
        log_path=args.log_path,
        once=args.once,
        notify=not args.no_notify,
    )


if __name__ == "__main__":
    raise SystemExit(main())
