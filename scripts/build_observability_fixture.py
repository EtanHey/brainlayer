#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brainlayer.agent_provenance import PROVENANCE_CLASSES, SOURCE_CLASSES
from brainlayer.vector_store import VectorStore

SEED, GENERATED_AT = 20260913, datetime(2026, 9, 13, 12, tzinfo=UTC)
FIXED_MTIME = int(GENERATED_AT.timestamp())
HELDOUT_SHA256, MANIFEST = "2430cd2eaacb391f71121f2bb1b4a98002b0c062a69e71473478df78b1743098", Path(__file__).resolve().parents[1] / "tests/fixtures/observability/cases.json"  # fmt: skip


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    failure: str
    profile: str = "healthy"

    @property
    def split(self) -> str:
        if self.case_id == "legacy-no-op-dev":
            return "dev"
        return "heldout" if hashlib.sha256(self.case_id.encode()).digest()[0] < 0x60 else "dev"

    @property
    def db_path(self) -> str:
        return f"db/{self.case_id}.sqlite"


def _mtime_iso(*, future: bool = False) -> str:
    return datetime.fromtimestamp(FIXED_MTIME + (4 * 3600 if future else 0), tz=UTC).isoformat().replace("+00:00", "Z")


def case_definitions() -> list[CaseDefinition]:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    result = [CaseDefinition(case["case_id"], case["failure"], case["profile"]) for case in document["cases"]]
    for entry, case in zip(document["cases"], result, strict=True):
        if entry["split"] != case.split or case.split not in case.case_id:
            raise ValueError(f"manifest split does not match derived split for {case.case_id}")
    return result


def _write(path: Path, text: str, *, future: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (FIXED_MTIME + (4 * 3600 if future else 0),) * 2)


def _rows() -> list[tuple[object, ...]]:
    source_classes = [*sorted(SOURCE_CLASSES), None]
    provenance = [*sorted(PROVENANCE_CLASSES), None]
    emitters = [("realtime_watcher", "assistant", "/synthetic/.claude/projects/brainlayer/session.jsonl"), ("claude_code", "other", "/synthetic/.claude/projects/golems/session.jsonl"), ("codex_cli", "assistant", "/synthetic/.codex/sessions/2026/09/13/rollout.jsonl"), ("mcp", None, "brainbar-store"), (None, "assistant", "/synthetic/sender-only/session.jsonl"), (None, None, "realtime-hook")]  # fmt: skip
    rows = []
    for index in range(25):
        source, sender, source_file = emitters[index % len(emitters)]
        created = (GENERATED_AT - timedelta(hours=index if index < 12 else 24 + index * 5)).isoformat().replace("+00:00", "Z")  # fmt: skip
        # fmt: off
        rows.append((f"synthetic-{index:02d}", f"Synthetic observability fixture row {index:02d}",
            "{}", source_file, "brainlayer-fixture", "assistant_text", source, sender, created,
            (None if index in {5, 17} else provenance[index % len(provenance)]),
            source_classes[index % len(source_classes)],
            ("knowledge", "decision", "operational", "test")[index % 4],
            "synthetic-00" if index == 21 else None,
            GENERATED_AT.isoformat().replace("+00:00", "Z") if index == 20 else None))
        # fmt: on
    rows.append(("synthetic-null-created", "Synthetic row with NULL created_at", "{}", "brainbar-store",
        "brainlayer-fixture", "assistant_text", "mcp", None, None, None, "mcp", "knowledge", None, None))
    rows.append(("synthetic-tilde-created", "Synthetic row with malformed created_at", "{}", "brainbar-store",
        "brainlayer-fixture", "assistant_text", "mcp", None, "2026-05-28T~12:35:00Z", None, "mcp", "knowledge", None, None))
    return rows


def _build_db(path: Path, case: CaseDefinition, pid_root: Path) -> None:
    os.environ["BRAINLAYER_WRITER_PIDFILE_DIR"] = str(pid_root)
    store = VectorStore(path)
    try:
        if case.failure != "empty_db":
            columns = "id content metadata source_file project content_type source sender created_at provenance_class source_class content_class superseded_by archived_at".split()
            chunks = [dict(zip(columns, row)) for row in _rows()]
            store.upsert_chunks(chunks, [[0.0] * 1024] * len(chunks))
            store.conn.execute("UPDATE chunks SET created_at = NULL WHERE id = 'synthetic-null-created'")
            store.conn.execute("UPDATE chunks SET created_at = '2026-05-28T~12:35:00Z' WHERE id = 'synthetic-tilde-created'")
            store.conn.execute("UPDATE chunks SET last_seen_at = ? WHERE id IN ('synthetic-null-created', 'synthetic-tilde-created')", (GENERATED_AT.isoformat().replace("+00:00", "Z"),))
            store.conn.execute("UPDATE chunks SET source = NULLIF(source, 'unknown'), ingested_at = ?", (FIXED_MTIME,))
            store.conn.execute("UPDATE chunks SET superseded_by = 'synthetic-00' WHERE id = 'synthetic-21'")
            store.conn.execute("UPDATE chunks SET archived_at = ? WHERE id = 'synthetic-20'", (GENERATED_AT.isoformat().replace("+00:00", "Z"),))  # fmt: skip
        store.conn.execute("UPDATE schema_migrations SET applied_at = ?", (GENERATED_AT.isoformat(),))
        if case.failure == "missing_source_class":
            store.conn.execute("DROP INDEX idx_chunks_source_class")
            store.conn.execute("ALTER TABLE chunks DROP COLUMN source_class")
        store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        store.close()
    connection = sqlite3.connect(path)
    try:
        connection.execute("VACUUM")
    finally:
        connection.close()
    for suffix in ("-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)
    os.utime(path, (FIXED_MTIME, FIXED_MTIME))


def _jsonl_log(profile: str) -> str:
    if profile in {"no_op", "legacy_no_op"}:
        receipt = {"status": "no-op", "message": "no-op, 0 files already covered", "uploaded": False, "verified": True}
        if profile == "no_op":
            receipt["attempted_at"] = "2026-09-13T10:00:00Z"
        return json.dumps(receipt) + "\n"
    # fmt: off
    shapes = [
        {"status": "uploaded", "archive": "claude-jsonl-2026-09-10.tar.gz", "uploaded": True,
         "verified": True, "bytes": 100, "drive_file": "synthetic-legacy", "retention_deleted": 1,
         "already_covered_files": 2, "archive_listing_count": 3, "bundled_file_count": 4,
         "gzip_test": True, "skipped_active_count": 1, "source_file_count": 5},
        {"status": "uploaded", "archive": "claude-jsonl-2026-09-11.tar.gz", "uploaded": True,
         "verified": True, "bytes": 110, "drive_file": "synthetic-current", "retention_deleted": 1,
         "already_covered_files": 2, "archive_listing_count": 3, "bundled_file_count": 4,
         "forever_files": 1, "forever_uploaded_file_count": 1, "gzip_test": True,
         "local_archive_removed": True, "skipped_active_count": 1, "source_file_count": 5},
        {"status": "uploaded", "attempted_at": "2026-09-13T10:00:00Z",
         "archive": "claude-jsonl-2026-09-13.tar.gz", "archive_id": "synthetic-archive-815",
         "md5Checksum": "0123456789abcdef0123456789abcdef", "uploaded": True, "verified": True,
         "retention_invariant": "PASS", "surviving_archives_30d": 7},
    ]
    # fmt: on
    return "".join(json.dumps(item, sort_keys=True) + "\n" for item in shapes)


def _daily_log(profile: str) -> str:
    if profile == "errors":
        failures = [{"attempted_at": f"2026-09-{day:02d}T05:00:00Z", "error_type": "FileNotFoundError", "error": "Synthetic Drive token fixture missing", "uploaded": False, "verified": False, "backup_log_provenance": "real"} for day in (11, 12, 13)]  # fmt: skip
        return "".join(json.dumps(item, sort_keys=True) + "\n" for item in failures)
    receipt = {"attempted_at": "2026-09-13T09:00:00Z", "snapshot": "/synthetic/backups/2026-09-13.db.gz",
        "db": "/synthetic/brainlayer.db", "drive_file": {"id": "synthetic-id", "name": "synthetic-drive", "size": "123"},
        "uploaded": True, "verified": True, "drive_md5_match": True,
        "backup_log_provenance": "real"}  # fmt: skip
    if profile == "legacy_no_op":
        receipt.pop("attempted_at")
    return "drive upload progress: 50/100 bytes\n" + json.dumps(receipt, sort_keys=True) + "\ndrive upload progress: 100/100 bytes\n"  # fmt: skip


def build_fixture_bundle(root: Path, *, seed: int) -> None:
    if seed != SEED:
        raise ValueError(f"seed must be {SEED}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["heldout_goldens_sha256"] != HELDOUT_SHA256:
        raise ValueError("cases.json held-out digest does not match the sealed digest")
    root.mkdir(parents=True, exist_ok=True)
    for generated_dir in ("db", "logs", "launchd", ".writer-pids"):
        path = root / generated_dir
        if path.exists():
            shutil.rmtree(path)
    prior_pid_root = os.environ.get("BRAINLAYER_WRITER_PIDFILE_DIR")
    try:
        for case in case_definitions():
            _build_case(root, case)
    finally:
        if prior_pid_root is None:
            os.environ.pop("BRAINLAYER_WRITER_PIDFILE_DIR", None)
        else:
            os.environ["BRAINLAYER_WRITER_PIDFILE_DIR"] = prior_pid_root
    # This second pass fixes final page layout; removing it changes every committed fixture DB byte.
    for case in case_definitions():
        db_path = root / case.db_path
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("VACUUM")
        finally:
            connection.close()
        for suffix in ("-wal", "-shm"):
            db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
        os.utime(db_path, (FIXED_MTIME, FIXED_MTIME))
    shutil.rmtree(root / ".writer-pids", ignore_errors=True)
    for case in manifest["cases"]:
        future = case["failure"] == "clock_skew"
        case["input_mtimes"] = {
            path: _mtime_iso(future=future and path.endswith("jsonl-backup.log"))
            for path in case["declared_inputs"]
        }
    _write(root / "cases.json", json.dumps(manifest, indent=2) + "\n")


def _build_case(root: Path, case: CaseDefinition) -> None:
    _build_db(root / case.db_path, case, root / ".writer-pids")
    log_root = root / "logs" / case.case_id
    if case.failure != "missing_log":
        payload = "{malformed\n" if case.failure == "malformed_log" else _jsonl_log(case.profile)
        _write(log_root / "jsonl-backup.log", payload, future=case.failure == "clock_skew")
    _write(log_root / "backup-daily.log", _daily_log(case.profile))
    launchd = root / "launchd" / f"{case.case_id}.txt"
    if case.failure == "missing_launchd":
        _write(launchd, "")
    elif case.profile in {"no_op", "legacy_no_op"}:
        _write(launchd, 'Bad request.\nCould not find service "com.brainlayer.jsonl-backup" in domain for user gui: 501\n')  # fmt: skip
    else:
        _write(launchd, "gui/501/com.brainlayer.jsonl-backup = {\n\tstate = running\n\truns = 6\n\tpid = 4242\n\tlast exit code = 0\n}\n")  # fmt: skip
    if case.profile == "healthy":
        disabled_dir = root / "launchd" / f"{case.case_id}.disabled"
        _write(disabled_dir / "com.brainlayer.jsonl-backup.plist", "synthetic disabled fixture\n")  # fmt: skip
        os.utime(disabled_dir, (FIXED_MTIME, FIXED_MTIME))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/observability"))
    args = parser.parse_args()
    build_fixture_bundle(args.output, seed=args.seed)
    print(f"built {len(case_definitions())} cases at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
