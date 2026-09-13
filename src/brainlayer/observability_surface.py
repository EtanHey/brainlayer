"""Produce the versioned BrainLayer observability document."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .paths import get_db_path
from .pipeline.secret_scrub import scrub_secrets

# fmt: off
WINDOW_HOURS = 24
DERIVATION_NOTE = "metadata.attributionAgent is absent in the 2026-09-13 census; emitter derives from source, then sender, then source_file, or the section is unmeasurable"
CENSUS_NOTE = "never_classified = 476,679 of 731,153 live (65.2%); both-NULL = 65,017; classified_unknown literal not observed on the copy"
def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
class InputRecorder:
    def __init__(self, *, root: Path | None, trace_path: Path | None, now: datetime) -> None:
        self.root = root.resolve() if root else None
        self.trace_path = trace_path
        self.now = now
        self.trace: list[str] = []

    def display_path(self, path: Path) -> str:
        resolved = path.expanduser().resolve()
        if self.root is not None:
            try:
                return str(resolved.relative_to(self.root))
            except ValueError:
                pass
        return str(resolved)

    def __call__(self, path: Path | str, *, status: str = "read", rows_or_bytes: int | None = None,
                 skipped_lines: int = 0, in_section_inputs: bool = True) -> dict[str, Any]:
        del in_section_inputs  # The caller decides whether to retain the returned object.
        resolved = Path(path).expanduser().resolve()
        displayed = self.display_path(resolved)
        self.trace.append(displayed)
        try:
            stat = resolved.stat()
        except FileNotFoundError:
            return _input(displayed, "missing", None, None, None, skipped_lines)
        effective_status = status
        if resolved.is_file() and stat.st_size == 0:
            effective_status = "empty"
        elif status == "read" and datetime.fromtimestamp(stat.st_mtime, UTC) > self.now:
            effective_status = "future"
        digest = None
        if resolved.is_file() and resolved.suffix not in {".sqlite", ".db"}:
            with resolved.open("rb") as handle:
                digest = hashlib.sha256(handle.read(65_536)).hexdigest()
        return _input(displayed, effective_status, _iso_utc(datetime.fromtimestamp(stat.st_mtime, UTC)),
                      stat.st_size if rows_or_bytes is None else rows_or_bytes, digest, skipped_lines)

    def write_trace(self) -> None:
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.write_text(json.dumps(self.trace, indent=2) + "\n", encoding="utf-8")


def _clean(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
def _input(path: str, status: str, mtime: str | None, size: int | None,
           digest: str | None, skipped: int) -> dict[str, Any]:
    return {"path": path, "status": status, "mtime": mtime, "rows_or_bytes": size,
            "sha256_first_64kb": digest, "skipped_lines": skipped}


def derive_emitter(source: object, sender: object, source_file: object) -> tuple[str, str]:
    """Derive the emitter in source, sender, source_file precedence order."""
    for value, origin in ((source, "source"), (sender, "sender")):
        if cleaned := _clean(value):
            return cleaned, origin
    path = _clean(source_file) or "unknown"
    if "/.codex/sessions/" in path:
        return "codex", "source_file"
    if "/.claude/projects/" in path:
        project_dir = path.split("/.claude/projects/", 1)[1].split("/", 1)[0]
        if "-Gits-" in project_dir:
            return project_dir.rsplit("-Gits-", 1)[1] or "unknown", "source_file"
        return project_dir or "unknown", "source_file"
    return path, "source_file"


def _unmeasurable(reason: str, db_input: dict[str, Any]) -> dict[str, Any]:
    return {"state": "unmeasurable", "reason": reason, "inputs": [db_input]}
def _measured(db_input: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {"state": "measured", "reason": "", "inputs": [db_input], **fields}


def _missing(columns: set[str], required: tuple[str, ...]) -> str | None:
    return next((name for name in required if name not in columns), None)
def _stores(connection: sqlite3.Connection, columns: set[str], db_input: dict[str, Any], now: datetime) -> dict[str, Any]:
    required = ("source_class", "id", "content", "source", "sender", "source_file", "created_at", "content_class")
    if missing := _missing(columns, required):
        return _unmeasurable(f"required column missing: chunks.{missing}", db_input)
    cutoff = _iso_utc(now - timedelta(hours=WINDOW_HOURS))
    now_text = _iso_utc(now)
    total = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    rows = connection.execute("SELECT content_class, COUNT(*) FROM chunks GROUP BY content_class ORDER BY content_class IS NULL, content_class")
    content_classes = [{"content_class": row[0], "count": row[1]} for row in rows]
    rows = connection.execute("SELECT strftime('%Y-%m-%dT%H:00:00Z', created_at), COUNT(*) FROM chunks WHERE created_at >= ? AND created_at <= ? GROUP BY 1 ORDER BY 1", (cutoff, now_text))
    by_hour = [{"hour": row[0], "count": row[1]} for row in rows]
    latest = []
    for row in connection.execute(
        "SELECT id, created_at, source_class, source, sender, source_file, content "
        "FROM chunks ORDER BY created_at DESC LIMIT 5"
    ):
        emitter, _ = derive_emitter(row[3], row[4], row[5])
        latest.append({"chunk_id": str(row[0]), "stored_at": _iso_utc(_parse_time(row[1])),
                       "source_class": row[2], "emitter": emitter,
                       "preview": scrub_secrets(row[6] or "").text[:80]})
    return _measured(db_input, total_chunks=total,
                     in_window={"count": sum(item["count"] for item in by_hour), "by_hour": by_hour},
                     by_content_class=content_classes, latest=latest)


def _emitters(connection: sqlite3.Connection, columns: set[str], db_input: dict[str, Any], now: datetime) -> dict[str, Any]:
    required = ("source_class", "source", "sender", "source_file", "created_at")
    if missing := _missing(columns, required):
        return _unmeasurable(f"required column missing: chunks.{missing}", db_input)
    cutoff, now_text = _iso_utc(now - timedelta(hours=WINDOW_HOURS)), _iso_utc(now)
    rows = connection.execute("SELECT source_class, COUNT(*), SUM(CASE WHEN created_at >= ? AND created_at <= ? THEN 1 ELSE 0 END) FROM chunks GROUP BY source_class ORDER BY source_class IS NULL, source_class", (cutoff, now_text))
    by_source = [{"source_class": row[0], "count": row[1], "in_window": row[2]} for row in rows]
    counts: Counter[tuple[str, str]] = Counter()
    for row in connection.execute(
        "SELECT source, sender, source_file FROM chunks WHERE created_at >= ? AND created_at <= ?",
        (cutoff, now_text),
    ):
        counts[derive_emitter(*row)] += 1
    by_emitter = [{"emitter": emitter, "derived_from": origin, "count_in_window": count}
                  for (emitter, origin), count in sorted(counts.items())]
    hidden = connection.execute("SELECT COUNT(*) FROM chunks WHERE source_class IN ('desktop', 'brain-worker')").fetchone()[0]
    return _measured(db_input, by_source_class=by_source, by_emitter=by_emitter,
                     derivation_note=DERIVATION_NOTE, hidden_from_default_search=hidden)


def _author_unknown(connection: sqlite3.Connection, columns: set[str], db_input: dict[str, Any], now: datetime) -> dict[str, Any]:
    required = ("source_class", "provenance_class", "archived_at", "created_at", "source_file")
    if missing := _missing(columns, required):
        return _unmeasurable(f"required column missing: chunks.{missing}", db_input)
    live = connection.execute("SELECT COUNT(*) FROM chunks WHERE archived_at IS NULL").fetchone()[0]
    never_where = "archived_at IS NULL AND (provenance_class IS NULL OR source_class IS NULL)"
    unknown_where = "archived_at IS NULL AND provenance_class = 'unknown'"
    never = connection.execute(f"SELECT COUNT(*) FROM chunks WHERE {never_where}").fetchone()[0]
    unknown = connection.execute(f"SELECT COUNT(*) FROM chunks WHERE {unknown_where}").fetchone()[0]
    start_day = (now - timedelta(days=6)).date().isoformat()
    daily: dict[str, list[int]] = {}
    for row in connection.execute(
        f"SELECT date(created_at), SUM(CASE WHEN {never_where} THEN 1 ELSE 0 END), "
        f"SUM(CASE WHEN {unknown_where} THEN 1 ELSE 0 END) FROM chunks "
        "WHERE date(created_at) >= ? AND date(created_at) <= ? GROUP BY 1",
        (start_day, now.date().isoformat()),
    ):
        daily[row[0]] = [row[1], row[2]]
    trend = []
    for offset in range(7):
        day = (now.date() - timedelta(days=6 - offset)).isoformat()
        values = daily.get(day, [0, 0])
        trend.append({"day": day, "never_classified": values[0], "classified_unknown": values[1]})
    rows = connection.execute(f"SELECT source_file, COUNT(*) FROM chunks WHERE {never_where} OR ({unknown_where}) GROUP BY source_file ORDER BY COUNT(*) DESC, source_file LIMIT 5")
    top_files = [{"source_file": row[0] or "", "count": row[1]} for row in rows]
    return _measured(
        db_input,
        never_classified={
            "count": never,
            "share": round(never / live, 6) if live else 0.0,
            "definition": "provenance_class IS NULL OR source_class IS NULL, archived_at IS NULL",
        },
        classified_unknown={
            "count": unknown,
            "share": round(unknown / live, 6) if live else 0.0,
            "definition": "provenance_class = 'unknown', archived_at IS NULL",
        },
        census_2026_09_13=CENSUS_NOTE,
        trend_7d=trend,
        top_source_files=top_files,
    )


def build_document(*, env: Mapping[str, str] = os.environ) -> tuple[dict[str, Any], InputRecorder]:
    expected_root = env.get("BRAINLAYER_OBSERVABILITY_PRODUCER_ROOT")
    if expected_root and not Path(__file__).resolve().is_relative_to(Path(expected_root).resolve() / "src"):
        raise RuntimeError(f"observability_surface imported outside producer root: {Path(__file__).resolve()}")
    now = _parse_time(env.get("BRAINLAYER_OBSERVABILITY_NOW", datetime.now(UTC).isoformat()))
    db_path = Path(env.get("BRAINLAYER_DB", str(get_db_path()))).expanduser().resolve()
    root = Path(env["BRAINLAYER_OBSERVABILITY_INPUT_ROOT"]).expanduser() if env.get("BRAINLAYER_OBSERVABILITY_INPUT_ROOT") else None
    trace_path = (
        Path(env["BRAINLAYER_OBSERVABILITY_TRACE_PATH"]).expanduser() if env.get("BRAINLAYER_OBSERVABILITY_TRACE_PATH") else None
    )
    recorder = InputRecorder(root=root, trace_path=trace_path, now=now)
    connection: sqlite3.Connection | None = None
    db_status, db_rows, failure = "read", None, None
    try:
        connection = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        db_rows = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
    except (OSError, sqlite3.Error) as exc:
        db_status = "missing" if not db_path.exists() else "malformed"
        columns, failure = set(), f"database unreadable: {exc}"
    db_input = recorder(db_path, status=db_status, rows_or_bytes=db_rows)
    if db_input["status"] in {"missing", "malformed", "empty", "future"}:
        failure = failure or f"database input status: {db_input['status']}"
    if failure or connection is None:
        stores = emitters = author_unknown = _unmeasurable(failure or "database unavailable", db_input)
    else:
        stores = _stores(connection, columns, db_input, now)
        emitters = _emitters(connection, columns, db_input, now)
        author_unknown = _author_unknown(connection, columns, db_input, now)
    if connection is not None:
        connection.close()
    try:
        from .observability_backup import build_backups_section

        backups = build_backups_section(env=env, record_input=recorder, now=now)
    except ImportError:
        for name in ("JSONL_BACKUP_LOG", "BACKUP_DAILY_LOG", "LAUNCHD_OUTPUT", "DISABLED_DIR"):
            if path := env.get(f"BRAINLAYER_OBSERVABILITY_{name}"):
                recorder(path, in_section_inputs=False)
        backups = {"state": "unmeasurable", "reason": "backups module not installed", "inputs": []}
    document = {
        "schema_version": 1,
        "generated_at": _iso_utc(now),
        "db_path": recorder.display_path(db_path),
        "window_hours": WINDOW_HOURS,
        "stores": stores,
        "emitters": emitters,
        "author_unknown": author_unknown,
        "backups": backups,
    }
    return document, recorder
def write_document(*, env: Mapping[str, str] = os.environ, stdout: bool = False) -> dict[str, Any]:
    document, recorder = build_document(env=env)
    try:
        payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
        if stdout:
            print(payload, end="")
        else:
            db_path = Path(env.get("BRAINLAYER_DB", str(get_db_path()))).expanduser().resolve()
            output = Path(env.get("BRAINLAYER_OBSERVABILITY_PATH", str(db_path.parent / "observability.json"))).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
        return document
    finally:
        recorder.write_trace()


def main() -> int:
    write_document()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# fmt: on
