"""Manual provider redaction on copies or a guarded, quiesced runtime DB."""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

import apsw

from . import chunk_origin_wipe, maintenance
from .chunk_origin_wipe import assert_not_live_db
from .chunk_write import canonical_content_hash
from .dedupe import BUSY_RETRY_ATTEMPTS, _busy_retry_delay, compute_dedupe_fields
from .maintenance import _maintenance_lock
from .pipeline.secret_scrub import scrub_secrets
from .runtime_store import ReadonlyStore, WriterRuntimeStore
from .vector_store import value_free_sqlite_logging
from .wal_checkpoint import checkpoint_guard

PROVIDERS = frozenset({"google_oauth_access", "google_oauth_refresh", "google_client_secret"})
PROVIDER_MODES = {
    "google_oauth": PROVIDERS,
    "context7": frozenset({"context7"}),
    "exa_labeled": frozenset({"exa_labeled"}),
}
_PROVIDER_PREFIXES = {
    "google_oauth_access": "ya29.",
    "google_oauth_refresh": "1//",
    "google_client_secret": "GOCSPX-",
    "context7": "ctx7sk-",
    "exa_labeled": "exa",
}
LIVE_SERVICES = (
    "fleet-watchdog",
    "throughput-watchdog",
    "tier0-watchdog",
    "health-check",
    # The daemon's UI-watchdog can fall back to openBundle after the UI bootout.
    # Stop that supervisor before its subject; the UI only kickstarts the daemon.
    "brainbar-daemon",
    "brainbar",
    "hotlane-brainbar",
    "watch",
    "drain",
    "index",
    "t3-ingest",
    "decay",
    "enrichment",
)


BRAINBAR_EXIT_TIMEOUT_SECONDS = 30.0
BACKUP_WAIT_SAFE_MARGIN_SECONDS = 60.0
_QUIESCE_DETAILS = frozenset(
    {"quiesce-services", "brainbar-process-probe", "lsof-writers", "process:BrainBar", "process:BrainBarDaemon"}
    | {
        f"{step}:{maintenance._launchd_label(service)}"
        for step in ("bootout", "state", "loaded")
        for service in LIVE_SERVICES
    }
)


class ScrubAtRestError(RuntimeError):
    """Value-free failure; a failed batch is rolled back, earlier batches may remain."""

    def __init__(self, message, *, reason="scrub-failed", detail=None):
        super().__init__(message)
        self.reason = reason
        # Only fixed service labels and gate names can reach refusal JSON.
        self.detail = detail if isinstance(detail, str) and detail in _QUIESCE_DETAILS else None


def _safe_cause(exc):
    # Preserve the error category without retaining SQLite messages/SQL/notes.
    try:
        return type(exc)(type(exc).__name__)
    except Exception:
        return RuntimeError(type(exc).__name__)


def _failure(exc, cleanup=None):
    error = ScrubAtRestError(f"at-rest scrub failed ({type(exc).__name__}); no matched values logged")
    error.__cause__ = _safe_cause(exc)
    if isinstance(exc, ScrubAtRestError):
        error.reason = exc.reason
        error.detail = exc.detail
        error.__cause__ = exc.__cause__
        for note in getattr(exc, "__notes__", []):
            error.add_note(note)
    if cleanup is not None:
        error.add_note(f"cleanup failed ({type(cleanup).__name__})")
    return error


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _tables(conn):
    inventory = list(conn.execute("PRAGMA table_list"))
    targets = []
    for _, table, kind, _, without_rowid, _ in inventory:
        if kind in {"shadow", "view"} or table.startswith("sqlite_"):
            continue
        sql = conn.execute("SELECT sql FROM sqlite_schema WHERE name=?", (table,)).fetchone()[0] or ""
        if kind == "virtual":
            if re.search(r"\busing\s+vec0\b", sql, re.IGNORECASE):
                continue
            if not re.search(r"\busing\s+fts5\b", sql, re.IGNORECASE) or re.search(
                r"\bcontent\s*=", sql, re.IGNORECASE
            ):
                raise ValueError("unsupported virtual or external-content table")
        info = list(conn.execute(f"PRAGMA table_xinfo({_quote(table)})"))
        columns = [r[1] for r in info if not r[6]]
        if columns:
            keys = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5]] if without_rowid else ["_rowid_"]
            if not keys or "_rowid_" in columns:
                raise ValueError("unsupported table layout")
            targets.append((table, columns, kind, keys))
    # Base rows before derived copies; history and FTS are scrubbed last.
    return sorted(targets, key=lambda t: (t[0] == "_chunks_history", t[2] == "virtual", t[0] != "chunks", t[0]))


def _batch(conn, table, columns, keys, last, size, selected=PROVIDERS):
    prefixes = tuple(_PROVIDER_PREFIXES[p] for p in sorted(selected))

    # EXA labels need both exa and key; the regex enforces their label boundaries.
    # The other provider prefixes are exact.
    def predicate(column, prefix):
        quoted = _quote(column)
        expression = f"lower({quoted})" if prefix == "exa" else quoted
        key_filter = f" AND instr({expression}, 'key') > 0" if prefix == "exa" else ""
        return f"(typeof({quoted})='text' AND instr({expression}, ?) > 0{key_filter})"

    where = " OR ".join(predicate(c, p) for c in columns for p in prefixes)
    params = [p for _ in columns for p in prefixes]
    key_sql = ",".join(map(_quote, keys))
    pagination = "1" if last is None else f"({key_sql}) > ({','.join('?' for _ in keys)})"
    if last is not None:
        params = [*last, *params]
    return list(
        conn.execute(
            f"SELECT {key_sql}, {','.join(map(_quote, columns))} FROM {_quote(table)} "
            f"WHERE {pagination} AND ({where}) ORDER BY {key_sql} LIMIT ?",
            (*params, size),
        )
    )


def _rewrite(conn, table, columns, keys, rows, apply, selected=PROVIDERS):
    counts = {"rows": 0, "columns": {}}
    for row in rows:
        key, values = row[: len(keys)], row[len(keys) :]
        changes = {}
        for column, value in zip(columns, values):
            if not isinstance(value, str):
                continue
            result = scrub_secrets(value, providers=selected)
            if result.redactions:
                changes[column] = result.text
                providers = {r.provider for r in result.redactions}
                counts["columns"].setdefault(column, Counter()).update(providers)
        if not changes:
            continue
        counts["rows"] += 1
        if not apply:
            continue
        if any(c == "id" or c.endswith("_id") for c in changes):
            raise ValueError("identity/reference matches require a separate repair")
        if table in {"chunks", "_chunks_history"} and "content" in changes:
            content = changes["content"]
            original = dict(zip(columns, values))
            fields = compute_dedupe_fields(content, original.get("created_at"))
            changes.update(
                content_hash=canonical_content_hash(content),
                dedupe_hash=fields.dedupe_hash,
                simhash=fields.simhash,
                **{f"simhash_band_{i}": v for i, v in enumerate(fields.bands)},
            )
            changes["char_count"] = len(content)
        conn.execute(
            f"UPDATE {_quote(table)} SET {','.join(_quote(c) + '=?' for c in changes)} WHERE {' AND '.join(_quote(k) + '= ?' for k in keys)}",
            (*changes.values(), *key),
        )
        if table == "session_enrichments":
            session = conn.execute(
                "SELECT session_id,session_summary,what_worked,what_failed FROM session_enrichments WHERE rowid=?",
                key,
            ).fetchone()
            conn.execute("DELETE FROM session_enrichments_fts WHERE session_id=?", (session[0],))
            if any(session[1:]):
                conn.execute(
                    "INSERT INTO session_enrichments_fts(session_id,session_summary,what_worked,what_failed) VALUES(?,?,?,?)",
                    session,
                )
    return counts


def _run(store, dry_run, batch_size, selected=PROVIDERS):
    conn = store.conn
    tables = _tables(conn)
    report = {"dry_run": dry_run, "batches": 0, "tables": {}}
    for table, columns, _, keys in tables:
        total = {"rows": 0, "columns": {}}
        report["tables"][table] = total
        last = None
        while True:
            for attempt in range(BUSY_RETRY_ATTEMPTS):
                try:
                    if not dry_run:
                        conn.execute("BEGIN IMMEDIATE")
                    rows = _batch(conn, table, columns, keys, last, batch_size, selected)
                    # Preserve independent previews and avoid unsanitized history preimages.
                    suppressed = (
                        list(
                            conn.execute(
                                "SELECT name,sql FROM sqlite_schema WHERE type='trigger' AND "
                                "name IN ('chunks_bitemporal_update','chunks_preview_text_update')"
                            )
                        )
                        if table == "chunks" and rows and not dry_run
                        else []
                    )
                    for name, _ in suppressed:
                        conn.execute(f"DROP TRIGGER {_quote(name)}")
                    counts = _rewrite(conn, table, columns, keys, rows, not dry_run, selected)
                    for _, sql in suppressed:
                        conn.execute(sql)
                    if not dry_run:
                        conn.execute("COMMIT")
                    break
                except BaseException as exc:
                    if not conn.getautocommit():
                        try:
                            conn.execute("ROLLBACK")
                        except Exception as cleanup:
                            raise _failure(exc, cleanup) from _safe_cause(exc)
                    if not isinstance(exc, apsw.BusyError) or attempt == BUSY_RETRY_ATTEMPTS - 1:
                        raise
                    time.sleep(_busy_retry_delay(attempt))
            if not rows:
                break
            total["rows"] += counts["rows"]
            for column, providers in counts["columns"].items():
                total["columns"].setdefault(column, Counter()).update(providers)
            report["batches"] += 1
            last = rows[-1][: len(keys)]
    return report


def _checkpoint(store):
    with checkpoint_guard(store.db_path):
        if store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
            raise apsw.BusyError("scrub checkpoint busy")


def _apply(path, batch_size, selected=PROVIDERS):
    with WriterRuntimeStore(path) as store:
        _checkpoint(store)
        failure = None
        try:
            result = _run(store, False, batch_size, selected)
            if any(t["rows"] for t in _run(store, True, batch_size, selected)["tables"].values()):
                raise RuntimeError("remaining provider matches")
        except BaseException as exc:
            failure = _failure(exc)
        try:
            _checkpoint(store)
        except Exception as cleanup:
            if failure is None:
                failure = _failure(cleanup)
            else:
                failure.add_note(f"checkpoint failed ({type(cleanup).__name__})")
        if failure is not None:
            raise failure
        return result


def _live_requirements(config):
    if not maintenance._service_is_deliberately_paused("enrichment"):
        raise ScrubAtRestError("active enrichment pause required", reason="enrichment-pause-required")
    if maintenance._recent_verified_backup(config) is None:
        raise ScrubAtRestError("verified backup within 24 hours required", reason="verified-backup-required")


def _wait_for_verified_backup(path, timeout_seconds):
    """Wait without holding a maintenance lock or stopping any writer services."""
    config = maintenance.MaintenanceConfig(db_path=path, backup_reuse_max_age_hours=24)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            _live_requirements(config)
            return
        except ScrubAtRestError as exc:
            if exc.reason != "verified-backup-required":
                raise
        remaining = min(
            deadline - time.monotonic(),
            maintenance._remaining_quiet_window_seconds(config) - BACKUP_WAIT_SAFE_MARGIN_SECONDS,
        )
        if remaining <= 0:
            raise ScrubAtRestError("verified backup wait timed out", reason="verified-backup-timeout")
        time.sleep(min(30.0, remaining))
        # The final poll may have produced a receipt; inspect it on the next loop.
        # Preserve the window margin even when that final receipt is available.
        if maintenance._remaining_quiet_window_seconds(config) <= BACKUP_WAIT_SAFE_MARGIN_SECONDS:
            raise ScrubAtRestError("verified backup wait timed out", reason="verified-backup-timeout")


def _check_no_brainbar_processes():
    # An unregistered UI can open the daemon bundle even after its job is booted out.
    try:
        processes = maintenance.run_command(["ps", "-axo", "comm="], check=True)
    except Exception as exc:
        raise ScrubAtRestError(
            "could not verify BrainBar process exit", reason="quiesce-failed", detail="brainbar-process-probe"
        ) from _safe_cause(exc)
    for line in processes.stdout.splitlines():
        name = Path(line.strip()).name
        if name in {"BrainBar", "BrainBarDaemon"}:
            raise ScrubAtRestError(
                "BrainBar process remains running", reason="quiesce-failed", detail=f"process:{name}"
            )


def _wait_for_brainbar_exit():
    """Bootout removes a job before its process necessarily finishes SIGTERM cleanup."""
    deadline = time.monotonic() + BRAINBAR_EXIT_TIMEOUT_SECONDS
    while True:
        try:
            _check_no_brainbar_processes()
            return
        except ScrubAtRestError as exc:
            if exc.detail not in {"process:BrainBar", "process:BrainBarDaemon"}:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(0.1, remaining))


def _quiesced_gates(config):
    # Both healer jobs and unregistered UI processes can revive resident writers.
    for service in LIVE_SERVICES:
        try:
            loaded = maintenance._service_is_loaded(service)
        except maintenance.MaintenanceAbort as exc:
            raise ScrubAtRestError(
                "writer service state unknown", reason="quiesce-failed", detail=exc.detail
            ) from _safe_cause(exc)
        if loaded:
            raise ScrubAtRestError(
                "writer service remains loaded",
                reason="quiesce-failed",
                detail=f"loaded:{maintenance._launchd_label(service)}",
            )
    _check_no_brainbar_processes()
    config.expected_writer_patterns = ()
    try:
        maintenance._check_lsof_clean(config)
    except Exception as exc:
        raise ScrubAtRestError(
            "writer file-descriptor gate failed", reason="quiesce-failed", detail="lsof-writers"
        ) from _safe_cause(exc)


def _guarded_apply(path, batch_size, expect_rows, selected=PROVIDERS):
    if expect_rows is None or expect_rows < 0:
        raise ScrubAtRestError("expected row count required", reason="expected-row-count-required")
    config = maintenance.MaintenanceConfig(db_path=path, backup_reuse_max_age_hours=24)
    _live_requirements(config)
    maintenance._run_gates(config)
    booted_out, failure = {}, None
    try:
        try:
            maintenance._quiesce_services(LIVE_SERVICES, booted_out)
        except Exception as exc:
            raise ScrubAtRestError(
                "failed to quiesce services",
                reason="quiesce-failed",
                detail=getattr(exc, "detail", None) or "quiesce-services",
            ) from _safe_cause(exc)
        _wait_for_brainbar_exit()
        # A successful bootout does not by itself establish that the job stayed down.
        _quiesced_gates(config)
        _live_requirements(config)
        with ReadonlyStore(path) as store:
            current = _run(store, True, batch_size, selected)
        if sum(t["rows"] for t in current["tables"].values()) != expect_rows:
            raise ScrubAtRestError("expected row count mismatch", reason="expected-row-count-mismatch")
        _live_requirements(config)
        _quiesced_gates(config)
        result = _apply(path, batch_size, selected)
    except BaseException as exc:
        failure = _failure(exc)
    finally:
        try:
            resume_failures = maintenance._resume_services(config.repo_root, tuple(reversed(LIVE_SERVICES)), booted_out)
            if resume_failures:
                error = ScrubAtRestError(
                    f"failed to resume services: count={len(resume_failures)}", reason="resume-failed"
                )
                if failure is None:
                    failure = error
                else:
                    failure.add_note(str(error))
        except Exception as exc:
            if failure is None:
                failure = _failure(exc)
            else:
                failure.add_note(f"resume failed ({type(exc).__name__})")
    if failure is not None:
        raise failure
    return result


def scrub_at_rest(
    db_path: Path,
    *,
    dry_run: bool = False,
    batch_size: int = 100,
    providers: str = "google_oauth",
    allow_live_db: bool = False,
    expect_rows: int | None = None,
    wait_for_backup_seconds: int = 0,
) -> dict:
    """Read-only surveys need no opt-in; guarded applies require a current row total."""
    failure = None
    try:
        with value_free_sqlite_logging():
            if providers not in PROVIDER_MODES or not 1 <= batch_size <= 1000 or wait_for_backup_seconds < 0:
                raise ValueError("invalid scrub options")
            selected = PROVIDER_MODES[providers]
            try:
                path = assert_not_live_db(Path(db_path), allow_live=dry_run or allow_live_db)
            except RuntimeError as exc:
                raise ScrubAtRestError("live apply requires opt-in", reason="live-db-requires-allow") from _safe_cause(
                    exc
                )
            # Normalize hardlink aliases to the runtime path for its locks/backup receipt.
            for candidate in chunk_origin_wipe._live_db_candidates():
                if chunk_origin_wipe._same_file(path, candidate):
                    path = candidate.expanduser().resolve()
                    break
            if dry_run:
                with ReadonlyStore(path) as store:
                    return _run(store, True, batch_size, selected)
            if allow_live_db and wait_for_backup_seconds > 0:
                _wait_for_verified_backup(path, wait_for_backup_seconds)
            with _maintenance_lock(path):
                path = assert_not_live_db(path, allow_live=allow_live_db)
                if allow_live_db:
                    return _guarded_apply(path, batch_size, expect_rows, selected)
                return _apply(path, batch_size, selected)
    except Exception as exc:
        failure = _failure(exc)
    raise failure
