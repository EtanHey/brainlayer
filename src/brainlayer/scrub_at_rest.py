"""One-off, copy-only provider redaction; reports contain schema names and counts."""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

import apsw

from .chunk_origin_wipe import assert_not_live_db
from .chunk_write import canonical_content_hash
from .dedupe import BUSY_RETRY_ATTEMPTS, _busy_retry_delay, compute_dedupe_fields
from .maintenance import _maintenance_lock
from .pipeline.secret_scrub import scrub_secrets
from .runtime_store import ReadonlyStore, WriterRuntimeStore
from .wal_checkpoint import checkpoint_guard

PROVIDERS = frozenset({"google_oauth_access", "google_oauth_refresh", "google_client_secret"})
PREFIXES = ("ya29.", "1//", "GOCSPX-")


class ScrubAtRestError(RuntimeError):
    """Value-free failure; a failed batch is rolled back, earlier batches may remain."""


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


def _batch(conn, table, columns, keys, last, size):
    where = " OR ".join(
        f"(typeof({_quote(c)})='text' AND instr({_quote(c)}, ?) > 0)" for c in columns for _ in PREFIXES
    )
    params = [p for _ in columns for p in PREFIXES]
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


def _rewrite(conn, table, columns, keys, rows, apply):
    counts = {"rows": 0, "columns": {}}
    for row in rows:
        key, values = row[: len(keys)], row[len(keys) :]
        changes = {}
        for column, value in zip(columns, values):
            if not isinstance(value, str):
                continue
            result = scrub_secrets(value, providers=PROVIDERS)
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


def _run(store, dry_run, batch_size):
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
                    rows = _batch(conn, table, columns, keys, last, batch_size)
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
                    counts = _rewrite(conn, table, columns, keys, rows, not dry_run)
                    for _, sql in suppressed:
                        conn.execute(sql)
                    if not dry_run:
                        conn.execute("COMMIT")
                    break
                except BaseException as exc:
                    if not conn.getautocommit():
                        conn.execute("ROLLBACK")
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


def scrub_at_rest(
    db_path: Path, *, dry_run: bool = False, batch_size: int = 100, providers: str = "google_oauth"
) -> dict:
    """Scrub explicit offline copies only. Canonical/configured DBs and aliases refuse."""
    try:
        if providers != "google_oauth" or not 1 <= batch_size <= 1000:
            raise ValueError("invalid scrub options")
        path = assert_not_live_db(Path(db_path))
        if dry_run:
            with ReadonlyStore(path) as store:
                return _run(store, True, batch_size)
        with _maintenance_lock(path):
            path = assert_not_live_db(path)
            with WriterRuntimeStore(path) as store:
                _checkpoint(store)
                try:
                    result = _run(store, False, batch_size)
                    if any(t["rows"] for t in _run(store, True, batch_size)["tables"].values()):
                        raise RuntimeError("remaining provider matches")
                    return result
                finally:
                    _checkpoint(store)
    except Exception as exc:
        raise ScrubAtRestError(f"at-rest scrub failed ({type(exc).__name__}); no matched values logged") from None
