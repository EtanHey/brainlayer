"""Stat-pinned, read-only planning and bounded queue admission of explicit rollouts."""

import json
import sqlite3
import stat
from contextlib import closing, contextmanager, nullcontext
from pathlib import Path

from ..dedupe import _active_clause, _scope_clause, find_duplicate, normalized_exact_hash
from ..ingest_denylist import is_denylisted
from ..t3_provenance import compact_codex_header
from ..watcher import OffsetRegistry, normalize_provider_entry
from ..watcher_bridge import create_flush_callback
from ..zstd_stream import ZstdDecodedFile, logical_jsonl_path


def _verify(path, expected):
    for key in ("size", "mtime_ns", "inode"):
        if type(expected.get(key)) is not int:
            raise ValueError(f"missing-expected-{key}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("source-is-not-a-regular-file")
    if (info.st_size, info.st_mtime_ns, info.st_ino) != (expected["size"], expected["mtime_ns"], expected["inode"]):
        raise ValueError("source-stats-mismatch")


def _read_entries(path, limit, receipt):
    """Yield bounded decode windows; completion requires a successful decoder EOF."""
    pending = bytearray()
    session = (None, None)
    header = None
    logical = str(logical_jsonl_path(path))
    with ZstdDecodedFile(path) as stream:
        while receipt["decoded_bytes"] < limit:
            data = stream.read(min(65536, limit - receipt["decoded_bytes"]))
            receipt["decoded_bytes"] += len(data)
            pending.extend(data)
            if len(pending) > 128 * 1024 * 1024:
                raise ValueError("record-size-budget")
            eof = not data
            entries = []
            start = 0
            while (end := pending.find(b"\n", start)) >= 0 or (eof and start < len(pending)):
                if end < 0:
                    end = len(pending)
                    consumed = end
                else:
                    consumed = end + 1
                raw = pending[start:end]
                record_start = receipt["decoded_bytes"] - len(pending) + start
                offset = receipt["decoded_bytes"] - len(pending) + consumed
                start = consumed
                if not raw.strip():
                    continue
                try:
                    line = json.loads(raw)
                    if not isinstance(line, dict):
                        raise ValueError("JSONL record is not an object")
                except (ValueError, UnicodeDecodeError) as error:
                    reason = (
                        "truncated_final_record" if eof and end == len(pending) else f"malformed_record@{record_start}"
                    )
                    if reason == "truncated_final_record":
                        receipt["quarantined_tail"] = True
                        continue
                    raise ValueError(reason) from error
                if record_start == 0 and end < len(pending):
                    header = compact_codex_header(line, consumed)

                receipt["records"] += 1
                if line.get("type") == "session_meta":
                    payload = line.get("payload")
                    if not isinstance(payload, dict):
                        raise ValueError("session_meta payload is not an object")
                    session = (payload.get("id"), payload.get("cwd"))
                entry = normalize_provider_entry(line, "codex")
                if entry:
                    entry.update(
                        _source_file=logical,
                        _line_end_offset=offset,
                        _provider="codex",
                        _compressed_archive=True,
                        _codex_header=header,
                        codex_native_session_id=session[0],
                        cwd=session[1],
                    )
                    entries.append(entry)
            del pending[:start]
            if entries:
                yield entries
            if eof:
                receipt["decoded_bytes_total"] = receipt["decoded_bytes"]
                return
    raise ValueError("decoded-byte-budget")


@contextmanager
def _committed(db_path):
    """One read-only connection per admitted file, closed on every exit."""
    db = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True) if Path(db_path).exists() else None
    try:
        yield db
    finally:
        if db is not None:
            db.close()


def _resolved_candidate(db, row):
    """Resolve read-only in the writer's duplicate-before-reactivation order."""
    if db is None:
        return None
    own = db.execute(
        "SELECT id, source_file, source, archived_at FROM chunks WHERE id = ? AND length(trim(content)) > 0",
        (row["chunk_id"],),
    ).fetchone()
    reactivate = own and (
        own[2] == "realtime_watcher"
        and own[1] == row.get("source_file")
        and own[3] is not None
        and row.get("source_end_offset") is not None
    )
    if own and not reactivate:
        return (*own[:2], "own_id")
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_id_alias'").fetchone():
        alias = db.execute(
            f"SELECT c.id, c.source_file FROM chunk_id_alias a JOIN "
            f"(SELECT id, source_file FROM chunks WHERE {_active_clause(db)} AND length(trim(content)) > 0) c "
            "ON c.id = a.canonical_chunk_id WHERE a.old_chunk_id = ?",
            (row["chunk_id"],),
        ).fetchone()
        if alias:
            return (*alias, "writer_alias")
    if reactivate:
        # The writer checks duplicates BEFORE reactivating its own archived row.
        hit, _ = find_duplicate(
            db,
            chunk_id=row["chunk_id"],
            content=str(row["content"]),
            created_at=row.get("created_at"),
            project=row.get("project"),
            content_type=row.get("content_type"),
        )
        match = (
            db.execute("SELECT id, source_file FROM chunks WHERE id = ?", (hit.canonical_chunk_id,)).fetchone()
            if hit
            else None
        )
        return (*match, "writer_" + hit.mechanism) if match else None
    scope, params = _scope_clause(row.get("project"), row.get("content_type") or "assistant_text")
    exact = db.execute(
        f"SELECT id, source_file FROM chunks WHERE dedupe_hash = ? AND {_active_clause(db)} "
        f"AND {scope} ORDER BY created_at ASC, id ASC LIMIT 1",
        (normalized_exact_hash(str(row["content"]).strip()), *params),
    ).fetchone()
    return (*exact, "exact_hash") if exact else None


def _plan_window(plan, entries, preview, receipt, committed, dry_run):
    from ..codex_zst import queue_has_capacity
    from ..queue_io import enqueue_watcher_chunk
    from ..t3_linkage_alarm import read_only

    preview.clear()
    with read_only() if dry_run else nullcontext():
        result = plan(entries)
    receipt["filtered"] += result.skipped
    if result.deferred_entries:
        raise ValueError("provenance-unresolved")
    receipt["would_insert"] += len(preview)
    if dry_run:
        return
    for row in preview:
        match = _resolved_candidate(committed, row)
        if match:
            if match[0] != row["chunk_id"] or match[1] != row["source_file"]:
                receipt["duplicate_content"] += 1
                example = dict(chunk_id=match[0], source_file=match[1])
                if len(receipt["duplicate_matches"]) < 5 and example not in receipt["duplicate_matches"]:
                    receipt["duplicate_matches"].append(example)
            continue
        if not queue_has_capacity():
            raise ValueError("queue-backpressure")
        enqueue_watcher_chunk(**row)
        receipt["queued"] += 1


def _classify_receipt(receipt, dry_run):
    reason = receipt["skip_reason"]
    if reason:
        category = (
            "deferred"
            if reason in {"admission-budget", "decoded-byte-budget", "queue-backpressure", "provenance-unresolved"}
            else "error"
        )
        if reason == "ingest-denylist":
            category = "filtered"
        if reason == "provenance-unresolved":
            receipt["state"] = "deferred"
    elif receipt["state"] == "skipped_other_machine":
        category, reason = "filtered", "skipped_other_machine"
    elif not receipt["records"]:
        category = "empty"
    elif not receipt["would_insert"]:
        category = "filtered"
    else:
        category = "readable" if dry_run else "accepted"
        if not dry_run and receipt["duplicate_content"] == receipt["would_insert"]:
            receipt["state"] = reason = "indexed_elsewhere_dedup"
    if not reason and receipt.get("quarantined_tail"):
        category, reason = ("readable" if dry_run else "accepted"), "quarantined_tail"
    receipt["class"] = category
    receipt["reason"] = (
        reason
        or {
            "readable": "validated-queue-candidates",
            "accepted": "queued-or-already-committed",
            "empty": "no-records",
            "filtered": "no-queue-candidates",
        }[category]
    )


def ingest_codex_manifest(
    manifest, db_path, *, dry_run=False, max_files=2, max_decoded_bytes=1048576, machine=None, start_at=0
):
    """Yield content-free receipts. Native header IDs never decide admission or dedup.

    An explicit manifest re-evaluates ALL bytes of its listed files, including offset-only
    files. It never resets a checkpoint. Receipts confirm complete validation
    and durable queue publication; interrupted publication can safely replay canonical ids.
    """
    if machine not in {None, "M1", "M4"}:
        raise ValueError("manifest-machine must be M1 or M4")
    if max_files <= 0 or max_decoded_bytes <= 0:
        raise ValueError("manifest budgets must be positive")
    registry = OffsetRegistry(Path(db_path).parent / "offsets.json")
    remaining = max_decoded_bytes
    touched = 0
    position = 0
    seen = set()
    with Path(manifest).open() as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            receipt = _new_receipt(line_number)
            try:
                expected = json.loads(raw)
                if not isinstance(expected, dict):
                    raise ValueError("manifest-row-is-not-an-object")
                if machine is not None:
                    row_machine = expected.get("machine")
                    if not isinstance(row_machine, str) or row_machine not in {"M1", "M4"}:
                        raise ValueError("invalid_machine")
                    if row_machine != machine:
                        receipt["state"] = "skipped_other_machine"
                        _classify_receipt(receipt, dry_run)
                        yield receipt
                        continue
                ordinal = position
                position += 1
                if ordinal < start_at:
                    continue
                path = Path(expected["physical_path"])
                logical = str(logical_jsonl_path(path))
                receipt.update(physical_path=str(path), logical_path=logical, stored_offset=registry.get(logical)[0])
                if (
                    not path.is_absolute()
                    or not str(path).endswith(".jsonl.zst")
                    or expected.get("logical_path") != logical
                ):
                    raise ValueError("invalid-manifest-path")
                if logical in seen:
                    raise ValueError("duplicate-manifest-path")
                seen.add(logical)
                if is_denylisted(str(path)) or is_denylisted(logical):
                    raise ValueError("ingest-denylist")
                if touched >= max_files or remaining <= 0:
                    raise ValueError("admission-budget")
                touched += 1
                _verify(path, expected)
                receipt["decodable"] = False
                preview = []
                plan = create_flush_callback(db_path, arbitrated=True, preview_chunks=preview)
                receipt["state"] = "decodable" if dry_run else "filtered"
                # Stream bounded decode windows; never accumulate a whole rollout.
                with (
                    _committed(db_path) if not dry_run else nullcontext(None) as committed,
                    closing(_read_entries(path, remaining, receipt)) as reader,
                ):
                    if not dry_run:
                        receipt["live_chunks_at_start"] = _source_chunks(committed, logical)
                    for entries in reader:
                        _plan_window(plan, entries, preview, receipt, committed, dry_run)
                _verify(path, expected)
                receipt["decodable"] = True
                if dry_run and not receipt["would_insert"]:
                    receipt["state"] = "empty" if receipt["records"] == 0 else "filtered"
                if not dry_run:
                    # The live watcher owns offsets.json. Queue events carry decoded
                    # source_end_offset; explicit replay never races or rewrites its registry.
                    receipt["state"] = (
                        "queued" if receipt["queued"] else ("committed" if receipt["would_insert"] else "filtered")
                    )
            except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
                _record_error(receipt, error)
            _classify_receipt(receipt, dry_run)
            remaining -= receipt["decoded_bytes"]
            yield receipt


def _new_receipt(line_number):
    return dict(
        manifest_line=line_number,
        physical_path=None,
        logical_path=None,
        decodable=None,
        records=0,
        decoded_bytes=0,
        decoded_bytes_total=None,
        would_insert=0,
        queued=0,
        live_chunks_at_start=0,
        duplicate_content=0,
        duplicate_matches=[],
        filtered=0,
        skip_reason="",
        state="held",
        count_kind="queue_candidates_before_DB_dedup",
        stored_offset=0,
    )


def _record_error(receipt, error):
    receipt["skip_reason"] = str(error)
    held = str(error) in {
        "admission-budget",
        "decoded-byte-budget",
        "queue-backpressure",
        "provenance-unresolved",
    }
    receipt["state"] = "partially_queued" if receipt["queued"] else ("held" if held else "error")
    if held and not receipt["decodable"]:
        receipt["decodable"] = None
    if receipt["decodable"] is None and receipt["decoded_bytes"]:
        receipt["decodable"] = False


def _source_chunks(db, source_file):
    return (
        db.execute(
            f"SELECT COUNT(*) FROM chunks WHERE source_file = ? AND {_active_clause(db)}", (source_file,)
        ).fetchone()[0]
        if db
        else 0
    )
