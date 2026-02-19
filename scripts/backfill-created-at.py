#!/usr/bin/env python3
"""Backfill created_at column for existing chunks.

Sources (in priority order):
  1. WhatsApp/YouTube: metadata->timestamp (already stored in JSON)
  2. Claude Code: first message timestamp from source JSONL file
  3. Archive lookup: find archived JSONL by UUID in ~/.claude-archive/
  4. Manifest lookup: archiver manifest has originalMtime per session
  5. Fallback: source file modification time

Archive locations searched:
  - ~/.claude/projects/         (current sessions)
  - ~/.claude-archive/          (archived by session-archiver)
  - ~/Library/Mobile Documents/com~apple~CloudDocs/golem-archives/
  - ~/Library/Mobile Documents/com~apple~CloudDocs/golems-backups/

Usage:
    python3 scripts/backfill-created-at.py [--dry-run]
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.vector_store import VectorStore

# All known locations where session JSONL files might live
ARCHIVE_SEARCH_PATHS = [
    Path.home() / ".claude" / "projects",
    Path.home() / ".claude-archive",
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "golem-archives",
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "golems-backups",
]


def build_uuid_index():
    """Build UUID → file path index from all known archive locations."""
    uuid_to_path = {}
    manifest_timestamps = {}

    for base_dir in ARCHIVE_SEARCH_PATHS:
        if not base_dir.exists():
            continue
        for root, dirs, files in os.walk(str(base_dir)):
            for f in files:
                full_path = os.path.join(root, f)
                if f.endswith(".jsonl"):
                    uuid = f.replace(".jsonl", "")
                    uuid_to_path[uuid] = full_path
                elif f == "manifest.json":
                    try:
                        with open(full_path) as mf:
                            manifest = json.load(mf)
                            for s in manifest.get("sessions", []):
                                ts = s.get("firstMessageTimestamp") or s.get("originalMtime")
                                if ts:
                                    manifest_timestamps[s["uuid"]] = ts
                    except (json.JSONDecodeError, IOError):
                        pass

    print(f"  UUID index: {len(uuid_to_path):,} JSONL files, {len(manifest_timestamps):,} manifest entries")
    return uuid_to_path, manifest_timestamps


def extract_timestamp_from_jsonl(filepath):
    """Extract first message timestamp from a JSONL file."""
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                ts = data.get("timestamp")
                if ts:
                    return ts
    except (json.JSONDecodeError, IOError, OSError):
        pass

    # Fall back to file mtime
    try:
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


def backfill_from_metadata(store):
    """Backfill created_at from metadata->timestamp (WhatsApp, YouTube)."""
    cursor = store.conn.cursor()
    cursor.execute("""
        UPDATE chunks
        SET created_at = json_extract(metadata, '$.timestamp')
        WHERE created_at IS NULL
          AND json_valid(metadata) = 1
          AND json_extract(metadata, '$.timestamp') IS NOT NULL
    """)
    updated = store.conn.changes()
    print(f"  Metadata backfill: {updated:,} chunks updated")
    return updated


def backfill_from_jsonl(store, uuid_to_path, manifest_timestamps):
    """Backfill created_at from JSONL files — original path, archive, or manifest."""
    cursor = store.conn.cursor()

    files = list(cursor.execute("""
        SELECT DISTINCT source_file, COUNT(*) as cnt
        FROM chunks
        WHERE created_at IS NULL AND source_file IS NOT NULL
        GROUP BY source_file
        ORDER BY cnt DESC
    """))

    total_updated = 0
    from_original = 0
    from_archive = 0
    from_manifest = 0
    files_missing = 0

    for source_file, chunk_count in files:
        if not source_file:
            continue

        timestamp = None
        source_type = None

        # Try 1: Original path
        if os.path.exists(source_file):
            timestamp = extract_timestamp_from_jsonl(source_file)
            source_type = "original"

        # Try 2: UUID lookup in archive
        if not timestamp:
            basename = os.path.basename(source_file)
            uuid = basename.replace(".jsonl", "")
            if uuid in uuid_to_path:
                timestamp = extract_timestamp_from_jsonl(uuid_to_path[uuid])
                source_type = "archive"

            # Try 3: Manifest timestamp
            if not timestamp and uuid in manifest_timestamps:
                timestamp = manifest_timestamps[uuid]
                source_type = "manifest"

        if not timestamp:
            files_missing += 1
            continue

        cursor.execute("""
            UPDATE chunks SET created_at = ?
            WHERE source_file = ? AND created_at IS NULL
        """, [timestamp, source_file])
        updated = store.conn.changes()
        total_updated += updated

        if source_type == "original":
            from_original += 1
        elif source_type == "archive":
            from_archive += 1
        elif source_type == "manifest":
            from_manifest += 1

        processed = from_original + from_archive + from_manifest
        if processed % 200 == 0:
            print(f"    [{processed}/{len(files)}] {total_updated:,} chunks updated")

    total_files = from_original + from_archive + from_manifest
    print(f"  JSONL backfill: {total_updated:,} chunks from {total_files:,} files")
    print(f"    From original path: {from_original:,}")
    print(f"    From archive: {from_archive:,}")
    print(f"    From manifest: {from_manifest:,}")
    print(f"    Missing: {files_missing:,}")
    return total_updated


def backfill_from_mtime(store):
    """Use source file modification time for remaining NULL chunks (if file exists)."""
    cursor = store.conn.cursor()

    remaining = list(cursor.execute("""
        SELECT DISTINCT source_file
        FROM chunks
        WHERE created_at IS NULL AND source_file IS NOT NULL
    """))

    total_updated = 0
    for (source_file,) in remaining:
        if not source_file or not os.path.exists(source_file):
            continue
        try:
            mtime = os.path.getmtime(source_file)
            ts = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            cursor.execute("""
                UPDATE chunks SET created_at = ?
                WHERE source_file = ? AND created_at IS NULL
            """, [ts, source_file])
            total_updated += store.conn.changes()
        except OSError:
            continue

    print(f"  Mtime backfill: {total_updated:,} chunks updated")
    return total_updated


def backfill_from_interpolation(store):
    """Estimate created_at from nearby chunks using rowid ordering.

    The indexer processes sessions sequentially, so chunks with nearby rowids
    were indexed around the same time. We use chunks WITH known created_at
    as anchor points and interpolate dates for chunks WITHOUT.

    These are marked as estimates (prefixed with ~) for transparency.
    """
    import bisect

    cursor = store.conn.cursor()

    # Build "date ladder" from chunks that have created_at
    ladder = list(cursor.execute("""
        SELECT rowid, created_at FROM chunks WHERE created_at IS NOT NULL ORDER BY rowid
    """))

    if not ladder:
        print("  No anchor points available for interpolation")
        return 0

    ladder_rowids = [r for r, _ in ladder]
    ladder_dates = [d for _, d in ladder]

    print(f"  Using {len(ladder):,} anchor points (rowid {ladder_rowids[0]:,} - {ladder_rowids[-1]:,})")

    # Get source_file groups that still need dates
    source_groups = list(cursor.execute("""
        SELECT source_file, MIN(rowid) as min_r, MAX(rowid) as max_r, COUNT(*) as cnt
        FROM chunks
        WHERE created_at IS NULL AND source_file IS NOT NULL
        GROUP BY source_file
        ORDER BY min_r
    """))

    total_updated = 0
    for sf, min_r, max_r, cnt in source_groups:
        mid_r = (min_r + max_r) // 2
        idx = bisect.bisect_left(ladder_rowids, mid_r)

        if idx == 0:
            est_date = ladder_dates[0]
        elif idx >= len(ladder):
            est_date = ladder_dates[-1]
        else:
            est_date = ladder_dates[idx - 1]

        # Use the date portion only (more honest than fabricating a time)
        date_only = est_date[:10] if len(est_date) >= 10 else est_date
        estimated_ts = f"{date_only}T00:00:00+00:00"

        cursor.execute("""
            UPDATE chunks SET created_at = ?
            WHERE source_file = ? AND created_at IS NULL
        """, [estimated_ts, sf])
        total_updated += store.conn.changes()

    print(f"  Interpolation backfill: {total_updated:,} chunks estimated from nearby anchors")
    return total_updated


def main():
    dry_run = "--dry-run" in sys.argv

    print(f"Database: {DEFAULT_DB_PATH}")
    store = VectorStore(DEFAULT_DB_PATH)

    cursor = store.conn.cursor()
    total = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
    has_date = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE created_at IS NOT NULL"))[0][0]
    print(f"Total chunks: {total:,}")
    print(f"Already have created_at: {has_date:,}")
    print(f"Need backfill: {total - has_date:,}")

    if dry_run:
        print("\n[DRY RUN] Would backfill. Run without --dry-run to execute.")
        return

    print("\nBuilding UUID index from all archive locations...")
    uuid_to_path, manifest_timestamps = build_uuid_index()

    print("\n1/4 Backfilling from metadata (WhatsApp/YouTube)...")
    backfill_from_metadata(store)

    print("\n2/4 Backfilling from JSONL timestamps (original + archive + manifest)...")
    backfill_from_jsonl(store, uuid_to_path, manifest_timestamps)

    print("\n3/4 Backfilling from file mtimes...")
    backfill_from_mtime(store)

    print("\n4/4 Estimating dates from nearby chunks (rowid interpolation)...")
    backfill_from_interpolation(store)

    # Final stats
    has_date = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE created_at IS NOT NULL"))[0][0]
    still_null = total - has_date
    print(f"\nDone! {has_date:,}/{total:,} chunks have created_at ({has_date*100/total:.1f}%)")
    if still_null > 0:
        print(f"Still missing: {still_null:,} chunks")


if __name__ == "__main__":
    main()
