"""Copy-only, direct realtime writer rehearsal for #1661."""

import argparse
import hashlib
import json
import os
import pwd
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


def safe(path):
    path = Path(path).expanduser().resolve()
    forbidden = (Path(pwd.getpwuid(os.getuid()).pw_dir) / ".local/share/brainlayer").resolve()
    roots = [forbidden, Path("/tmp"), Path("/private/tmp"), Path("/var/folders")]
    if os.environ.get("TMPDIR"):
        roots.append(Path(os.environ["TMPDIR"]).expanduser())
    if any(path.is_relative_to(root.resolve()) for root in roots):
        raise ValueError("refusing production or temporary path")
    return path


def snapshot(db):
    with closing(sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)) as conn:
        rows = conn.execute("SELECT id, project, source_file, content_hash FROM chunks ORDER BY id").fetchall()
        return {
            "integrity": conn.execute("PRAGMA integrity_check").fetchall(),
            "foreign_keys": conn.execute("PRAGMA foreign_key_check").fetchall(),
            "count": len(rows),
            "hash": digest(rows),
        }, rows


def digest(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


def rehearse(db, source_dir):
    db, source_dir = safe(db), safe(source_dir)
    if not db.is_file():
        raise ValueError("existing database copy required")
    if source_dir.is_relative_to(db.parent) or db.parent.is_relative_to(source_dir):
        raise ValueError("source directory must be outside the database copy directory")
    source_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    scratch = None
    environment = {}
    previous = {}
    try:
        scratch = Path(tempfile.mkdtemp(prefix="rehearse-1661-", dir=db.parent))
        environment = {
            "BRAINLAYER_WRITER_TELEMETRY_PATH": str(scratch / "writer-telemetry.jsonl"),
            "BRAINLAYER_WRITER_HEARTBEAT_DIR": str(scratch),
            "BRAINLAYER_WRITER_PIDFILE_DIR": str(scratch),
            "BRAINLAYER_QUEUE_DIR": str(scratch / "queue"),
        }
        previous = {key: os.environ.get(key) for key in environment}
        os.environ.update(environment)
        _run(db, source_dir, environment)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        try:
            if scratch is not None:
                shutil.rmtree(scratch)
        finally:
            shutil.rmtree(source_dir)


def _run(db, source_dir, environment):
    before, original = snapshot(db)
    print(json.dumps({"before": before}), flush=True)
    assert before["integrity"] == [("ok",)] and not before["foreign_keys"]
    from brainlayer.watcher_bridge import create_flush_callback

    entries = []
    for project in ("rehearsal-alpha", "rehearsal-beta"):
        path = source_dir / "projects" / project / "cse-rehearsal-identical-stem.jsonl"
        path.parent.mkdir(parents=True)
        entry = {
            "type": "user",
            "cwd": f"/rehearsal/{project}",
            "timestamp": "2026-11-07T12:00:00Z",
            "message": {
                "role": "user",
                "content": "Please explain how the realtime watcher preserves project attribution when identical substantive messages appear in separate coding sessions.",
            },
        }
        path.write_text(json.dumps(entry) + "\n")
        entries.append(dict(entry, _source_file=str(path), _line_end_offset=path.stat().st_size))
    flush = create_flush_callback(db, arbitrated=False)
    try:
        counts = []
        for _ in range(2):
            flush(entries)
            after, rows = snapshot(db)
            preserved = [r for r in rows if r[0] in {r[0] for r in original}]
            assert digest(preserved) == before["hash"], "pre-existing rows changed"
            assert after["integrity"] == [("ok",)] and not after["foreign_keys"]
            counts.append(after["count"])
            added = [r for r in rows if r[2] in {e["_source_file"] for e in entries}]
            assert len(added) == 2 and len({r[1] for r in added}) == 2, "cross-project collision"
        assert counts[0] == counts[1] == before["count"] + 2, "non-idempotent ingest"
        print(
            json.dumps(
                {
                    "before": before,
                    "counts": counts,
                    "result": "PASS",
                    "db": str(db),
                    "source_dir": str(source_dir),
                    "environment": environment,
                }
            )
        )
    finally:
        flush.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--source-dir", required=True)
    args = parser.parse_args()
    rehearse(args.db, args.source_dir)
