"""Copy-only, direct realtime writer rehearsal for #1661."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def safe(path):
    path = Path(path).expanduser().resolve()
    forbidden = (Path.home() / '.local/share/brainlayer').resolve()
    if forbidden in (path, *path.parents) or any(p in (path, *path.parents) for p in (Path('/tmp'), Path('/private/tmp'))):
        raise ValueError('refusing production or temporary path')
    return path


def snapshot(db):
    with sqlite3.connect(f'{db.as_uri()}?mode=ro', uri=True) as conn:
        rows = conn.execute('SELECT id, project, source_file, content_hash FROM chunks ORDER BY id').fetchall()
        return {'integrity': conn.execute('PRAGMA integrity_check').fetchall(),
                'foreign_keys': conn.execute('PRAGMA foreign_key_check').fetchall(),
                'count': len(rows), 'hash': digest(rows)}, rows


def digest(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


def rehearse(db, source_dir):
    db, source_dir = safe(db), safe(source_dir)
    if not db.is_file():
        raise ValueError('existing database copy required')
    source_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    before, original = snapshot(db)
    print(json.dumps({'before': before}), flush=True)
    assert before['integrity'] == [('ok',)] and not before['foreign_keys']
    from brainlayer.watcher_bridge import create_flush_callback
    entries = []
    for project in ('rehearsal-alpha', 'rehearsal-beta'):
        path = source_dir / 'projects' / project / 'cse-rehearsal-identical-stem.jsonl'
        path.parent.mkdir(parents=True)
        entry = {'type': 'user', 'cwd': f'/rehearsal/{project}',
                 'timestamp': '2026-11-07T12:00:00Z',
                 'message': {'role': 'user', 'content': 'Please explain how the realtime watcher preserves project attribution when identical substantive messages appear in separate coding sessions.'}}
        path.write_text(json.dumps(entry) + '\n')
        entries.append(dict(entry, _source_file=str(path), _line_end_offset=path.stat().st_size))
    flush = create_flush_callback(db, arbitrated=False)
    counts = []
    for _ in range(2):
        flush(entries)
        after, rows = snapshot(db)
        preserved = [r for r in rows if r[0] in {r[0] for r in original}]
        assert digest(preserved) == before['hash'], 'pre-existing rows changed'
        assert after['integrity'] == [('ok',)] and not after['foreign_keys']
        counts.append(after['count'])
        added = [r for r in rows if r[2] in {e['_source_file'] for e in entries}]
        assert len(added) == 2 and len({r[1] for r in added}) == 2, 'cross-project collision'
    assert counts[0] == counts[1] == before['count'] + 2, 'non-idempotent ingest'
    print(json.dumps({'before': before, 'counts': counts, 'result': 'PASS'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--source-dir', required=True)
    args = parser.parse_args()
    rehearse(args.db, args.source_dir)
