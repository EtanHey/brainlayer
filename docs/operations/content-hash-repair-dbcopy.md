# content_hash repair: DB-copy acceptance runbook (#623)

**Status: LOCAL / UNRUN.** Nobody has run this against a copy of the real database yet.
The synthetic proof is `tests/test_content_hash_repair_dbcopy.py`.

This runbook checks `python -m brainlayer.content_hash_repair` against a **copy**. That tool
ships with #1637. Passing it does **not** authorise touching the live DB:

- The tool refuses the live DB by inode.
- It also refuses whatever `BRAINLAYER_DB` names.
- It has no `--allow-live` flag.

## 0. Preconditions

- **Interpreter.** Use a worktree venv that contains #1637, or the keg python once a release
  contains it (`scripts/release_tag_contains.py`). Never use bare `python3`: on the M4 it
  imports the repo root's live working tree through `_brainlayer.pth`.
- **Copy method.** Take a consistent snapshot of the live DB:
  - preferably with BrainBar's `brain_backup_vacuum_into`;
  - or with `sqlite3 "file:$LIVE?mode=ro" ".backup '$COPY'"`.

  Never `cp` only the main file: uncheckpointed WAL frames would be missing from the copy.
- **Copy location.** Keep `$COPY` outside `~/.local/share/brainlayer`, and do **not** export
  `BRAINLAYER_DB=$COPY`. The tool treats that path as live and refuses it, by design.
- **Do not upload the copy anywhere.** It is personal data.

```bash
COPY=/path/outside/brainlayer/brainlayer-623-copy.db
PY=/path/to/worktree/.venv/bin/python   # or the keg python, once the release contains #1637
```

## 1. Read-only precheck: record every output

```bash
sqlite3 -readonly "$COPY" "PRAGMA quick_check;"                       # expect: ok
sqlite3 -readonly "$COPY" "SELECT COUNT(*) FROM pragma_table_info('chunks') WHERE name='content_hash';"  # expect: 1
# Any trigger that could fire on a content_hash UPDATE. Expect no rows. Only the header (before
# BEGIN) is read: INSERT triggers' bodies contain "ON CONFLICT ... DO UPDATE" and are not UPDATE triggers.
sqlite3 -readonly "$COPY" "SELECT name FROM (SELECT name, substr(sql, 1, instr(upper(sql), 'BEGIN')) AS h
  FROM sqlite_master WHERE type='trigger' AND tbl_name='chunks')
  WHERE h LIKE '%UPDATE%' AND (h NOT LIKE '%UPDATE OF%' OR h LIKE '%content_hash%');"
sqlite3 -readonly "$COPY" "SELECT COUNT(*), SUM(content_hash IS NULL) FROM chunks
  WHERE substr(content,1,28)='[Merged duplicate originals]';"      # merged rows, merged NULL-hash rows
```

Fingerprint everything the repair must **not** change. That is every `chunks` column except
`content_hash`, the FTS row count, and the vector row count:

```bash
cat > /tmp/fp623.py <<'EOF'
import hashlib, sqlite3, sys
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
cols = [r[1] for r in c.execute("PRAGMA table_info(chunks)") if r[1] != "content_hash"]
h = hashlib.sha256()
for row in c.execute(f"SELECT {', '.join(cols)} FROM chunks ORDER BY id"):
    h.update(repr(row).encode())
fts = c.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
vec = c.execute("SELECT COUNT(*) FROM chunk_vectors_rowids").fetchone()[0]
print(h.hexdigest(), "fts", fts, "vectors", vec)
EOF
"$PY" -I /tmp/fp623.py "$COPY" | tee /tmp/fp623.before
```

## 2. Dry-run: read-only

```bash
"$PY" -m brainlayer.content_hash_repair --db "$COPY" | tee /tmp/623.dry1.json
```

Record `stale`, `merged_rows_scanned` and `more_remaining`. The repair (e) census counted
2,256 stale rows across **all** content changes. Merged-marker rows are a subset, so
`stale` should be at most 2,256. A larger number is a finding: stop and report it.

## 3. Explicit apply: bounded runs until nothing is stale

```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  "$PY" -m brainlayer.content_hash_repair --db "$COPY" --apply --batch-size 500 --max-rows 5000 \
    | tee "/tmp/623.apply$i.json"
  grep -q '"stale": 0' "/tmp/623.apply$i.json" && break
done
```

On a quiescent copy, every run must report `skipped_identity_changed: 0`. A non-zero count
means something else is writing to the copy: stop.

## 4. Output checks

```bash
"$PY" -m brainlayer.content_hash_repair --db "$COPY" | tee /tmp/623.dry2.json   # expect "stale": 0
"$PY" -I /tmp/fp623.py "$COPY" | diff /tmp/fp623.before -                        # expect: no diff
sqlite3 -readonly "$COPY" "PRAGMA integrity_check;"                               # expect: ok
```

Spot-check the `sample_ids` from step 2. For each one, `content_hash` must equal
`sha256(content.strip())`:

```bash
"$PY" -I -c 'import hashlib,sqlite3,sys,json
c=sqlite3.connect(f"file:{sys.argv[1]}?mode=ro",uri=True)
for i in json.load(open(sys.argv[2]))["sample_ids"]:
    t,h=c.execute("SELECT content,content_hash FROM chunks WHERE id=?",(i,)).fetchone()
    print(i, h==hashlib.sha256(t.strip().encode()).hexdigest())' "$COPY" /tmp/623.dry1.json
```

## 5. Receipt and stop conditions

- **Receipt.** Put the JSON outputs, the two fingerprints and the counts in a small text
  receipt under `docs.local/`. Never include the copy or any row content.
- **Stop and report**, without retrying around the check, on any of these:
  - a precheck trigger row;
  - `stale` above 2,256;
  - `skipped_identity_changed` above 0;
  - a fingerprint diff;
  - an integrity result other than `ok`;
  - any refusal.

## What a pass does NOT cover

- **Production.** Applying this to the live DB is a separate lead decision. The tool cannot
  do it as written, because it has no live mode.
- **Order of operations.** The fixed merge writer must be **deployed** before any production
  application. Until then, the old writer keeps leaving new stale rows behind.
- **The other content changes.** Stale hashes left by non-merge content changes (the rest of
  the repair (e) census) are out of scope. This tool leaves them alone.
