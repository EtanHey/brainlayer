# Explicit T3 V2 provider provenance

Status: source configuration contract; deployment requires reviewed copy validation.

The watcher defaults to legacy T3 linkage. After installing this implementation,
select V2 explicitly in its private environment:

```dotenv
BRAINLAYER_T3_STATE_DB=/Users/example/.t3/userdata/statev2.sqlite
BRAINLAYER_T3_PROJECTION_VERSION=2
```

The selected V2 provider projection links Codex `nativeThreadRef` and
`nativeConversationHeadRef` identities by structural driver metadata, independently of provider instance names.
The transcript's first `session_meta` record supplies its native identity; fork
filename suffixes, task text and working directories do not establish linkage.
Headers are bounded to 1 MiB and cached by file identity within each flush.
Linked records persist both `t3-app-session` provenance and `cli-agent` source class,
including queued writes. T3 is a coding-agent surface visible in default retrieval
(Etan's 2026-10-11 ruling). Claude Desktop, ChatGPT and Gemini desktop rows remain
hidden. Memory-reader class exclusion has precedence.

Missing selected schema, malformed references or headers, and unlinked T3-origin
headers raise an alarm and defer affected entries. Deferred entries retain their
text and cannot advance watcher watermarks; ordinary Claude continues. Recovery
retries buffered entries against a new linkage snapshot. Ordinary structurally
unlinked CLI headers remain CLI. Legacy selection and filename behavior remain
compatible when V2 is not selected. Changing only the DB filename is insufficient.

Existing desktop rows are corrected when `source='t3'`, `source_file` contains
`/.codex/sessions/`, or provenance is `t3-thread`/`t3-app-session`. Provenance can
instead hold epistemic labels such as `AGENT-INFERENCE`, so it cannot be the sole
selector. Sources `chatgpt`, `claude_desktop`, and `gemini_desktop` are excluded;
other source classes are preserved. The correction sets `cli-agent` directly.

Run `scripts/migrate_source_class.py --t3-only --db <copy> --git-sha
<40-character-SHA> --actor <operator>`. This uses SQLite `mode=ro` by default,
including on the canonical DB without a gate; add `--apply` to write a copy.
Canonical writes require `BRAINLAYER_OFFLINE_MIGRATOR_GATED_SWAP=1`. Canonical
symlinks/hardlinks use the same gate and resolve to the canonical WAL/SHM paths.
The command reports before/after counts, commits 5,000–10,000 rows per batch,
and FULL checkpoints before, every three batches, and after.

Each committed batch writes a `migration_events` row in the same transaction.
Its details contain the exact changed IDs, count, prior class (`desktop`), result
class, run ID, and SHA-256 of the sorted IDs encoded as compact JSON. The event
also records the migration commit, actor, and DB path. Ledger failure rolls back
that batch; earlier committed batches remain auditable and a rerun completes the
remaining candidates. A zero-candidate repeat creates no events. For supervised
rollback, stop writers again, verify a backup, check the recorded count/hash, and
restore only those IDs still labeled `cli-agent` to the recorded prior class in
a transaction; investigate any mismatch before changing rows.

## Supervised in-place correction

1. Stop the watcher, BrainBar daemon, and any legacy enrichment writers. Verify
   that no writer remains. Keep retired enrichment services stopped afterward.
2. Verify an existing backup, its integrity, and its chunk count before continuing.
   On a storage-constrained Mac, rehearse on an APFS clone of that backup; never
   start a fresh canonical `VACUUM INTO` without enough free space.
3. Run `PRAGMA wal_checkpoint(TRUNCATE)` on the canonical DB with the appropriate
   SQLite extensions loaded. Require the checkpoint's busy result to be zero.
4. Run the read-only `--t3-only` dry-run against the canonical DB. Review the
   candidate count and confirm that true desktop sources are excluded.
5. Run the same command with the supervised gate and `--apply`:
   ```sh
   BRAINLAYER_OFFLINE_MIGRATOR_GATED_SWAP=1 python scripts/migrate_source_class.py \
     --t3-only --apply --db <canonical-db> --git-sha <40-character-SHA> --actor <operator>
   ```
6. Repeat the gated apply and require zero candidates and zero updates. With a
   read-only connection, independently check the remaining candidate count:
   ```sql
   SELECT COUNT(*) FROM chunks WHERE source_class = 'desktop'
     AND (source = 't3' OR source_file LIKE '%/.codex/sessions/%'
          OR provenance_class IN ('t3-thread', 't3-app-session'))
     AND source NOT IN ('chatgpt', 'claude_desktop', 'gemini_desktop');
   ```
   Require zero and verify the ledger and preserved true-desktop counts.
7. Restart BrainBar and the watcher with the reviewed installed classifier.
   Use normal `brain_search`, without hidden-class opt-in, to retrieve a known
   T3 chunk. Record that installed-runtime proof separately from copy tests.

Changing provenance itself still requires a separately verified exact-ID allowlist
and rollback ledger; do not run broad source-file retagging or rewind checkpoints.
