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

Existing `t3-thread` and `t3-app-session` rows can have their source class re-derived
on a quiesced offline DB copy with `scripts/migrate_source_class.py --t3-only --db
<copy> --git-sha <40-character-SHA> --actor <operator>`. This is read-only by default;
add `--apply` to write the copy. It reports before/after counts, commits 5,000–10,000
rows per batch, and checkpoints before, every three batches, and after. It refuses
the canonical DB and its aliases and preserves provenance and all other sources.
Changing provenance itself still requires a separately verified exact-ID allowlist
and rollback ledger; do not run broad source-file retagging or rewind checkpoints.
