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
Linked records persist both `t3-app-session` provenance and `desktop` source class,
including queued writes. Desktop rows stay hidden from default retrieval and
remain accessible by exact ID. Memory-reader class exclusion has precedence.

Missing selected schema, malformed references or headers, and unlinked T3-origin
headers raise an alarm and defer affected entries. Deferred entries retain their
text and cannot advance watcher watermarks; ordinary Claude continues. Recovery
retries buffered entries against a new linkage snapshot. Ordinary structurally
unlinked CLI headers remain CLI. Legacy selection and filename behavior remain
compatible when V2 is not selected. Changing only the DB filename is insufficient.

This changes new classification. Existing wrong-class rows require a separately
verified exact-ID allowlist and a rollback ledger for both class fields; do not
run broad source-file retagging or rewind checkpoints. Rolling configuration back
does not delete already persisted messages or restore a whole database.
