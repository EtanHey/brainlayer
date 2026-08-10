# Wave 3 source-class review disposition

Status: COMPLETE — the first CodeRabbit CLI attempt returned `rate_limit`; the retry reviewed the exact branch diff against `origin/main` on 2026-08-10.

Fallback: completed the CodeRabbit red-team and blue-team prompt reviews against the uncommitted diff.

- HIGH: none.
- MEDIUM: ordinary workflow and Cursor coding-agent rows retained the interim `ISOLATE` policy, contradicting the approved source-class visibility contract — FIXED by making both searchable `KEEP` sources and updating end-to-end watcher tests.
- MEDIUM: idempotent migration reruns accepted a different code SHA than the recorded ledger entry — FIXED by rejecting SHA mismatch and adding a regression test.
- LOW: historical recon rows without readable source JSONL needed a stable path signature — FIXED by recognizing `brain-worker`, `session-miner`, and `weave` path segments under `subagents`.

Verification: focused policy, migration, watcher, queue/drain, upsert, and retro-quarantine suites pass after dispositions.

The exact-diff retry reported three additional valid findings:

- MAJOR: rollback did not relocate canonical WAL/SHM sidecars before installing the restored DB — FIXED in the runbook, including an assertion that neither canonical sidecar remains.
- MAJOR: malformed JSON or invalid distribution in the idempotent migration branch leaked its SQLite connection — FIXED with unconditional cleanup and a RED-first malformed-ledger regression test.
- MINOR: `sqlite3.total_changes` overcounted logical FTS deletions and omitted `chunk_fts_rowids` from the receipt — FIXED with count-before-delete across all four tables and RED-first receipt assertions.

Verification after the retry: focused migration suite passes 6/6; runbook shell blocks parse with `bash -n`; final reviewed migration commit is rehearsed separately and pinned by exact SHA in both ledgers.

## Hosted Codex and pair-review MUST_FIX disposition

- FIXED: the fleet's executing Swift path now applies the same default visibility contract in `BrainDatabase` FTS and candidate searches. Desktop and brain-worker rows are hidden, NULL remains visible, and schema-absent compatibility keeps v1.5.6 safe to deploy before migration.
- FIXED: source role is derived from durable attribution/provenance and exact paths, never from an arbitrary content mention. Historical `provenance_class=recon-agent` remains authoritative as brain-worker even if its source JSONL is no longer readable.
- FIXED: brain-worker migration cleanup removes FTS, float-vector, and binary-vector index rows. The migrator and runbook verifier explicitly load vec0 before touching the virtual vector tables.
- FIXED: KNN candidate expansion counts only source-class-eligible vector rows, for both float and binary arms, so hidden rows do not exhaust the overfetch budget.
- FIXED: source-class values are exact-taxonomy validated at index, vector upsert, drain, and replay boundaries; replay now restores the class.
- FIXED: watcher direct INSERT guards on column presence. The LIVE runbook requires v1.5.6 install, restart, executing-binary path/hash/capability probes, then writer stop, backup, and migration on each host. Migrating before that deploy proof is forbidden.
- FIXED: writer inventory covers `com.brainlayer.*` plus `com.mcplayer.*`; the stop/assert list includes gemini-loopback, p0-counter, jsonl-backup, and the BrainLayer proxy/TCP bridge.
- FIXED: rollback authority is proven by one persistent read-only connection comparing `PRAGMA data_version` before backup and only after the remaining daemon is stopped. Pipeline failures propagate under `pipefail`.
- FIXED: pause-sentinel parsing matches the exact active label; worktree/package project parsing no longer mislabels fleet coordination; the aggregate desktop verifier cannot pass on a single convenient row.
- NOT ADOPTED: classifying role from migration-row content. The project letter requires class at ingest, and arbitrary content can mention a role without being that role; durable attribution/provenance is the authoritative signal.
- DEFERRED (nonblocking N3): retry attempts still overwrite the fixed migration-event row rather than retaining a separate per-attempt audit history. The successful immutable schema ledger and final success event remain exact-SHA pinned; adding an attempt-history table is outside this migration's gate.

Final real-copy evidence at `3964412f8291a083150a424e38df08ece817783d`: 744,335 rows preserved; 84 brain-workers retained as class but removed from all search indexes; six-bucket visibility/expansion green; aggregate desktop audit sampled 72 tokens with no leak; `PRAGMA quick_check=ok`; exact-SHA idempotent rerun green.

## Final local CodeRabbit exact-diff disposition

- MAJOR: Swift exact-ID search bypassed the searchable clause — FIXED. Default `search()` exact-ID lookups now hide desktop and brain-worker; `expandChunk` remains the explicit exact-expansion path for every class.
- MAJOR: the BrainBar helper fast profile suppressed KNN retry even when hidden source-class rows exhausted the initial 400 candidates — FIXED. It preserves the bounded first attempt and permits only the existing source-class-aware retry.
- MAJOR: a 60-second hybrid cache entry could survive a cross-process NULL-to-desktop class update — FIXED. Every entry records SQLite `data_version` and is returned only when the current value matches.
- MINOR: raw MCP smoke receipts inherited caller umask — FIXED. New and existing output files are forced to mode 0600 before the response is written; a real socket smoke verified the mode.
- MINOR: the prose allowed later v1.5.6+ releases while preflight required exact 1.5.6 — FIXED by making this scheduled Tuesday run require exact v1.5.6 consistently.
