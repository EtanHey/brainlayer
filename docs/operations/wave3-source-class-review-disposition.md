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
