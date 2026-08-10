# Wave 3 source-class review disposition

Status: SKIPPED — CodeRabbit CLI returned `rate_limit` (free OSS reset: 10 minutes) on 2026-08-10.

Fallback: completed the CodeRabbit red-team and blue-team prompt reviews against the uncommitted diff.

- HIGH: none.
- MEDIUM: ordinary workflow and Cursor coding-agent rows retained the interim `ISOLATE` policy, contradicting the approved source-class visibility contract — FIXED by making both searchable `KEEP` sources and updating end-to-end watcher tests.
- MEDIUM: idempotent migration reruns accepted a different code SHA than the recorded ledger entry — FIXED by rejecting SHA mismatch and adding a regression test.
- LOW: historical recon rows without readable source JSONL needed a stable path signature — FIXED by recognizing `brain-worker`, `session-miner`, and `weave` path segments under `subagents`.

Verification: focused policy, migration, watcher, queue/drain, upsert, and retro-quarantine suites pass after dispositions.
