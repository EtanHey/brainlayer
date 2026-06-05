# DB Shrink FTS5 + Content Dedup Plan

## Constraints

- Never write to `~/.local/share/brainlayer/brainlayer.db` unless an explicit live override is passed.
- Materialize and mutate only temp snapshot copies.
- Benchmark the same qrels before and after migration; do not ship if quality regresses beyond noise.

## Snapshot Source

- Drive folder: `brainlayer-db`
- File: `2026-06-01-pre-contentclass-apply-014318.db.gz`
- Drive file id: `1F-K1dxLb81fpGy-Fa6I1SSkTz-7VVLsx`
- Local matching gzip: `/Users/etanheyman/.local/share/brainlayer/backups/2026-06-01-pre-contentclass-apply-014318.db.gz`
- Temp baseline: `/tmp/brainlayer-db-shrink/baseline.db` (read-only)
- Temp migration copy: `/tmp/brainlayer-db-shrink/after.db`

## Implementation

1. Add `brainlayer.db_shrink` with parameterized APIs for:
   - live-path guard with `--i-know-this-is-live`;
   - FTS table sizing and counts;
   - FTS migration to a compact single trigram table;
   - exact normalized-content duplicate analysis;
   - physical duplicate delete with canonical aliases and direct reference repoints.
2. Add `scripts/db_shrink_migrate.py` as the operator CLI.
3. Add `scripts/db_shrink_eval.py` to run before/after benchmarks and persist `eval_results/db-shrink-2026-06-01.json`.
4. Update search/vector-store behavior only as needed to avoid recreating the removed redundant FTS table.

## Tests

1. Guard refuses the canonical DB path without `--i-know-this-is-live`.
2. FTS migration converts a dual-table FTS fixture into compact single-trigram mode and keeps counts synced.
3. Content dedup keeps protected/qrel chunk IDs canonical, writes aliases, repoints direct references, transfers tags/entities, and deletes duplicate chunk/vector rows.
4. CLI smoke tests cover dry-run/analyze paths where practical.

## Eval

Run `scripts/run_benchmark.py` or the new eval wrapper for at least:

- `fts5`
- `hybrid_rrf`

Capture metrics: `ndcg@3`, `precision@3`, `ndcg@10`, `recall@20`, `map@10`, `mrr`.
