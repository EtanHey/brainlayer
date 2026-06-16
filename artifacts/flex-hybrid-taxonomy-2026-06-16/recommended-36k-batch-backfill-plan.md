# Recommended 36K Hybrid-Taxonomy Backfill Plan

Status: recommendation only. Do not run until Etan explicitly approves.

## Gate Result

The 50-row cloud-safe Flex A/B gate passed with `gemini-2.5-flash` as judge:

- tags: 3.92 -> 4.30, delta +0.38
- summary: 3.82 -> 4.34, delta +0.52
- sentiment: 3.84 -> 4.74, delta +0.90
- debt_impact: 3.24 -> 3.80, delta +0.56
- overall: 3.60 -> 4.00, delta +0.40

AFTER = production prompt + hybrid taxonomy. BEFORE = old Flex free-form with tag validation off.

## Preconditions

1. Keep live realtime enrichment paused until after Etan approves.
2. Confirm backlog count and candidate query are unchanged.
3. Snapshot the canonical DB and WAL state before any import.
4. Set a fixed run id, for example:
   `BRAINLAYER_ENRICHMENT_RUN_ID=hybrid-backfill-2026-06-16-r1`
5. Keep `BRAINLAYER_ENRICHMENT_TAG_MODE=hybrid`.

## Batch Shape

1. Export the preserved 36K backlog to Gemini Batch JSONL with the production prompt and hybrid tag rules.
2. Use Gemini Batch, not realtime, for cost and throughput control.
3. Stamp every output with:
   - prompt version
   - taxonomy git SHA
   - taxonomy content SHA
   - model
   - backend
   - run id

## Sample-Verify Before Full Import

1. Run the batch generation first without mutating live enrichment fields.
2. Import only preview fields or a scratch DB for the first few thousand rows.
3. Sample-judge a few-K slice by strata:
   - high-value assistant/user chunks
   - project-heavy chunks
   - short conversational chunks
   - prior singleton-heavy tags
   - meta/noise chunks
4. Gate checks:
   - parse success rate is acceptable
   - tags stay specific and normalized
   - singleton rate does not spike
   - no confidential rows appear in cloud artifacts
   - DB lock/WAL growth remains bounded

## Full Backfill Recommendation

If the few-K verification passes, import the remaining 36K in bounded chunks:

1. Stop or keep paused any live enrichment/realtime writers.
2. Apply imports through the existing queue/drain path or a single-writer batch importer.
3. Checkpoint after each chunk.
4. Record per-chunk counts: attempted, imported, parse_failed, skipped_stale, DB busy retries.
5. Rebuild/refresh derived tag indexes only after import completes.
6. Run post-import search smoke tests and tag distribution diagnostics.

## Rollback

1. Preserve the pre-import DB snapshot until post-import smoke and spot-judge pass.
2. If tag distribution or search quality regresses, restore from snapshot or roll back only rows with the run id.
3. Keep `BRAINLAYER_ENRICHMENT_TAG_MODE=taxonomy` available only as an A/B rollback/debug flag, not as the recommended production mode.
