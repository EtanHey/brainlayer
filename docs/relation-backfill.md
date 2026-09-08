# Backfill relations on existing entities

The old KG rebuild's seed/tag tier creates entity links without relations. Its LLM
tier skips linked chunks and requires importance >= 6, so repeating it cannot
repair those chunks. Relation extraction now has an independent completion ledger.

Extraction quality must pass its pre-registered gold-set evaluation, including
corpus cross-verification, before any corpus run. The current runner is not yet
qualified. Never loosen evidence validation to increase graph size; passing unit
tests does not authorise a corpus run or canonical writes.

Rehearse on a database copy first. Start an owned, niced MLX server with an explicit
model, one prompt/decode at a time, bounded KV cache and small prefill batches.
Do not use ports 8080, 8081 or 8178: they belong to other workloads. Then run:

```sh
python -m brainlayer.pipeline.relation_inference \
  --db /absolute/path/to/copy.db \
  --endpoint http://127.0.0.1:8183 \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --conversations --limit 100
```

Both endpoint and model are required. The client rejects a different served model,
truncated output, unknown IDs, unsupported relations and incomplete responses. An
invalid extraction gets one explicit model correction request; failure stays loud
and retryable. Empty output is never synthesized as a fallback. The model returns entity names and quotes only, with no IDs. Unambiguous canonical
names resolve deterministically against the supplied existing entities. Unknown or
ambiguous names stay retryable; no fuzzy match or new entity is invented. Source data
and extraction instructions use separate message roles. The optional `on_response`
callback retains raw request/response envelopes before validation or correction;
keep such traces private because they contain source text.

`--conversations` limits this run to CLI conversation sources (claude_code,
codex_cli, cursor, realtime, realtime_watcher) and user_message/assistant_text.
It is a connection-local read filter; source rows are untouched. Omit it for all
active linked sources. `--window-chars` defaults to 6,000, with overlapping and entity-pair windows:
all source text is visited, including long chunks; only windows containing at least
two known entity names need inference. Every distinct endpoint pair within the
configured context span shares a window. No whole source is silently truncated.

Every new fact retains an exact supporting quote, chunk ID and source content hash.
Ended or historical-only facts are inserted as non-current. Existing relations,
including expired facts, remain unchanged. Chunks and entities
are never updated. All windows must succeed before that chunk's facts and completion
commit together. Completion fingerprints include text, entity names/types/IDs and
window size, so changed inputs become eligible again. Conflicting temporal states
within a source are rejected together instead of letting window order choose one.
To advance a bounded scan, pass the emitted `next_chunk_id` as `--after-chunk ID`.
The keyset cursor skips earlier completed/rejected batches without reloading their
entities. Omit the cursor deliberately to retry rejected or changed earlier sources;
blindly repeating from newest can revisit the same rejected batch.

The command prints completed/rejected chunk, window and newly inserted relation counts.
`--continue-on-rejection` records semantic rejections, leaves those chunks incomplete,
and processes other sources in the bounded run; any rejection still produces exit 2.
Transport/envelope failures, wrong served models and truncated output stop immediately.
Zero added can be correct abstention, especially for transcript fragments incorrectly
stored as entities. Check evidence before broadening: never loosen truth gates just
to increase graph size. Entity quality is a separate repair.

Production runs require the owner's operational approval: stop enrichment writers
by label, checkpoint before/after, sequence one writer and keep run windows bounded.
Never lift an existing enrichment pause or drain its queue. This command does not
restart BrainBar, manage services, or change the sentinel. Shut down only the exact
owned MLX process after the backfill finishes. Human-check sampled proposed facts;
passing transport/quote checks alone does not prove semantic correctness.
