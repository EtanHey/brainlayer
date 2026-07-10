# Deterministic Scoped Fan-Out Design

## Status

Approved by the BL-B dispatch brief, which directs implementation of the V2 synthesis verdict `SHIP` and permits either an opt-in `brain_search` parameter or an internal weak-result fallback. This document records the selected integration and its invariants.

## Goal

Add a zero-LLM, bounded search fan-out that improves recall without creating a daemon, weakening consumer visibility, or hiding degraded search legs.

## Approaches considered

1. **Opt-in `brain_search(fan_out=true)` (selected).** Preserves the existing default path and makes the additional read work explicit. It is straightforward to test at the MCP schema and handler boundaries.
2. **Automatic fan-out after a weak first result.** Reduces caller decisions, but “weak” requires a policy threshold and would unpredictably change latency for existing calls.
3. **A separate tool or daemon.** Makes isolation obvious but duplicates the search contract and violates the no-new-daemon constraint.

## Architecture

The existing dispatcher resolves consumer visibility first. Generic searches with `fan_out=true` then execute at most four sequential search legs using the existing `_search` function:

- `raw`: the query under the already-resolved consumer scope;
- `project`: an explicit project-filtered leg when a project scope exists;
- `recent`: a relevance search constrained to the last 30 days (or the caller's stricter lower date bound);
- `tag:<tag>`: one tag-filtered leg when the normalized query surface matches a known taxonomy tag and the caller did not already supply a tag.

Sequential execution is intentional. `_search` protects embedding work with a single in-flight lock; parallel legs would deterministically degrade all but one leg to keyword-only fallback.

Each leg asks for at most 10 results, so four legs can inspect no more than 40 candidates. Results are deduplicated by `chunk_id` and ranked with reciprocal-rank fusion (`k=60`). Equal fused scores use first-seen order, then `chunk_id`, making output stable for identical leg outputs.

## Contract and honesty

Each returned result includes `fan_out_provenance`, listing every leg that found its `chunk_id`, and `fan_out_score`. The top-level structured response records the executed scopes, candidate count, and ranking rule.

If any leg returns an MCP error, reports `degraded=true`, or falls back from hybrid search, the merged response sets `degraded=true`, lists the affected scopes and reasons, and appends a warning to the text response. Successful legs are still returned; a degraded leg is never converted into a silent empty result.

## Testing

Tests use deterministic fake search callables only—never the canonical DB or a real-DB fixture. They prove:

- schema and routing are opt-in;
- four-leg and 40-candidate hard bounds;
- stable merge/dedupe ordering and per-result provenance;
- tag and recency scope planning;
- degraded propagation while preserving successful results.
