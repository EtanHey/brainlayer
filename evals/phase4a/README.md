# Phase 4a — Eval Framework

> **Status**: SEED PR (DRAFT) — data files only. Runner, metrics, and CI gate are next iteration.
>
> **Purpose**: 80+ query eval set with Hebrew weighting that GATES any retrieval-shape change (Phase 4b hnlx + Int8 + dual-datastore cannot ship without this passing).
>
> **Source dispatch brief**: `~/Gits/orchestrator/docs.local/handoffs/2026-05-22/phase-4a-eval-framework-dispatch.md`

## Directory contents

```
evals/phase4a/
├── README.md          # this file
├── queries.yaml       # 80 queries: 15 Hebrew + 12 health + 15 conceptual + 15 frustration + 8 temporal + 15 entity
├── sentinel.yaml      # 5 fast pre-commit smoke queries (one per category)
├── thresholds.yaml    # pass criteria (recall@20 per category, ndcg@10, latency budgets)
├── runner.py          # TODO next iteration: invokes brain_search × queries, captures metrics
├── metrics.py         # TODO next iteration: Ranx + RAGAS + DeepEval wrappers
├── ci_gate.py         # TODO next iteration: CLI `bl-eval smoke` / `bl-eval full --compare-to baseline.json`
├── baseline.json      # TODO next iteration: committed snapshot of metrics from current DB
└── tests/             # TODO next iteration: test_runner / test_metrics / test_ci_gate
```

## Why this is a SEED PR

The 80 queries are pre-curated based on:
- Today's BrainBar verification work (Hebrew probes from CD-1 latency report)
- Etan's verbatim corrections captured in BrainLayer (`fru-*` category)
- coachClaude domain (`hlt-*` category)
- Known entities from BrainLayer's `kg_entities` (`ent-*` category)

Landing the query set FIRST lets the runner be developed against fixed data. Subsequent commits add the framework.

## Categories and weighting (per `queries.yaml`)

| Category | Count | Purpose | Failure tolerance |
|----------|-------|---------|-------------------|
| `hebrew` | 15 | Token coverage + trigram fuzziness + cross-script transliteration (Bug E) | 85% (heb-05 is known miss pre-multilingual-embed) |
| `health` | 12 | coachClaude alignment + WHOOP/sleep/recovery domain | 92% |
| `conceptual` | 15 | Abstract retrieval — phrasing variability tests | 90% |
| `frustration` | 15 | Recurring user corrections — MUST surface | 95% |
| `temporal` | 8 | Time-anchored retrieval — recency intent tests | 85% |
| `entity` | 15 | Known kg_entities — baseline anchor | 95% |

## Sentinel set (`sentinel.yaml`)

5 queries that run in <30s total for pre-commit hooks. One representative from each category. All must pass; CI workflow blocks if any sentinel fails.

## Thresholds (`thresholds.yaml`)

- `recall@20` minimum 90% per category (with category overrides per known-miss tolerance)
- `ndcg@10` aggregate ≥0.85
- No category may regress more than 5% vs `baseline.json`
- Latency: sentinel total <30s, full eval <120s, per-query p95 <500ms, per-query max <8s

## How Phase 4b will use this

Phase 4b (hnlx + Int8 + dual-datastore) MUST pass:
- All sentinel queries (smoke pre-commit)
- Full 80-query eval ≥thresholds (CI gate before merge)

If Phase 4b regresses >5% on any category → block merge, iterate or split into smaller PRs.

## Why this lands BEFORE the runner

`/post-merge-deploy-check` lesson from 2026-05-22: data + code shipping in lockstep risks subtle drift. By shipping queries first:
- Queries can be reviewed independently for accuracy (Hebrew spelling, entity names, frustration phrasing)
- Runner can be developed against fixed query data (TDD-friendly)
- Future query updates don't require Python changes

## Next iteration

After this DRAFT PR is reviewed for query accuracy:
1. Land `runner.py` (~50 LOC) — invokes brain_search, captures latency + result IDs
2. Land `metrics.py` (~100 LOC) — Ranx + RAGAS + DeepEval wrappers
3. Land `ci_gate.py` (~50 LOC) — CLI entry + threshold comparison
4. Land `baseline.json` — initial snapshot from current DB
5. Land `tests/` (~200 LOC) — TDD coverage
6. Land `.github/workflows/eval.yml` — CI wiring

Then mark PR ready-for-review + merge.

## Cross-references

- Dispatch brief: `~/Gits/orchestrator/docs.local/handoffs/2026-05-22/phase-4a-eval-framework-dispatch.md`
- Phase 4b skeleton: `~/Gits/orchestrator/docs.local/handoffs/2026-05-22/phase-4b-hnlx-int8-dual-datastore-dispatch.md`
- Binding design doc: `~/Gits/orchestrator/docs.local/plans/2026-05-21-brainlayer-readpath-redesign/PHASE4-DESIGN.md`
