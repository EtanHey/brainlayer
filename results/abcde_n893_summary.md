# ABCDE Enrichment Retrieval Benchmark — n=893 (aggregate, PII-free)

> Aggregate metrics only. Raw enriched rows contain personal chunk content and are **gitignored + archived to Brain Drive** (not committed).

## Run metadata
- **Date:** 2026-06-02
- **Sample:** 893 chunks × 3 variants (A production, C density-max, E hyde-structure) = 2,679 enrichment calls
- **Backend:** Nebius `meta-llama/Llama-3.3-70B-Instruct` (OpenAI-compatible)
- **Outcome:** 2,624 ok / 55 transient errors (2%), 0 safety-blocks, ~67 min, **$0.99**
- **Code SHA:** `a1d16429` (branch `feat/brainlayer-abcde-enrich-runner`, PR #430)
- **Raw data (gitignored):** `eval_results/abcde_enrich_nebius.jsonl` → archived to Brain Drive

## Method
Self-retrieval: per variant, build an in-corpus index over the enriched text (summary + tags + key_facts + resolved_queries); query = top-12 distinctive tokens from each chunk's **raw** content (variant-agnostic); gold = the source chunk. Pure term-overlap × idf ranking. Same queries across variants.

## Results (recall@k / MRR / nDCG@10)

| variant | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| A — production | 0.620 | 0.849 | 0.917 | 0.726 | 0.769 |
| C — density-max | 0.625 | 0.858 | **0.934** | 0.731 | 0.778 |
| **E — hyde-structure** | **0.638** | **0.874** | 0.930 | **0.741** | **0.784** |

(B/D not run this pass — only A,C,E were re-enriched.)

## Finding (honest)
**E (hyde-structure) is consistently the best enrichment recipe** for retrieval (wins 4/5 metrics; C edges recall@10) — **but the effect is modest**: A/C/E are within ~3% (recall@1 E vs A = +0.018). Ordering E > C > A is real and consistent; magnitude is small at scale.

**Correction:** an earlier n=24 micro-benchmark showed a dramatic gap (C 0.96 vs A 0.61). That was a **small-corpus artifact** amplified by the self-retrieval method. The n=893 result above is the realistic picture.

## Caveats
- Self-retrieval measures **term-preservation**, not real-query relevance over the corpus.
- The definitive verdict needs the human-labeled query set (Task B, 51 queries) with gold judgments, plus the LLM-judge **quality** pass (pending calibration floors).
