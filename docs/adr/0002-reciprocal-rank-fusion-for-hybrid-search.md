# ADR-0002: Reciprocal Rank Fusion for hybrid search

**Status:** Accepted

**Date:** 2025-12 (approximate — decision predates ADR documentation)

**Deciders:** EtanHey

## Context

BrainLayer needs to combine two search signals for retrieval:

1. **Semantic search** — sqlite-vec KNN over bge-large-en-v1.5 embeddings (1024 dims). Good at finding conceptually related content ("how did I implement auth?" matches a chunk about "OAuth2 token refresh").
2. **Keyword search** — FTS5 full-text search over chunk content, summaries, and tags. Good at finding exact matches that semantic search misses (specific function names, error codes, file paths).

Neither signal alone is sufficient. Semantic search misses exact terms; keyword search misses paraphrased intent. The question is how to merge the two ranked result lists into a single ranking.

Candidates considered:

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Linear combination** | `α * semantic_score + (1-α) * keyword_score` | Simple, tunable | Requires score normalization across different scales |
| **Reciprocal Rank Fusion (RRF)** | `Σ 1/(k + rank_i)` per result | Score-agnostic, no normalization needed, robust | Single hyperparameter (k), no weight tuning |
| **Cross-encoder re-ranker** | LLM re-scores merged candidates | Highest quality | Requires a second model, adds latency |
| **Keyword-only fallback** | Semantic first, FTS if no results | Simple | Misses boosting from keyword overlap |

## Decision

Use **Reciprocal Rank Fusion (RRF)** to merge semantic and keyword results, with post-RRF boosting for importance and recency.

The implementation in `search_repo.py` works as follows:

1. **Retrieve candidates** — run both semantic (top 30) and FTS5 (top 30) searches with the same filters.
2. **Compute RRF scores** — for each unique chunk_id across both result sets:
   ```
   score = 0
   if chunk in semantic results: score += 1 / (k + semantic_rank)
   if chunk in FTS results:      score += 1 / (k + fts_rank)
   ```
   where `k = 60` (standard default from the original RRF paper).
3. **Post-RRF boosting** — multiply the RRF score by two heuristic factors:
   - **Importance boost**: `1.0 + min(importance, 10) / 20` — range 1.0x to 1.5x based on the chunk's enriched importance score (0-10).
   - **Recency boost**: `0.7 + 0.3 * exp(-0.023 * age_days)` — exponential decay with a 30-day half-life, range 0.7x (old) to 1.0x (fresh).
4. **Sort and return** top `n_results` by boosted score.

Results are cached in a module-level LRU cache (128 entries, 60-second TTL) keyed on `(store_path, query_text, embedding_hash, all_filters, k)`.

## Consequences

### Positive

- **No score normalization needed** — RRF operates on ranks, not raw scores. This avoids the fragile calibration problem of combining L2 distances (semantic) with BM25 scores (FTS5) on different scales.
- **Robust with minimal tuning** — the single `k` parameter (60) is the standard default from [Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009](https://dl.acm.org/doi/10.1145/1571941.1572114) and works well in practice. No per-query weight adjustment needed.
- **Chunks appearing in both lists get a natural boost** — a result ranked highly by both semantic and keyword search receives contributions from both terms, surfacing the most relevant content.
- **Handles disjoint results gracefully** — chunks appearing in only one list still get a score and can surface if ranked highly enough.
- **Fast** — the RRF merge itself is O(n + m) after both searches complete. The dominant cost is the semantic KNN scan, not the fusion.

### Negative

- **`k = 60` is hardcoded** — while the standard default works well for general retrieval, domain-specific tuning (e.g., favouring keyword precision for error-code lookups vs. semantic recall for concept queries) would require making `k` configurable. This is a known limitation.
- **No learned relevance** — RRF is a heuristic. A cross-encoder re-ranker would produce higher-quality rankings but at the cost of loading a second model and adding 100-500ms latency per query.
- **Post-RRF boosting adds implicit bias** — the importance and recency multipliers mean that a highly-important recent chunk can outrank a more semantically relevant older one. This is intentional (recent decisions matter more) but could surprise users searching for historical content.
- **Cache invalidation is time-based only** — the 60-second TTL means writes within the cache window are invisible to search. Acceptable for the typical MCP usage pattern (search → store → next prompt) but could cause stale results in rapid write-then-read loops.

### Neutral

- The 3x over-fetch (top 30 from each source for a default `n_results=10`) provides a large enough candidate pool for RRF fusion without excessive DB load.
- Adding a third signal (e.g., knowledge graph proximity) would simply add another `1/(k + rank_i)` term to the RRF sum — the algorithm extends naturally.
