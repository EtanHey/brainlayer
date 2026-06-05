# Phoenix Eval Integration — Design Doc

> Author: brainlayerClaude-LEAD-v6 · 2026-06-02
> Status: **DESIGN (gate-before-build)**. The BUILD is delegated to a fresh Codex (xhigh) and gated by LEAD.
> Decision context: Etan picked **Arize Phoenix** as the local retrieval-eval platform (see `/tmp/eval_platform_comparison.md`, v5's cited comparison; Phoenix = top pick on all 4 needs + strongest privacy + lowest setup).

---

## 0. TL;DR

Stand up **Arize Phoenix self-hosted at `localhost:6006`** as a **read-only eval viewer** over our existing ABCDE retrieval data. Phoenix gets its **own storage** (its SQLite/Postgres volume) — it **never** touches the BrainLayer DB and **never** calls `brain_store`. We push two things into it:

1. **A labeled dataset** (51 queries → returned chunks → relevance labels) so Etan can see **query + retrieved context + per-metric pass/fail** cleanly.
2. **Experiments** comparing ABCDE variants **A / C / E** on `recall@{1,5,10}`, `MRR`, `nDCG@10` over the n=893 benchmark — as side-by-side experiment runs.

Pure retrieval metrics = **ZERO network** (air-gapped). The only optional egress is an LLM-judge relevance pass, which uses a provider key we already hold (Nebius/Grok). Phone access: recommend a **Tailscale tunnel to `localhost:6006`** for live viewing, with **agent-html static snapshots** as the zero-trust fallback.

**Plus two reconciliations folded in (§6, §7):** the *promptfoo* UI Etan saw tonight (a deviation from the Phoenix standard), and the query-vs-description / truncation / only-failing display confusion (a quirk of that promptfoo tool, fixed by Phoenix's layout).

---

## 1. Our data → Phoenix datasets + experiments

Phoenix's model: a **Dataset** is a versioned set of examples (input → expected → metadata); an **Experiment** runs a task over a dataset and attaches **evaluators** (scores) per example; the UI diffs experiments and supports **human annotations**. We map our two artifacts onto this directly.

### 1a. Labeled retrieval set → Phoenix **Dataset** (the human-label / inspection surface)

Source: `/tmp/labeled_eval_candidates.json` — **51 queries**, each with `returned_chunks[]` (machine-proposed labels). Distribution: **236 `relevant` + 20 `not`** chunk labels; **5 queries flagged `suspected_miss`**, **8 `suspected_misorder`**.

Per-query record shape (verified):
```
{ "query": "...",                       # the search string sent to the retriever
  "returned_chunks": [
    { "id": "brainbar-7f14a516-78a",
      "snippet": "...",                 # retrieved context (what the user actually grades)
      "machine_label": "relevant"|"not",
      "why": "token overlap 0.86 ..." } ],
  "suspected_miss": bool, "suspected_misorder": bool, "notes": "..." }
```

**Mapping → one Phoenix dataset `brainlayer-labeled-51`:**
- **example input** = `query` (the retriever input — keep this distinct from any human label; see §7).
- **example output / reference** = the ordered `returned_chunks` (id + snippet + `why`), rendered as the **retrieved context** block.
- **metadata** = `machine_label` per chunk, `suspected_miss`, `suspected_misorder`, `notes`, `n_chunks`.
- **annotations** = Etan's human relevance label per chunk (Phoenix human-annotation UI), which **supersedes** `machine_label`. This is the "is this chunk actually relevant?" pass.

This is the surface that answers Etan's confusion: each row shows **query** (input) + **retrieved chunks** (context, full snippet, not truncated) + **per-chunk label** and **per-query flags**, and **all rows are navigable** (not just failures).

### 1b. ABCDE benchmark → Phoenix **Experiments** (variant comparison)

Source: `/tmp/retrieval_benchmark_results.json` (aggregate metrics, n=893) + raw rows `~/Gits/brainlayer-abcde/eval_results/abcde_enrich_nebius.jsonl` (**2624 ok / 2679 — PERSONAL, stays local**).

Verified `meta`: `n_chunks=893`, method = pure-python term-overlap×idf self-retrieval (queries = top-12 raw-content tokens, variant-agnostic; gold = source chunk). **Caveat baked into meta:** self-retrieval favors verbatim-copying variants — real quality needs the LLM-judge pass (parked).

Verified `per_variant[A..E]` keys: `ok_chunks, scored_queries, indexed_coverage, recall@1, recall@5, recall@10, mrr, ndcg@10`. Example (A): recall@1 **0.620**, recall@5 0.849, recall@10 0.917, MRR 0.726, nDCG@10 0.769.

**Mapping → Phoenix experiments, one per variant we care about (A / C / E):**
- Build a **dataset `brainlayer-abcde-893`** from the n=893 gold pairs (query tokens → gold chunk id), metadata-only (no full personal text needed for the metric — the metric is id-match).
- Each **variant = one experiment run** over that dataset; the **task** = "retrieve top-k for this query under variant V"; the **evaluators** = `recall@1/5/10`, `MRR`, `nDCG@10` (Phoenix native retrieval evals: nDCG@k / precision@k / hit).
- Phoenix's **experiment-diff** view then shows **A vs C vs E side-by-side** per metric — exactly the comparison the headline result needs.

**Headline to preserve in the UI (from boot):** E (hyde-structure) consistently best but **MODEST** — E>C>A within ~3% (recall@1 E 0.638 / C 0.625 / A 0.620). The n=24 micro (C 0.96 vs A 0.61) **OVERSTATED** it — small-corpus artifact. The Phoenix experiment-diff makes the "modest, not dramatic" truth visually obvious (overlapping bars), which is the honest framing.

> **Ingest path, two options for the Codex (gate on whichever is cleaner):**
> (a) **Pre-computed provider** (fastest, mirrors what already works): a thin task fn that *replays* our precomputed `returned_chunks` per query — no live retrieval call, deterministic, fully offline. Recommended for v1.
> (b) **Live retrieval task**: task fn calls BrainLayer's retriever (read-only `brain_search`-equivalent) at eval time. Higher fidelity but couples Phoenix runs to the live DB; defer to v2.

---

## 2. Sandbox from BrainLayer (hard isolation)

**Requirement: Phoenix must not read or write the BrainLayer DB, and must never `brain_store`.**

- **Storage:** Phoenix self-host uses its **own** persistence — a dedicated Docker volume (Postgres) or a local SQLite file under a Phoenix-only working dir (e.g. `~/.local/share/phoenix/`). **Never** point `PHOENIX_SQL_DATABASE_URL` at `~/.local/share/brainlayer/brainlayer.db`.
- **Data flow is one-way and snapshot-based:** we *export* JSON snapshots (the 51-query set; the n=893 aggregate + id-only gold pairs) from `/tmp` artifacts into Phoenix datasets. BrainLayer is the source; Phoenix is a downstream read-only mirror. No process writes back to BrainLayer.
- **No MCP write tools in the Phoenix path.** The Codex build gets **read-only** data files; it does **not** import `vector_store.py` write paths and does **not** call any `brain_*` mutate tool.
- **Process isolation:** Phoenix runs in its own container / venv. It does not share BrainLayer's socket (`/tmp/brainlayer.sock`), enrichment lock, or WAL. Zero contention with the indexer/enrichment workers (respects the CLAUDE.md bulk-op safety rules by simply never touching the DB).

Net: a leak is structurally impossible because Phoenix has no handle to the BrainLayer DB and no write tool.

---

## 3. Phone-access layer (compute stays on the Mac)

Phoenix UI binds `localhost:6006` on the Mac. Three ways to reach it from a phone:

| Option | What Etan gets | Privacy posture | Effort |
|---|---|---|---|
| **(a) Tailscale tunnel** ⭐ | **LIVE** Phoenix UI on phone (annotate, diff experiments) over the tailnet | Data never leaves Etan's devices; WireGuard E2E, no public exposure | **Low** (already-common in this ecosystem) |
| (b) agent-html / Lakebed **static snapshots** | **Read-only** rendered tables/charts on phone, published as static HTML | Snapshot is a derived view; **must scrub personal chunk text before publish** | Low-Med |
| (c) MCP endpoint | Query eval results via an MCP tool from any Claude surface | Stays in the agent layer; no browser | Med (build a tool) |

**Recommendation: (a) Tailscale** for live interaction (Etan can actually *annotate* relevance from the phone, which (b) can't do), with **(b) static snapshots as the zero-trust fallback** for quick read-only glances and for sharing a frozen result. (c) is a nice-to-have later, not v1.

**Guardrail for (b):** static snapshots of the **labeled-51** set contain personal chunk snippets → snapshots must be **metric/aggregate-only** OR scrubbed. The **893 experiment-diff** (id-match metrics, no full text) is safe to snapshot as-is. This mirrors the live PII-incident lesson (§ parked) — never publish raw personal chunk text.

---

## 4. Air-gapped confirmation (for Etan)

- **Pure retrieval metrics** (`recall@k`, `MRR`, `nDCG@10`) over our own labels/gold = **ZERO network**. Phoenix self-host is explicitly air-gappable ("fully air-gapped, nothing sent to Arize" — `arize.com/docs/phoenix/self-hosting`). Disable any usage telemetry env flag at boot to be belt-and-suspenders.
- **The ONLY optional egress** is the **LLM-judge relevance scoring** (faithfulness/usefulness) — and that's opt-in, calling a provider **we** pick with **our** key (Nebius working / Grok ~$1 left). Smoke-first the key before any metered judge run (boot rule). The deterministic metric path needs **no** key at all.
- Verify post-boot: `lsof -i -P | grep phoenix` should show only `localhost:6006` listeners, no outbound — confirm during the build gate.

---

## 5. Clean display: query + retrieved-context + per-metric pass/fail

This is the explicit ask. Phoenix's example/experiment view must render, per row:
1. **Query** — labeled "Query (retriever input)": the exact search string sent to the retriever (vectorized + FTS).
2. **Retrieved context** — the ordered chunks (id, importance, type, full snippet, retrieval `why`), **not truncated**, scrollable.
3. **Per-metric pass/fail** — each evaluator (`recall@1`, `hit`, `nDCG@10`, human-relevance) shown as its own column with a clear pass/fail/score, **all rows navigable** regardless of pass/fail.

The `provider.js` rendering already built in `brainlayer-eval-ui` (renders id + machine_label + importance + type + tags + key_facts + summary + full_text per chunk) is a **good content template** to port into the Phoenix task fn — we reuse the rendering, drop the promptfoo harness (§6).

---

## 6. RECONCILE: promptfoo deviation (found tonight)

**Finding: an agent stood up a `promptfoo` eval UI — this is a deviation from the Phoenix standard.**

- **Location:** `~/Gits/brainlayer-eval-ui/` (separate repo). Created **today 2026-06-02, 12:16–12:20**.
- **Contents:** `package.json` → `promptfoo@^0.121.13`; `promptfooconfig.json` (26 KB, "BrainLayer ABCDE Retrieval Grading — human relevance labeling", 51 tests); `provider.js` (custom provider that **replays** precomputed BrainLayer retrieval results — **no LLM calls**); `data/eval_dataset.json` (627 KB, **personal chunk text — local only**); `node_modules/`.
- **It ran:** a task output shows `Port 15500 is already in use. Do you have another Promptfoo instance running?` → `promptfoo view` was launched. This is the eval UI Etan saw tonight.
- **Risk: LOW / contained.** The repo is **local-only**: `master` has **no commits**, **no remote**, everything untracked. The 627 KB personal dataset has **not** left the machine. No public exposure (unlike the PR #430 incident). But it does hold raw personal chunks on disk in a stray repo.

**Disposition (LEAD recommendation):**
- ✅ **Standardize on Arize Phoenix** (Etan's pick). promptfoo was a quick-and-dirty local grading harness, not the chosen platform.
- ♻️ **Salvage, don't trash:** port `provider.js`'s chunk-rendering and reuse `data/eval_dataset.json` as a Phoenix dataset source. The *data* is good; only the *harness* changes.
- 🧹 **Retire `brainlayer-eval-ui`** once Phoenix replays the same 51 set — and **scrub `data/eval_dataset.json`** (personal text) or move it under `docs.local/` (gitignored) so it's not a loose personal-data repo. **Ask Etan before deleting** (personal-data rule).
- 📌 No git history to purge (no commits, no remote) — cleanup is just `rm -rf` after Phoenix parity + Etan OK.

**Why it confused the picture:** it's a *second* eval UI living outside the brainlayer repo, named `brainlayer-eval-ui`, so "I thought we were using Phoenix" is exactly right — we *are*; this was an unsanctioned detour.

---

## 7. RECONCILE: Etan's eval-UI observations (folded into the Phoenix design)

Etan saw, in the promptfoo UI: (1) **"query" and "description" nearly identical**, some descriptions **cut mid**; (2) could **only navigate FAILING cases**. Root causes, verified against `promptfooconfig.json`/`provider.js`:

- **query == description:** In `promptfooconfig.json` **every test sets `description` = the `query` string verbatim** (e.g. both = `"gen-6 orc SESSION CHECKPOINT 2026-06-01 ETAN DECISIONS"`). They look identical **because they are the same string** — the generator used the query as the human label.
  - **Clarification for Etan:** **`query`** = the search string sent to the retriever (vectorized + FTS). **`description`** = a human label *of the test case*. They only collide here because whoever generated the config set the label = the query. In Phoenix we'll label the column **"Query (retriever input)"** and keep any human description as a *distinct* optional field, so they never masquerade as each other.
- **descriptions cut mid:** promptfoo's web table **truncates** the description column. A **display quirk of that tool**, not missing data — the full string is present in the config. Phoenix renders the full query + scrollable context (§5).
- **only FAILING cases navigable:** `promptfooconfig.json` injects synthetic **always-fail** assertions on flagged rows — `false /* ZERO-CHUNK GAP */` and `false /* SUSPECTED MISS */`. So the only "failures" promptfoo surfaces are the **auto-flagged** rows (the 5 misses / 8 misorders), and promptfoo's UI lets you filter to failures. That's a **filter/assertion artifact**, not "passing cases are hidden." Phoenix shows **all 51 rows**, every metric as its own pass/fail column, navigable regardless of outcome.

Net: all three are **display/harness quirks of promptfoo**, fully explained, and **designed away** by the Phoenix layout in §5.

---

## 8. Build plan (delegated to Codex xhigh — gated by LEAD)

**Scope for the worker (v1, read-only, offline-first):**
1. Phoenix self-host at `localhost:6006`, **own storage** (Phoenix-only volume/dir), telemetry off, sandboxed from BrainLayer DB (§2).
2. Loader script: `/tmp/labeled_eval_candidates.json` → dataset `brainlayer-labeled-51` (input=query, context=chunks, metadata=labels/flags), human-annotation enabled.
3. Loader script: n=893 aggregate + id-only gold → dataset `brainlayer-abcde-893`; **experiments A / C / E** with native `recall@k`/`nDCG@10`/`hit` evaluators; experiment-diff view.
4. Reuse `brainlayer-eval-ui/provider.js` chunk-rendering for the context block (§5); **drop promptfoo**.
5. Verify air-gap: `lsof` shows only localhost listener, no outbound on the metric path (§4).
6. **Do NOT** wire any `brain_*` write tool; **do NOT** open the BrainLayer DB; **do NOT** commit personal data (`eval_results/*.jsonl`, `data/eval_dataset.json` stay local/gitignored).

**Gate (LEAD verifies before "done"):** (a) `localhost:6006` up, datasets + experiments visible; (b) a row shows query + full context + per-metric columns, all 51 navigable; (c) `lsof` clean (no egress on metric path); (d) no BrainLayer DB handle, no `brain_store` call, no personal data staged for commit; (e) A/C/E diff reproduces the modest E>C>A (~3%) headline.

**Out of scope v1 (parked):** LLM-judge quality pass (needs Research-B kappa floors + metered key smoke-first); live-retrieval task fn (§1b option b); MCP endpoint (§3c); Tailscale wiring (Etan device-side).

---

## References
- `/tmp/eval_platform_comparison.md` — cited platform comparison (Phoenix top pick).
- `/tmp/labeled_eval_candidates.json` — 51-query labeled set (236 relevant / 20 not; 5 miss / 8 misorder).
- `/tmp/retrieval_benchmark_results.json` — n=893 per-variant metrics (A..E).
- `~/Gits/brainlayer-abcde/eval_results/abcde_enrich_nebius.jsonl` — raw rows (2624 ok, PERSONAL, local-only).
- `~/Gits/brainlayer-eval-ui/` — the promptfoo deviation (local, uncommitted) — salvage data, retire harness.
- Phoenix self-host / air-gap / retrieval evals: arize.com/docs/phoenix/self-hosting · /deployment-options/docker · /cookbook/evaluation/evaluate-rag
