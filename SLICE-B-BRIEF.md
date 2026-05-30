# L0 Slice B — Correction JUDGE (Grokipedia adjudication) (brainlayerCodex brief)

You are **brainlayerCodex-l0-sliceB**. Worktree: `~/Gits/brainlayer-l0-slice-b` (branch `feat/l0-correction-judge`, off main `81315a12` which already has Slice A's `entity_facts`). cd there; work THERE only (not canonical). Full design: `~/Gits/brainlayer/docs.local/plans/2026-05-30-L0-compound-memory/README.md` ("CORRECTION MODEL" + "Slice B").

## WHAT (the adjudication layer on top of Slice A's entity_facts)
When a NEW assertion about an entity CONFLICTS with an existing active `entity_facts` row (e.g. new "BrainLayer uses sqlite-vec" vs existing "BrainLayer uses ChromaDB"), an **LLM-as-judge** decides: **SUPERSEDE** (new replaces old → mark old `status='superseded'`, `superseded_by`=new, new active), **MERGE**, or **NOISE** (ignore). This judged decision IS the supersession mechanism that makes corrections override stale facts (the North Star). Slice A built the fact store + frequency; Slice B adds the judge.

## REUSE the existing Gemini backend — NO new endpoint/key/infra (Etan directive)
The judge MUST reuse BrainLayer's existing enrichment Gemini path, NOT a new API:
- `src/brainlayer/enrichment_controller.py`: `_get_gemini_client()` (line ~491), `_build_gemini_config()` (~558), `GEMINI_REALTIME_MODEL` = `gemini-2.5-flash-lite` (~34), and the rate-limited call `_generate_content_with_rate_limit(client, model, prompt, config, rate_limiter)` (~929). Reuse these for the judge's per-correction call (cheapest, no new keys).

## SWAP-TO-LOCAL SEAM (Etan directive — zero-rewrite later)
Define a backend-agnostic judge interface so a local LLM can drop in later with NO change to adjudication logic:
- `CorrectionJudge` protocol/ABC: `judge(entity, new_fact, conflicting_fact, context) -> Verdict{action: supersede|merge|noise, confidence, reasoning}`.
- `GeminiCorrectionJudge(CorrectionJudge)` — reuses the enrichment Gemini client/model/rate-limiter above + a judge prompt + a strict JSON `response_schema` for the verdict.
- A `LocalCorrectionJudge` STUB (raises NotImplementedError / TODO) to prove the seam — selected via `BRAINLAYER_JUDGE_BACKEND=gemini|local` (default gemini). Swapping backends must touch ONLY the factory, not the adjudication wiring.

## SCOPE (TDD, additive, PR-loop)
1. `src/brainlayer/correction_judge.py` — the `CorrectionJudge` ABC + `Verdict` + `GeminiCorrectionJudge` (reusing enrichment Gemini path) + `LocalCorrectionJudge` stub + a `get_correction_judge()` factory (env-selected).
2. **Conflict detection + adjudication** in the entity_facts path (kg_repo): when aggregating/adding a fact that conflicts with an existing active fact for the same entity, call the judge → apply the verdict (supersede → update `entity_facts.status`/`superseded_by`; merge; noise=skip). Keep it OFF the hot read path — adjudicate at digest/refresh time, not per brain_entity call.
3. Tests (TDD, RED first): (a) verdict JSON parsing + factory/seam (mock the Gemini client — do NOT hit the real API in unit tests); (b) adjudication state transitions on entity_facts (supersede marks old superseded + new active); (c) one e2e with the judge MOCKED returning supersede → brain_entity shows the corrected fact, stale gone. Add a LIVE-gated (skipped-by-default, `@pytest.mark.live`) real-Gemini test for manual runs.

## MANDATE
- **TDD** (RED first, paste stdout). Additive — don't change Slice A's deterministic aggregation behavior; the judge is layered ON it.
- **PR-loop** off `feat/l0-correction-judge`; do NOT merge — COMMANDER (s:7) reviews + runs the FUNCTIONAL acceptance (a real correction supersedes the stale entity_fact, brain_entity reflects it) + merges.
- Run ONLY targeted tests (your new test files + the entity_facts ones) — NOT the full flaky pre-push suite; if pre-push blocks, push with `git push --no-verify` (CI runs the full suite).
- Do NOT touch BrainBar UI. You do NOT need brain_* MCP tools — pure Python + pytest + the enrichment Gemini client. BrainLayer recall is best-effort; never block on it.
- **FUNCTIONAL acceptance:** seed an entity with a stale fact + feed a conflicting correction → judge (mock in test; real Gemini for the live demo) supersedes → `brain_entity` shows the correction, stale fact superseded. Report literal output.
- Sync progress in `~/Gits/orchestrator/docs.local/plans/2026-05-30-ecosystem-stabilization/collab.md` (L0 Slice-B note). Report concisely; never go silent. Emit TASK_DONE + PR link when pushed.
