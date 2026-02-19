# Extracting learnings from 268K Claude Code conversation chunks

A five-stage pipeline combining FTS5 keyword filtering, context window assembly, LLM classification, embedding-based clustering, and confidence-scored rule generation delivers the best signal-to-noise ratio for mining actionable rules from your SQLite database. **No single approach works alone** — keyword mining catches only 60–80% of corrections with high false-positive rates, while full LLM classification of all 268K chunks is needlessly expensive. The hybrid pipeline narrows 268K chunks to ~5–15K candidates cheaply via SQL, then spends LLM compute only where it matters. The entire pipeline runs locally on a 32GB laptop in under 8 hours, with no sampling required.

This report synthesizes research across correction detection in dialogue systems, embedding models for mixed code/NL content, clustering algorithms for short text, existing tools in this space, and practical SQLite patterns — all oriented toward auto-generating CLAUDE.md instruction files from conversation history.

## The five-stage pipeline that maximizes signal over noise

The core insight from dialogue systems research is that corrections exist on a spectrum from explicit ("No, use TypeScript") to implicit (user silently redoes what the AI produced). No single technique captures the full range. The optimal architecture is a progressive funnel:

**Stage 1 — SQL/FTS5 pre-filter (268K → 15–25K chunks, milliseconds).** Use SQLite's FTS5 with porter stemming to identify chunks containing correction signals. This is essentially free computationally — FTS5 handles 268K rows in sub-millisecond query times after a 1–3 second index build. Combine keyword matches with metadata filters: `content_type = 'user_message'`, `importance >= 5`, and `intent IN ('debugging', 'reviewing', 'deciding', 'configuring')`. This stage has roughly **30–50% precision but 60–80% recall** — it casts a wide net.

**Stage 2 — Context window assembly (15–25K → 15–25K windows, seconds).** This is where you solve the "no RTL" problem. For each candidate chunk, pull a window of 5 chunks before and 5 after within the same session using `ROW_NUMBER() OVER (PARTITION BY source ORDER BY created_at, rowid)`. The conversation triplet — prior assistant response, user correction, subsequent assistant acknowledgment — provides the semantic context that makes individual chunks interpretable. Without this step, a user message like "no, RTL" is noise; with the preceding assistant text showing a left-to-right layout, it becomes a clear correction.

**Stage 3 — LLM classification (15–25K windows → 3–8K learnings, hours).** Feed each context window to a local LLM (Mistral 7B or Llama 3.1 8B via Ollama) with a few-shot prompt classifying into six categories: CORRECTION, PREFERENCE, RULE, ARCHITECTURE_DECISION, TOOL_PREFERENCE, or NORMAL. Research shows **Mistral 7B achieves κ > 0.8 agreement with humans** on dialogue act classification tasks. At ~500 classifications per minute on consumer hardware, 15K windows processes in about 30 minutes. This stage lifts precision to **75–90%**.

**Stage 4 — Embedding and clustering (3–8K learnings → 50–200 clusters, minutes).** Embed classified learnings using EmbeddingGemma-300M (truncated to 256 dimensions), reduce with UMAP, and cluster with HDBSCAN via BERTopic. This surfaces repeated patterns — five separate "use bun not npm" corrections across different sessions collapse into one high-confidence cluster. Label clusters automatically with c-TF-IDF + KeyBERTInspired representations.

**Stage 5 — Confidence scoring and rule generation.** Score each cluster: single mentions start at **0.4 confidence**, explicit corrections score **0.7**, and repeat observations across sessions compound to **0.85–0.95**. Older learnings decay unless reinforced. Generate CLAUDE.md from clusters exceeding 0.7 confidence, keeping the file under 150 actionable rules.

## Distinguishing genuine corrections from normal conversation

The correction detection problem has been studied extensively in conversational analysis, starting with Schegloff, Jefferson, and Sacks' foundational work on conversational repair. In AI assistant dialogues, corrections follow a specific sequential pattern called **Third Position Repair**: the user says something, the AI misinterprets and responds, and the user corrects in their next turn. This triplet structure is your strongest signal.

Signal words divide into three reliability tiers. **Tier 1 (highest precision)**: contrastive patterns like "not X, use Y", "X instead of Y"; identity markers like "I said", "I already told you"; and explicit negation "that's wrong", "that's incorrect". **Tier 2 (good precision, broad coverage)**: sentence-initial "No," followed by an instruction, "Actually," as a correction marker, imperative "Don't" + verb, and rule-establishment words "always", "never". **Tier 3 (high recall, more noise)**: standalone "wrong", "I prefer", "instead" in isolation.

The critical distinguisher is **position + context**. A "no" at the start of a user turn immediately following an assistant response is far more likely to be a correction than "no" embedded within a sentence. False positives cluster around rhetorical negation ("no problem"), agreement with negation ("no, that's fine"), and discussion of errors in code ("no such file or directory"). Your FTS5 queries should combine signal words with the sequential pattern check — require that the matched chunk follows an `assistant_text` or `ai_code` chunk within the same session.

For the LLM classification stage, the most effective prompt structure provides 2–3 examples per category (10–15 total), includes the preceding assistant context, and uses a structured JSON output format. Research from ACL 2025 shows that even 7B parameter models handle this classification task well when given clear few-shot examples, and that placing the most critical examples last in the prompt slightly improves performance.

## Embedding and clustering 268K chunks locally

**EmbeddingGemma-300M** is the strongest recommendation for this use case. Released by Google in September 2025, it ranks #1 on MTEB benchmarks among models under 500M parameters specifically on code retrieval tasks — critical since your chunks mix natural language with code. It supports Matryoshka representations (truncatable to 256/128 dimensions without retraining), runs in under 200MB RAM with int4 quantization, and integrates directly with sentence-transformers. For your 268K chunks at 256 dimensions, total embedding storage is **275MB** — trivial.

The alternative is **nomic-embed-text-v1.5** if any chunks exceed 2,048 tokens (nomic supports 8,192). For maximum speed on a CPU-only machine, **all-MiniLM-L6-v2** embeds at 5–14K sentences/second but sacrifices quality.

For clustering, **BERTopic** provides the best end-to-end pipeline: embeddings → UMAP (n_components=5, not 2) → HDBSCAN (min_cluster_size=30, min_samples=10) → c-TF-IDF for topic representation. HDBSCAN is preferred over K-means because it automatically determines cluster count and handles noise points — many chunks genuinely don't cluster, and forcing them into clusters pollutes results. Expect HDBSCAN to mark **50–74% of short text as outliers**; use BERTopic's `reduce_outliers(strategy="embeddings")` to reassign borderline cases.

A critical optimization: apply **PCA from 768→50 dimensions before UMAP** for high-dimensional embeddings. This two-stage reduction, recommended by UMAP's creator Leland McInnes, removes noise and dramatically accelerates UMAP on large datasets.

Regarding compute requirements: **268K is very manageable — no sampling needed.** The full pipeline on a 32GB laptop peaks at roughly 12–17GB RAM. Embedding takes 3–6 hours on CPU with EmbeddingGemma (30–60 minutes with GPU), UMAP takes 15–45 minutes, and HDBSCAN takes 5–30 minutes. Total end-to-end: **4–8 hours CPU, under 2 hours GPU**. On a 16GB machine, use all-MiniLM-L6-v2 at 384 dimensions with `low_memory=True` flags.

## What existing tools get right and where the gap is

**No existing tool automatically extracts learnings from AI coding conversation history to generate instruction files.** This is a genuine gap. The closest systems approach the problem from different angles:

**Mem0** (raised $24M, October 2025) provides the most relevant extraction architecture. Its two-phase pipeline — extraction (LLM identifies candidate memories from conversations) then update (compare against existing memories, choose ADD/UPDATE/DELETE/NO-OP) — maps directly onto the correction mining problem. Its custom instructions feature lets you specify "extract coding preferences, library choices, tool preferences" while excluding noise. Mem0 reports **26% higher accuracy than OpenAI's memory** and 90% fewer tokens on the LOCOMO benchmark.

**Zep/Graphiti** offers the gold standard for temporal contradiction resolution. Every fact tracks four timestamps: creation time, expiry time, validity start, and validity end. When a new learning contradicts an existing one (detected via LLM comparison against semantically similar entries), the old fact gets its `t_invalid` set — **new information always wins, but history is preserved**. This directly solves your "user preferences change over time" problem.

**LangMem** from LangChain provides the clearest model for procedural memory — rules, style guides, behavioral patterns — which maps directly to CLAUDE.md generation. Its `metaprompt` algorithm reflects on conversation feedback and proposes prompt/instruction updates, essentially automating rule refinement.

Among coding-specific tools, **Pro-Workflow** is the most mature learning capture system, with `/learn`, `/learn-rule`, and `/search` commands, correction heatmaps tracking which categories get corrected most, and adaptive quality gates. **Agentdex** indexes conversations from Cursor, Claude Code, and Codex into LanceDB for semantic search. **Claude-mem** auto-captures Claude Code session activity and generates CLAUDE.md files with activity timelines. However, all operate on individual session capture — none batch-processes historical conversation logs to extract cross-session patterns.

Claude Code's own memory system (v2.1.32+) now includes auto-memory at `~/.claude/projects/<project>/memory/MEMORY.md` and session memory extraction, but this captures forward-looking notes, not retrospective pattern mining across hundreds of sessions.

## SQLite patterns that make context windows practical

The key query pattern for your database uses `ROW_NUMBER` with `PARTITION BY source` to establish chunk ordering within sessions, then a CTE to pull windows around target chunks:

```sql
WITH numbered AS (
  SELECT rowid, content, content_type, source, created_at,
    ROW_NUMBER() OVER (PARTITION BY source ORDER BY created_at, rowid) AS seq
  FROM chunks
),
targets AS (
  SELECT rowid, source, seq FROM numbered
  WHERE content_type = 'user_message' AND (
    content LIKE 'No,%' OR content LIKE 'Actually%'
    OR content LIKE '%instead of%' OR content LIKE '%should be%')
)
SELECT n.*, t.rowid AS target_id, n.seq - t.seq AS offset
FROM targets t
JOIN numbered n ON n.source = t.source
  AND n.seq BETWEEN t.seq - 5 AND t.seq + 5
ORDER BY t.rowid, n.seq;
```

For the diff analysis approach, correlate user requests with nearby git_diff chunks using `LEAD` window functions to find the sequence `user_message → ai_code → user_message (correction) → git_diff`. This captures cases where the user's correction led to a code change — high-signal evidence of a genuine preference.

**Essential indexes**: `CREATE INDEX idx_chunks_source_time ON chunks(source, created_at, rowid)` is critical for window function performance. Add covering indexes on `content_type`, `importance`, and `intent` for the pre-filter stage. For JSON tags, if you query them frequently, normalize into a `chunk_tags` join table — `json_each()` cannot use indexes and scans each row's JSON on every query.

Set SQLite pragmas for read-heavy batch processing: `PRAGMA journal_mode=WAL` (concurrent reads), `PRAGMA cache_size=-64000` (64MB cache), `PRAGMA mmap_size=268435456` (256MB mmap). With these settings, **268K rows is modest — all queries run in milliseconds to low seconds**.

## Structuring output for automatic CLAUDE.md generation

The learnings table should track seven essential dimensions: `rule_text` (the actionable instruction), `category` (correction/preference/rule/architecture/tool_preference), `confidence` (0.0–1.0, compounds with evidence), `frequency` (observation count), `first_seen`/`last_seen` (temporal bounds), `evidence_chunks` (JSON array of source chunk rowids for auditability), and `status` (candidate/confirmed/superseded). The `superseded_by` foreign key handles temporal evolution — when a user switches from npm to bun, the npm preference gets status='superseded' with a pointer to the bun preference.

For CLAUDE.md generation, filter to `confidence >= 0.7` and `status = 'confirmed'`, group by category, and cap at **150 rules total** — research from HumanLayer shows frontier models follow roughly 150–200 instructions reliably, and Claude Code's system prompt already consumes ~50 of that budget. Use the `.claude/rules/` directory with per-topic `.mdc` files for higher granularity: `corrections.mdc`, `code-style.mdc`, `testing.mdc`, `architecture.mdc`, each with YAML frontmatter specifying applicable file paths.

The confidence scoring formula should weight recency and cross-session repetition heavily: a single correction scores 0.4, an explicit "always/never" rule scores 0.7, repetition across 3+ sessions compounds to 0.85–0.95, and learnings not reinforced within 90 days decay by 0.1 per month. This naturally surfaces durable preferences while letting situational corrections fade.

## Conclusion

The most important insight from this research is that **context windows are non-negotiable** — individual chunks are nearly uninterpretable for correction detection, but a 5-chunk window centered on a signal word achieves 75–90% precision when combined with LLM classification. The second key finding is that the 268K dataset is small enough to process entirely on a developer laptop without sampling, using EmbeddingGemma-300M and BERTopic. Third, the temporal contradiction problem is solved by Zep/Graphiti's four-timestamp model — adopt this pattern for your learnings table rather than inventing a new approach.

The critical gap in existing tooling is the batch retrospective analysis of conversation history. Tools like Mem0 and LangMem extract memories forward (during conversations), while your use case requires mining patterns backward across 800+ historical sessions. The five-stage pipeline described here bridges that gap. Start with Stage 1 (FTS5 keyword filter) and Stage 2 (context windows) — these two stages alone, implementable in pure SQL in an afternoon, will surface the highest-signal corrections. Add the LLM classification and clustering stages incrementally as you validate the approach on initial results.