# Session-level enrichment architecture for BrainLayer

BrainLayer's next evolution should treat sessions as first-class analytical units with a hybrid flat-column/JSON schema, a tiered processing pipeline that routes sessions by size between Gemini Flash and local map-reduce, and a corrections-as-entities subsystem where user corrections graduate into reusable rules through spaced-repetition-inspired confidence scoring. This design draws on proven patterns from Zep's temporal knowledge graph, Mem0's AUDN memory loop, and Cognee's DataPoint model — adapted for SQLite with sqlite-vec and FTS5. The result: **268K chunks across 800+ sessions become a searchable knowledge base** that captures not just what happened, but what was learned, what failed, and what rules emerged.

---

## 1. The schema should blend relational precision with JSON flexibility

The central design tension is between queryable flat columns (for dashboards, filtering, aggregation) and flexible JSON columns (for variable-length arrays like decisions, corrections, and tool stats). The right answer is a hybrid: **flat columns for anything you filter or sort on, JSON for everything else**, with SQLite generated columns bridging the gap when a JSON field later needs indexing.

A single `session_enrichments` table works best here since the relationship is strictly 1:1 (one enrichment per session). Normalization into separate tables is warranted only for many-to-many relationships like topics and tool usage, where you need efficient reverse lookups ("find all sessions about TypeScript" or "aggregate Bash tool success rates").

```sql
CREATE TABLE session_enrichments (
    -- Identity
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    file_path TEXT,
    enrichment_version TEXT NOT NULL DEFAULT '1.0',
    enrichment_model TEXT,
    enrichment_timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),

    -- Timing (flat — for temporal queries)
    session_start_time TEXT,
    session_end_time TEXT,
    duration_seconds INTEGER,

    -- Message dynamics (flat — for aggregation dashboards)
    message_count INTEGER NOT NULL DEFAULT 0,
    user_message_count INTEGER NOT NULL DEFAULT 0,
    assistant_message_count INTEGER NOT NULL DEFAULT 0,
    tool_call_count INTEGER NOT NULL DEFAULT 0,
    total_input_tokens INTEGER,
    total_output_tokens INTEGER,
    compaction_count INTEGER DEFAULT 0,

    -- Content analysis (flat — for filtering)
    session_summary TEXT,
    primary_intent TEXT,
    narrative_arc TEXT,
    complexity_score INTEGER CHECK(complexity_score BETWEEN 1 AND 10),
    outcome TEXT CHECK(outcome IN ('success','partial_success','failure','abandoned','ongoing')),

    -- Content analysis (JSON — variable-length, read-heavy)
    secondary_intents TEXT DEFAULT '[]',
    topic_tags TEXT DEFAULT '[]',
    topic_evolution TEXT DEFAULT '[]',

    -- Quality scores (flat — for dashboards and alerts)
    session_quality_score INTEGER CHECK(session_quality_score BETWEEN 1 AND 10),
    ai_effectiveness_score INTEGER CHECK(ai_effectiveness_score BETWEEN 1 AND 10),
    tool_usage_quality_score INTEGER CHECK(tool_usage_quality_score BETWEEN 1 AND 10),
    frustration_level INTEGER CHECK(frustration_level BETWEEN 1 AND 10),
    success_rate REAL CHECK(success_rate BETWEEN 0.0 AND 1.0),

    -- Quality narratives (text — for human reading)
    what_worked TEXT,
    what_failed TEXT,
    quality_justification TEXT,

    -- Decisions and corrections (JSON — variable-length arrays)
    decisions_made TEXT DEFAULT '[]',
    corrections TEXT DEFAULT '[]',
    repeated_instructions TEXT DEFAULT '[]',
    frustration_signals TEXT DEFAULT '[]',

    -- Knowledge (JSON — flexible structured data)
    new_knowledge TEXT DEFAULT '[]',
    rules_established TEXT DEFAULT '[]',

    -- Tool usage (JSON — per-tool stats)
    tool_usage_stats TEXT DEFAULT '[]',
    most_used_tools TEXT DEFAULT '[]',
    tool_failures TEXT DEFAULT '[]',

    -- Embedding for semantic search
    summary_embedding BLOB
);
```

The JSON columns use SQLite's `json_each()` table-valued function for querying. When a JSON field becomes a frequent filter target, promote it without migration:

```sql
-- Note: tool_usage_stats stores a JSON array (e.g., [{"tool_name": "Read", "count": 42}])
-- Default should be '[]' not '{}' to match this array access pattern
ALTER TABLE session_enrichments ADD COLUMN primary_tool TEXT
    GENERATED ALWAYS AS (json_extract(tool_usage_stats, '$[0].tool_name')) VIRTUAL;
CREATE INDEX idx_primary_tool ON session_enrichments(primary_tool);
```

**Three supporting tables handle many-to-many relationships and the processing pipeline.** `session_topics` enables reverse lookups by topic. `session_tool_usage` enables per-tool analytics across sessions. `enrichment_jobs` tracks the multi-pass processing state, storing intermediate outputs from each pass so failed jobs can resume without reprocessing.

For full-text search, an FTS5 virtual table in content-sync mode indexes the narrative fields (summary, what_worked, what_failed, quality_justification) with Porter stemming. Triggers keep it synchronized. For vector search, sqlite-vec stores the summary embedding as a BLOB in the main table — at **268K rows with 384-dim float32 vectors, brute-force cosine similarity runs in 50–200ms**, which is acceptable for batch analysis and tolerable for interactive queries. Binary quantization (384 bits instead of 384 floats) can accelerate this 30× if needed.

---

## 2. Route sessions by size through a tiered processing pipeline

The fundamental constraint is that sessions range from a few thousand tokens to 200K+, and the available models span from an 8K-context local LLM to Gemini Flash's 1M context window. Research on long-context LLM performance reveals a critical insight: **advertised context windows are not effective context windows**. Gemini Flash shows reliable performance up to ~100K tokens, but developer reports indicate degradation after that, with "lost in the middle" effects causing missed or confused information.

The optimal architecture is a three-tier routing system:

**Tier 1 (≤100K tokens, ~60–70% of sessions): Full-context Gemini Flash.** Send the entire session in a single API call with structured extraction prompts. This produces the highest quality results with the simplest implementation. Place the transcript first in the prompt and extraction instructions last — Anthropic's research shows **queries at the end of long contexts improve quality by up to 30%**.

**Tier 2 (100K–500K tokens): Chunked Gemini Flash.** Split into 2–5 overlapping segments with 15% overlap, process each segment, then run a lightweight merge/reconciliation pass. Split on message boundaries, never mid-message. Include speaker metadata and timestamps in the overlap region for context continuity.

**Tier 3 (>500K tokens or rate-limited): Map-reduce with local LLM.** The LLM×MapReduce approach from Tsinghua University (2024) is the gold standard here. Each chunk gets a structured extraction pass (the "map") producing metadata plus a **confidence score**. The "reduce" phase merges results using confidence-weighted reconciliation. Key finding: LLM×MapReduce with a small model (4B parameters) outperformed 70B-scale models on long-context benchmarks, because the small model processes focused chunks more accurately than a large model processes diffuse context.

```python
def route_session(token_count, gemini_available=True):
    if gemini_available and token_count <= 100_000:
        return "gemini-full-context"
    elif gemini_available and token_count <= 500_000:
        return "gemini-chunked"
    else:
        return "local-map-reduce"
```

For the local map-reduce path with GLM-4.7-Flash's 8K context: use **4,000–6,000 token chunks with 15–20% overlap**, split on message boundaries. This leaves room for the extraction prompt and output tokens. Each map output should include a rationale, extracted fields, and a per-field confidence score. The reduce phase can use either Gemini (if available) or iterative local merging.

**Cost and throughput for 800 sessions via Gemini Flash free tier**: not feasible. At 20–100 requests per day on the free tier, processing would take **8–40 days**. Paid Tier 1 (just enabling billing, no minimum) costs roughly **$0.30/M input tokens**. Processing all 800 sessions (estimated 160M input tokens) costs approximately **$52 total, or ~$26 via the Batch API** at 50% discount. This is the pragmatic choice. Alternatively, the Gemini CLI provides dramatically better free-tier limits (60 RPM, 1,000 RPD).

---

## 3. Corrections must be first-class searchable entities, not embedded JSON

This is the most consequential architectural decision. Research across Zep, Mem0, Cognee, and knowledge graph literature converges on a clear answer: **corrections, learnings, and rules deserve their own table with their own lifecycle**. Embedding them only as JSON arrays inside session enrichments makes them invisible to cross-session analysis.

The key insight comes from Zep/Graphiti's temporal knowledge graph: facts change over time, and the system must track not just what's currently true but how knowledge evolved. Corrections follow a natural graduation pipeline:

- **Correction** (single instance, session-bound) → seen once, low confidence
- **Pattern** (2–3 occurrences across sessions) → reinforced, growing confidence  
- **Preference** (stable, high confidence) → consistent across many sessions
- **Rule** (permanent, always applied) → explicitly confirmed or 5+ consistent occurrences

This maps to a spaced-repetition-inspired confidence model: `effective_weight = base_confidence × (reinforcement_count ^ growth_factor) × e^(-days_since_last_use / half_life)`. Corrections that aren't reinforced decay; those that recur strengthen.

```sql
CREATE TABLE corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_session_id TEXT NOT NULL,
    source_chunk_id TEXT,
    correction_type TEXT NOT NULL,        -- preference, factual, style, process, tool_usage
    original_behavior TEXT,               -- what the agent did wrong
    corrected_behavior TEXT NOT NULL,     -- what the user wanted
    rule_text TEXT NOT NULL,              -- normalized, reusable rule statement
    confidence REAL DEFAULT 0.5,
    reinforcement_count INTEGER DEFAULT 1,
    half_life_days REAL DEFAULT 30.0,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    status TEXT DEFAULT 'correction',     -- correction → pattern → preference → rule
    tags TEXT DEFAULT '[]',
    embedding BLOB,
    FOREIGN KEY (source_session_id) REFERENCES session_enrichments(session_id)
);

CREATE TABLE correction_links (
    source_id INTEGER NOT NULL REFERENCES corrections(id),
    target_id INTEGER NOT NULL REFERENCES corrections(id),
    link_type TEXT NOT NULL,              -- duplicate, related, supersedes, contradicts
    similarity_score REAL,
    PRIMARY KEY (source_id, target_id)
);
```

**Why separate from session_enrichments?** Three reasons. First, corrections need their own embeddings for semantic deduplication — you need to find "user said don't use var" in session 47 when the same correction appears in session 312. Second, corrections accumulate metadata over their lifetime (reinforcement_count, confidence) that doesn't belong to any single session. Third, the `correction_links` table enables a graph of related corrections that spans sessions.

The `rule_text` field is critical: it's the **normalized, agent-consumable statement** derived from the raw correction context. "No, use const not var" becomes "Always use const instead of var in JavaScript/TypeScript files." This normalization step happens during extraction and makes corrections directly injectable into future agent prompts.

Mem0's AUDN (Add/Update/Delete/Noop) pattern provides the right operational model for processing new corrections: embed the candidate, search existing corrections by cosine similarity, and use the LLM to decide whether to ADD a new correction, UPDATE an existing one (incrementing reinforcement_count), DELETE a contradicted one, or NOOP.

---

## 4. Three-pass prompting extracts layered metadata without overwhelming the LLM

Research from Anthropic, OpenAI, and conversation intelligence platforms (Gong, Chorus, CallMiner) converges on a principle: **single-task prompts outperform multi-task prompts for extraction quality**. A three-pass strategy gives each pass the LLM's full attention on one cognitive task.

**Pass 1 — Structure and narrative** (broad understanding). Extract: session summary, primary intent, topic evolution, narrative arc, complexity score, outcome, message counts. This pass establishes the "what happened" frame that contextualizes everything else. The prompt should use XML tags for structure, place the transcript first with instructions after, and request JSON output matching a strict schema.

**Pass 2 — Detailed signal extraction** (specific signals). Takes Pass 1 output as context. Extract: decisions made (with rationale and outcome), user corrections (what was wrong, what was right, severity), repeated instructions, frustration signals (explicit and implicit), tool usage statistics, new knowledge established, rules the user expressed. This pass benefits from the narrative frame established in Pass 1, allowing it to focus on fine-grained extraction without needing to understand the overall arc.

**Pass 3 — Quality scoring** (judgment). Takes Pass 1 and Pass 2 outputs — not the full transcript. Score: session quality, AI effectiveness, tool usage quality, communication quality, recovery quality. Also produce what_worked, what_failed, and success_rate. This pass works from distilled information, making it suitable for a smaller model.

Key prompting techniques that significantly improve extraction quality:

- **"Quote before answering"**: Instruct the LLM to quote relevant transcript excerpts before making claims. This grounds responses in evidence and dramatically reduces hallucination on long documents.
- **Selective attention instructions**: "Focus primarily on user messages expressing intent, moments where direction changes, error/failure events, and the final outcome. Skim routine tool outputs confirming success."
- **Temperature 0.0–0.2** for extraction tasks ensures factual, deterministic outputs.
- **Structured output enforcement**: Use JSON schema constraints (Gemini's `response_schema` or function calling) for guaranteed schema compliance.

For the map-reduce path, Pass 1 runs independently on each chunk (the "map"). Passes 2 and 3 run on the merged Pass 1 output (the "reduce"). Each map output includes per-field confidence scores for conflict resolution during merging.

---

## 5. Cross-session pattern detection needs both real-time dedup and periodic batch analysis

Detecting that the same correction appears across sessions requires two complementary mechanisms: **real-time similarity matching during ingestion** and **periodic batch clustering** for pattern discovery.

**Real-time dedup during correction ingestion** uses sqlite-vec. When a new correction is extracted, embed it and query the vec_corrections table for neighbors within cosine distance **< 0.15** (i.e., cosine similarity > 0.85 — `vec_distance_cosine` returns distance where lower = more similar). If a near-duplicate exists, invoke the AUDN loop: the LLM decides whether to merge (incrementing reinforcement_count and updating confidence) or keep as distinct.

```sql
-- Find similar existing corrections (distance < 0.15 = similarity > 0.85)
SELECT c.id, c.rule_text, c.confidence, c.reinforcement_count,
       vec_distance_cosine(vc.embedding, :new_embedding) as distance
FROM vec_corrections vc
JOIN corrections c ON c.id = vc.correction_id
WHERE vec_distance_cosine(vc.embedding, :new_embedding) < 0.15
ORDER BY distance
LIMIT 5;
```

**Periodic batch analysis** (nightly or after every N sessions) handles broader pattern discovery that real-time matching misses:

1. Fetch all corrections added since the last batch run
2. Run HDBSCAN clustering on the full correction embedding space (HDBSCAN is preferred over K-means because it auto-detects cluster count and handles varying cluster densities)
3. For each cluster, generate a descriptive label using the LLM
4. Within clusters, merge near-duplicates (cosine similarity > 0.85)
5. Promote corrections exceeding confidence thresholds (correction → pattern at 2+ occurrences, pattern → preference at consistent cross-session presence, preference → rule at 5+ reinforcements or explicit user confirmation)
6. Decay old corrections not reinforced recently (reduce effective_weight)
7. Generate a summary report of emerging patterns

This mirrors how **log analysis systems** (Amazon CloudWatch Patterns, LogCluster) detect recurring issues: knowledge base initialization followed by online matching against existing clusters, with unmatched items seeding new clusters. Customer support ticket clustering research confirms this two-phase approach (real-time classification + periodic re-clustering) as the most practical architecture.

For the 268K-row scale, UMAP dimensionality reduction before HDBSCAN improves clustering quality on high-dimensional embeddings. The batch job stores cluster assignments in a `correction_clusters` table, enabling queries like "show me the top 10 most reinforced correction clusters."

---

## 6. Session enrichment should reference chunks bidirectionally

The relationship between session-level and chunk-level enrichment should be **bidirectional but loosely coupled**. Session enrichments reference specific chunk IDs where key events occurred (a correction at chunk #47, a decision at chunk #112), and chunk-level retrieval can be enhanced by session-level metadata.

**Session → Chunk references**: The JSON arrays in session_enrichments (decisions_made, corrections, frustration_signals) should include a `chunk_id` or `message_index` field pointing to the specific chunk where the event was detected. This enables drill-down: a user investigating a low-quality session can jump directly to the problematic exchange.

**Chunk → Session enhancement**: When retrieving chunks via semantic search (the primary BrainLayer use case), session-level metadata acts as a **reranking signal**. A chunk from a session with quality_score=9 and outcome=success is more likely to contain reliable information than the same semantic match from a session with quality_score=2 and outcome=failure. Implement this as a weighted score:

```sql
SELECT c.*, se.session_quality_score, se.outcome,
    (0.7 * semantic_score + 0.2 * (se.session_quality_score / 10.0) 
     + 0.1 * CASE se.outcome WHEN 'success' THEN 1.0 
            WHEN 'partial_success' THEN 0.6 ELSE 0.2 END) as enhanced_score
FROM chunks c
JOIN session_enrichments se ON se.session_id = c.session_id
ORDER BY enhanced_score DESC;
```

This pattern draws from Kernel Memory's tags system (metadata attached at the document level flows down to enhance chunk-level retrieval) and Cognee's strict provenance linking (inferred information always links back to source documents). The key principle is that **session context enriches chunk retrieval without replacing it** — chunks remain the primary unit of semantic search, but session metadata provides quality signals and navigational context.

**Do not store session-level embeddings in the same vec0 table as chunk embeddings.** They exist at different semantic granularities. Use separate vec0 tables (or separate embedding columns) so similarity searches within each level remain clean.

---

## 7. Lessons from Zep, Mem0, Cognee, and Letta shape the design

Each existing project contributes a distinct architectural insight that BrainLayer should incorporate:

**Zep/Graphiti's bi-temporal model** is the most sophisticated approach to handling changing facts. Every fact carries four timestamps: `t_created`, `t_expired` (ingestion time) and `t_valid`, `t_invalid` (event time). When new information contradicts old, the old edge gets invalidated rather than deleted. For BrainLayer's corrections table, this translates to `first_seen_at`, `last_seen_at`, and `invalidated_at` columns — never delete corrections, only mark them superseded.

**Mem0's incremental AUDN loop** (Add/Update/Delete/Noop) provides the operational model for processing corrections. Rather than reanalyzing entire sessions when new information arrives, process corrections incrementally: embed, search, decide operation. This scales to continuous enrichment as new sessions arrive without reprocessing the entire corpus.

**Cognee's Memify pattern** — post-processing that compresses repeated patterns from session traces into reusable meta-structures — maps directly to the batch clustering pipeline described above. Cognee also validates that memory updates actually improve agent performance, a principle BrainLayer should adopt: track whether surfacing a correction during retrieval actually prevents the error from recurring.

**Letta/MemGPT's two-tier memory** (editable core memory + archival vector store) suggests that BrainLayer's most important corrections and rules should be promoted to a "core rules" set that's always injected into the agent prompt, while the full corrections database serves as searchable archival memory.

**LangMem's schema-based extraction** using Pydantic models for structured memory types (semantic, episodic, procedural) provides a clean software pattern: define extraction schemas as data classes, use them to constrain LLM output, and store the validated results directly in SQLite.

One pattern all projects share: **LLM-based conflict resolution over brittle rules**. When two corrections contradict each other or a new fact conflicts with an old one, delegate the resolution decision to the LLM rather than building complex rule-based logic. This is both more robust and simpler to maintain.

---

## Conclusion: a practical implementation roadmap

The system described here is ambitious but modular — each component delivers value independently and can be built incrementally. **Start with the session_enrichments table and the three-pass Gemini pipeline**, which immediately makes 800+ sessions searchable by quality, intent, outcome, and narrative. Add the corrections table and real-time dedup second, since this is where the highest long-term value accumulates. Add batch clustering and correction graduation third, once enough corrections exist to make pattern detection meaningful.

Three non-obvious insights emerged from this research. First, **effective context windows are roughly half of advertised context windows** — plan the tiered routing around 100K tokens, not 1M. Second, **the correction graduation pipeline (correction → pattern → preference → rule) is the single most valuable feature** for an AI agent memory system, yet no existing open-source project implements it fully. Third, **session-level quality scores as chunk reranking signals** create a virtuous cycle: high-quality sessions get their chunks surfaced more often, which means the agent learns from its best interactions rather than its worst.

The total investment to process all 800 existing sessions through Gemini Flash Batch API is approximately **$26**. The schema handles 268K+ rows comfortably within SQLite's capabilities, with sqlite-vec providing 50–200ms vector search and FTS5 enabling instant keyword search. The architecture is designed to grow: as BrainLayer indexes more sessions, the corrections knowledge base becomes increasingly valuable, with each new correction either reinforcing existing patterns or revealing new ones.