"""Enrichment pilot: 100 diverse chunks through Gemini 2.5 Flash with faceted tag prompt."""

import sys as _gate_sys

# Enrichment retired (2026-10-07). Keep this gate until final script deletion.
_gate_sys.exit("RETIRED: enrichment is removed; this script cannot run.")

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Config ──────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY environment variable required")
    sys.exit(1)
MODEL = "gemini-2.5-flash"
RESULTS_PATH = Path.home() / "Gits/orchestrator/docs.local/plans/enrichment-pilot-results.md"

PROMPT_TEMPLATE = """You are a knowledge base tagger for a personal multi-project development knowledge base. Your job is to identify WHAT SPECIFIC THING each chunk is about — not the kind of work being done.

## Critical distinction

GOOD tags describe the SUBJECT: "brainlayer-search-quality", "6pm-confirmation-flow", "importance-calibration"
BAD tags describe the FORMAT: "typescript", "debugging", "code-review", "feature-dev"

Ask yourself: "If someone searches for this topic in 6 months, what words would they use?" Tag with THOSE words.

## Output schema (JSON)

Return a JSON object with these fields in this exact order:

- **a_reasoning** (string): 1-2 sentences explaining what specific subject this chunk discusses. Think before tagging.
- **b_topics** (string[]): 1-3 object tags — specific, hyphenated, 2-4 words. Use existing tags when they fit: brainlayer-search-quality, importance-calibration, enrichment-pipeline, auto-context-hooks, cmux-terminal-orchestration, voicelayer-tts-stt, coachclaude-scheduling, sprint-planning-methodology, skill-ecosystem, eval-runner, 6pm-confirmation-flow, knowledge-graph-rebuild, dedup-strategy, tag-taxonomy-redesign, compaction-survival, rate-limit-coordination, pr-loop-workflow, brain-digest-stub, golems-monorepo, etanheyman-portfolio. Create new tags following the same pattern when needed. Use ["_noise"] for system noise with no informational content.
- **c_activity** (string): Exactly one of: act:debugging, act:implementing, act:designing, act:reviewing, act:researching, act:planning, act:configuring, act:refactoring, act:testing, act:learning
- **d_domain** (string[]): 0-3 technology domains from: dom:typescript, dom:python, dom:swift, dom:sql, dom:react, dom:convex, dom:supabase, dom:mcp, dom:vertex-ai, dom:ollama, dom:mlx, dom:git, dom:telegram, dom:whatsapp, dom:macos, dom:cli, dom:css, dom:html, dom:docker, dom:railway, dom:linear, dom:obsidian. Empty array if no specific technology.
- **e_confidence** (number): 0.0-1.0 confidence in your tagging. Below 0.5 = low-content chunk.

## Examples

Chunk: "Phase 1 importance calibration implemented (PR #93). Heuristic SQL fix deflates importance inflation from 40.8% >= 7 to 7.6%."
Output: {"a_reasoning": "Completed milestone for BrainLayer importance scoring. Quantified SQL fix results.", "b_topics": ["importance-calibration", "brainlayer-search-quality"], "c_activity": "act:implementing", "d_domain": ["dom:sql"], "e_confidence": 0.95}

Chunk: "Found the double-message bug in 6pm-mini: flexibility message sent before fail check."
Output: {"a_reasoning": "Bug fix in 6pm dating app messaging flow.", "b_topics": ["6pm-confirmation-flow"], "c_activity": "act:debugging", "d_domain": ["dom:typescript", "dom:convex"], "e_confidence": 0.92}

Chunk: "R18 cmux event-driven patterns: tmux control mode (-CC) provides structured event streams."
Output: {"a_reasoning": "Research on cmux terminal architecture using tmux control mode.", "b_topics": ["cmux-terminal-orchestration"], "c_activity": "act:researching", "d_domain": ["dom:cli", "dom:mcp"], "e_confidence": 0.93}

Chunk: "[Request interrupted by user]"
Output: {"a_reasoning": "System noise, no content.", "b_topics": ["_noise"], "c_activity": "act:configuring", "d_domain": [], "e_confidence": 0.99}

Chunk: "כן אחי, אני בדרך"
Output: {"a_reasoning": "Short Hebrew acknowledgment, no technical content.", "b_topics": [], "c_activity": "act:planning", "d_domain": ["dom:whatsapp"], "e_confidence": 0.30}

Chunk: "OK so the coach should check WHOOP recovery score first thing in the morning, then adjust the workout plan."
Output: {"a_reasoning": "Voice discussion about coachClaude morning health workflow with WHOOP thresholds.", "b_topics": ["coachclaude-scheduling"], "c_activity": "act:designing", "d_domain": [], "e_confidence": 0.91}

## Now tag this chunk:

{chunk_content}"""


def select_chunks(store):
    """Select 100 diverse chunks: mix of sources, lengths, include gold validation."""
    cursor = store.conn.cursor()
    chunks = []

    # Gold validation: chunks with brain_store tags (up to 26)
    gold = list(
        cursor.execute(
            "SELECT id, content, source, tags, char_count FROM chunks "
            "WHERE tags LIKE '%brain_store%' AND char_count > 30 LIMIT 26"
        )
    )
    chunks.extend(gold)
    gold_ids = {r[0] for r in gold}
    print(f"Gold validation chunks: {len(gold)}")

    # claude_code: mix of lengths (short, medium, long)
    for length_range, limit in [
        ("AND char_count BETWEEN 50 AND 200", 15),
        ("AND char_count BETWEEN 200 AND 1000", 15),
        ("AND char_count BETWEEN 1000 AND 5000", 10),
    ]:
        rows = list(
            cursor.execute(
                f"SELECT id, content, source, tags, char_count FROM chunks "
                f"WHERE source = 'claude_code' {length_range} AND id NOT IN ({','.join('?' for _ in gold_ids)}) "
                f"ORDER BY RANDOM() LIMIT ?",
                list(gold_ids) + [limit],
            )
        )
        chunks.extend(rows)
        for r in rows:
            gold_ids.add(r[0])

    # youtube
    rows = list(
        cursor.execute(
            f"SELECT id, content, source, tags, char_count FROM chunks "
            f"WHERE source = 'youtube' AND char_count > 50 AND id NOT IN ({','.join('?' for _ in gold_ids)}) "
            f"ORDER BY RANDOM() LIMIT 15",
            list(gold_ids),
        )
    )
    chunks.extend(rows)
    for r in rows:
        gold_ids.add(r[0])

    # whatsapp
    rows = list(
        cursor.execute(
            f"SELECT id, content, source, tags, char_count FROM chunks "
            f"WHERE source = 'whatsapp' AND char_count > 10 AND id NOT IN ({','.join('?' for _ in gold_ids)}) "
            f"ORDER BY RANDOM() LIMIT 10",
            list(gold_ids),
        )
    )
    chunks.extend(rows)
    for r in rows:
        gold_ids.add(r[0])

    # manual (brain_store entries without brain_store tag)
    remaining = 100 - len(chunks)
    if remaining > 0:
        rows = list(
            cursor.execute(
                f"SELECT id, content, source, tags, char_count FROM chunks "
                f"WHERE source = 'manual' AND id NOT IN ({','.join('?' for _ in gold_ids)}) "
                f"ORDER BY RANDOM() LIMIT ?",
                list(gold_ids) + [remaining],
            )
        )
        chunks.extend(rows)

    return chunks[:100]
