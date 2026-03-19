"""Enrichment pilot: 100 diverse chunks through Gemini 2.5 Flash with faceted tag prompt."""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai
from google.genai import types
from src.brainlayer.vector_store import VectorStore
from src.brainlayer.paths import get_db_path

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
    gold = list(cursor.execute(
        "SELECT id, content, source, tags, char_count FROM chunks "
        "WHERE tags LIKE '%brain_store%' AND char_count > 30 LIMIT 26"
    ))
    chunks.extend(gold)
    gold_ids = {r[0] for r in gold}
    print(f"Gold validation chunks: {len(gold)}")

    # claude_code: mix of lengths (short, medium, long)
    for length_range, limit in [("AND char_count BETWEEN 50 AND 200", 15),
                                 ("AND char_count BETWEEN 200 AND 1000", 15),
                                 ("AND char_count BETWEEN 1000 AND 5000", 10)]:
        rows = list(cursor.execute(
            f"SELECT id, content, source, tags, char_count FROM chunks "
            f"WHERE source = 'claude_code' {length_range} AND id NOT IN ({','.join('?' for _ in gold_ids)}) "
            f"ORDER BY RANDOM() LIMIT ?",
            list(gold_ids) + [limit]
        ))
        chunks.extend(rows)
        for r in rows:
            gold_ids.add(r[0])

    # youtube
    rows = list(cursor.execute(
        f"SELECT id, content, source, tags, char_count FROM chunks "
        f"WHERE source = 'youtube' AND char_count > 50 AND id NOT IN ({','.join('?' for _ in gold_ids)}) "
        f"ORDER BY RANDOM() LIMIT 15",
        list(gold_ids)
    ))
    chunks.extend(rows)
    for r in rows:
        gold_ids.add(r[0])

    # whatsapp
    rows = list(cursor.execute(
        f"SELECT id, content, source, tags, char_count FROM chunks "
        f"WHERE source = 'whatsapp' AND char_count > 10 AND id NOT IN ({','.join('?' for _ in gold_ids)}) "
        f"ORDER BY RANDOM() LIMIT 10",
        list(gold_ids)
    ))
    chunks.extend(rows)
    for r in rows:
        gold_ids.add(r[0])

    # manual (brain_store entries without brain_store tag)
    remaining = 100 - len(chunks)
    if remaining > 0:
        rows = list(cursor.execute(
            f"SELECT id, content, source, tags, char_count FROM chunks "
            f"WHERE source = 'manual' AND id NOT IN ({','.join('?' for _ in gold_ids)}) "
            f"ORDER BY RANDOM() LIMIT ?",
            list(gold_ids) + [remaining]
        ))
        chunks.extend(rows)

    return chunks[:100]


def call_gemini(client, content):
    """Call Gemini 2.5 Flash with the enrichment prompt. Returns parsed JSON or error."""
    # Truncate very long chunks to 4000 chars to stay within limits
    truncated = content[:4000] if len(content) > 4000 else content
    prompt = PROMPT_TEMPLATE.replace("{chunk_content}", truncated)

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        text = response.text.strip()
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    except Exception as e:
        return None, f"API error: {e}"


def analyze_results(results, chunks_meta):
    """Analyze pilot results and return stats."""
    valid = [r for r in results if r["parsed"] is not None and isinstance(r["parsed"], dict)]
    invalid = [r for r in results if r["parsed"] is None or not isinstance(r["parsed"], dict)]

    # Collect all tags
    all_topics = []
    all_activities = []
    all_domains = []
    confidences = []
    noise_count = 0

    for r in valid:
        p = r["parsed"]
        topics = p.get("b_topics", [])
        if not isinstance(topics, list):
            topics = []
        all_topics.extend(topics)
        if "_noise" in topics:
            noise_count += 1
        act = p.get("c_activity", "")
        if act:
            all_activities.append(act)
        domains = p.get("d_domain", [])
        if not isinstance(domains, list):
            domains = []
        all_domains.extend(domains)
        conf = p.get("e_confidence")
        if conf is not None:
            confidences.append(conf)

    # Topic frequency
    topic_freq = {}
    for t in all_topics:
        topic_freq[t] = topic_freq.get(t, 0) + 1

    # Activity frequency
    act_freq = {}
    for a in all_activities:
        act_freq[a] = act_freq.get(a, 0) + 1

    # Domain frequency
    dom_freq = {}
    for d in all_domains:
        dom_freq[d] = dom_freq.get(d, 0) + 1

    # Confidence distribution
    conf_buckets = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
    for c in confidences:
        if c < 0.3:
            conf_buckets["0.0-0.3"] += 1
        elif c < 0.5:
            conf_buckets["0.3-0.5"] += 1
        elif c < 0.7:
            conf_buckets["0.5-0.7"] += 1
        elif c < 0.9:
            conf_buckets["0.7-0.9"] += 1
        else:
            conf_buckets["0.9-1.0"] += 1

    # Compare with existing tags
    tag_comparison = []
    for r in valid:
        old_tags = r.get("old_tags", "")
        if old_tags:
            try:
                old_list = json.loads(old_tags) if isinstance(old_tags, str) else old_tags
            except (json.JSONDecodeError, TypeError):
                old_list = []
            new_topics = r["parsed"].get("b_topics", [])
            if old_list or new_topics:
                tag_comparison.append({
                    "chunk_id": r["chunk_id"][:40],
                    "source": r["source"],
                    "old": old_list[:5] if old_list else [],
                    "new": new_topics,
                    "confidence": r["parsed"].get("e_confidence", 0),
                })

    return {
        "total": len(results),
        "valid_json": len(valid),
        "invalid_json": len(invalid),
        "errors": [r["error"] for r in invalid],
        "unique_topics": len(set(all_topics)),
        "topic_freq": dict(sorted(topic_freq.items(), key=lambda x: -x[1])[:30]),
        "activity_freq": dict(sorted(act_freq.items(), key=lambda x: -x[1])),
        "domain_freq": dict(sorted(dom_freq.items(), key=lambda x: -x[1])),
        "confidence_dist": conf_buckets,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "noise_count": noise_count,
        "tag_comparisons": tag_comparison[:20],
    }


def write_report(stats, results_path):
    """Write results to markdown."""
    lines = [
        "# Enrichment Pilot Results — Faceted Tag Prompt v2",
        "",
        f"> **Date:** 2026-03-19",
        f"> **Model:** Gemini 2.5 Flash",
        f"> **Chunks tested:** {stats['total']}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Valid JSON responses:** {stats['valid_json']}/{stats['total']} ({stats['valid_json']/stats['total']*100:.1f}%)",
        f"- **Parse failures:** {stats['invalid_json']}",
        f"- **Unique topic tags generated:** {stats['unique_topics']}",
        f"- **Average confidence:** {stats['avg_confidence']:.3f}",
        f"- **Noise-flagged chunks:** {stats['noise_count']}",
        "",
        "---",
        "",
        "## Confidence Distribution",
        "",
        "| Range | Count |",
        "|-------|-------|",
    ]
    for bucket, count in stats["confidence_dist"].items():
        lines.append(f"| {bucket} | {count} |")

    lines.extend([
        "",
        "---",
        "",
        "## Topic Tags (top 30)",
        "",
        "| Tag | Count |",
        "|-----|-------|",
    ])
    for tag, count in stats["topic_freq"].items():
        lines.append(f"| `{tag}` | {count} |")

    lines.extend([
        "",
        "---",
        "",
        "## Activity Distribution",
        "",
        "| Activity | Count |",
        "|----------|-------|",
    ])
    for act, count in stats["activity_freq"].items():
        lines.append(f"| `{act}` | {count} |")

    lines.extend([
        "",
        "---",
        "",
        "## Domain Distribution",
        "",
        "| Domain | Count |",
        "|--------|-------|",
    ])
    for dom, count in stats["domain_freq"].items():
        lines.append(f"| `{dom}` | {count} |")

    lines.extend([
        "",
        "---",
        "",
        "## Tag Comparison: Old vs New (sample)",
        "",
        "| Source | Old Tags | New Topics | Confidence |",
        "|--------|----------|------------|------------|",
    ])
    for comp in stats["tag_comparisons"]:
        old_str = ", ".join(comp["old"][:3]) if comp["old"] else "(none)"
        new_str = ", ".join(comp["new"][:3]) if comp["new"] else "(none)"
        lines.append(f"| {comp['source']} | {old_str} | {new_str} | {comp['confidence']:.2f} |")

    if stats["errors"]:
        lines.extend([
            "",
            "---",
            "",
            "## Errors",
            "",
        ])
        for err in stats["errors"][:10]:
            lines.append(f"- {err}")

    lines.extend([
        "",
        "---",
        "",
        "## Verdict",
        "",
        "**TODO:** Fill in after reviewing results.",
        "",
    ])

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines))
    print(f"\nReport written to {results_path}")


def main():
    client = genai.Client(api_key=API_KEY)

    store = VectorStore(get_db_path())
    print("Selecting 100 diverse chunks...")
    chunks = select_chunks(store)
    print(f"Selected {len(chunks)} chunks")

    # Source distribution
    source_dist = {}
    for c in chunks:
        s = c[2] or "unknown"
        source_dist[s] = source_dist.get(s, 0) + 1
    print(f"Source distribution: {source_dist}")

    # Run enrichment
    results = []
    start = time.time()
    for i, (chunk_id, content, source, old_tags, char_count) in enumerate(chunks):
        parsed, error = call_gemini(client, content)
        results.append({
            "chunk_id": chunk_id,
            "source": source,
            "old_tags": old_tags,
            "char_count": char_count,
            "parsed": parsed,
            "error": error,
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/100] {rate:.1f} chunks/sec, elapsed {elapsed:.0f}s")
        # Rate limit: 15 RPM for free tier, but flash should be higher
        # Small delay to be safe
        time.sleep(0.5)

    elapsed = time.time() - start
    print(f"\nCompleted {len(results)} chunks in {elapsed:.0f}s ({len(results)/elapsed:.1f}/sec)")

    # Save raw JSON FIRST (before analysis, so data is never lost)
    raw_path = RESULTS_PATH.with_suffix(".json")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Raw results saved to {raw_path}")

    # Analyze
    stats = analyze_results(results, chunks)

    # Write report
    write_report(stats, RESULTS_PATH)

    # Print summary
    print(f"\n{'='*60}")
    print(f"PILOT SUMMARY")
    print(f"{'='*60}")
    print(f"Valid JSON: {stats['valid_json']}/{stats['total']}")
    print(f"Unique topics: {stats['unique_topics']}")
    print(f"Avg confidence: {stats['avg_confidence']:.3f}")
    print(f"Noise flagged: {stats['noise_count']}")
    print(f"Top topics: {list(stats['topic_freq'].items())[:10]}")


if __name__ == "__main__":
    main()
