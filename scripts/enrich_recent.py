"""Enrich recent chunks with faceted tags via Gemini 2.5 Flash."""

import json
import os
import sys
import time
import apsw

from google import genai
from google.genai import types

API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyC4tEa3Zh01rvrAB49nFA57NDBEClv3VOo")
MODEL = "gemini-2.5-flash"
DB_PATH = os.path.expanduser("~/.local/share/brainlayer/brainlayer.db")
BATCH_COMMIT = 50
MAX_CHUNKS = 200  # cap per run

PROMPT = """You are a knowledge base tagger for a personal multi-project development knowledge base. Your job is to identify WHAT SPECIFIC THING each chunk is about — not the kind of work being done.

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


def main():
    client = genai.Client(api_key=API_KEY)
    conn = apsw.Connection(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 5000")
    cursor = conn.cursor()

    # Select recent chunks without faceted tags, ordered by importance desc
    rows = list(cursor.execute("""
        SELECT id, content, tags, importance
        FROM chunks
        WHERE created_at > datetime('now', '-7 days')
          AND (tags IS NULL OR tags NOT LIKE '%dom:%')
          AND char_count > 50
          AND char_count < 4000
        ORDER BY importance DESC NULLS LAST, created_at DESC
        LIMIT ?
    """, (MAX_CHUNKS,)))

    print(f"Found {len(rows)} chunks to enrich (capped at {MAX_CHUNKS})")
    if not rows:
        return

    enriched = 0
    errors = 0
    samples = []
    start = time.time()

    for i, (chunk_id, content, old_tags, importance) in enumerate(rows):
        truncated = content[:4000]
        prompt = PROMPT.replace("{chunk_content}", truncated)

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
            parsed = json.loads(text)

            # Handle array responses (take first)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                errors += 1
                continue

            # Build new tags: merge old + new faceted
            topics = parsed.get("b_topics", [])
            activity = parsed.get("c_activity", "")
            domains = parsed.get("d_domain", [])
            confidence = parsed.get("e_confidence", 0)

            new_tags = list(topics)
            if activity:
                new_tags.append(activity)
            new_tags.extend(domains)

            # Merge with old tags (keep old, add new)
            if old_tags:
                try:
                    old_list = json.loads(old_tags)
                    if isinstance(old_list, list):
                        for t in old_list:
                            if t not in new_tags:
                                new_tags.append(t)
                except (json.JSONDecodeError, TypeError):
                    pass

            tags_json = json.dumps(new_tags)

            # Update chunk
            cursor.execute(
                "UPDATE chunks SET tags = ?, tag_confidence = ? WHERE id = ?",
                (tags_json, confidence, chunk_id)
            )
            enriched += 1

            # Collect samples
            if len(samples) < 10:
                samples.append({
                    "id": chunk_id[:40],
                    "topics": topics,
                    "activity": activity,
                    "domains": domains,
                    "confidence": confidence,
                })

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error on {chunk_id[:30]}: {e}")

        # Progress + rate limiting
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(rows)}] {rate:.1f}/sec, enriched={enriched}, errors={errors}")

        # Commit every BATCH_COMMIT
        if enriched > 0 and enriched % BATCH_COMMIT == 0:
            print(f"  Checkpoint: committed {enriched} enriched chunks")

        time.sleep(0.3)  # rate limit safety

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s — enriched {enriched}, errors {errors}")
    print(f"\nSample results:")
    for s in samples:
        print(f"  {s['id']:40s} topics={s['topics']}, {s['activity']}, {s['domains']}, conf={s['confidence']}")


if __name__ == "__main__":
    main()
