#!/usr/bin/env python3
"""
BrainLayer UserPromptSubmit Hook — auto-searches memories relevant to the user's prompt.

Uses adaptive injection:
  - Hybrid search (FTS5 + vector) when embeddings/sqlite-vec are available
  - Score-gated injection of 1-5 chunks based on RRF confidence
  - FTS-only fallback when hybrid search is unavailable

Output: plain text to stdout (injected as Claude context).
Target: <500ms total.
"""

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

from brainlayer.phonetic import looks_hebrew, phonetic_key, phonetic_tokens

DEADLINE_MS = 450
RRF_K = 60
HIGH_CONFIDENCE_THRESHOLD = 0.015
MODERATE_CONFIDENCE_THRESHOLD = 0.010
LIGHT_CONFIDENCE_THRESHOLD = 0.005
MAX_ADAPTIVE_INJECTION = 5
MAX_HYBRID_CANDIDATES = 8

# Prompts shorter than this are probably greetings/commands — skip search
MIN_PROMPT_LENGTH = 15
HEBREW_CANDIDATE_RE = re.compile(r"[\u0590-\u05FF]{2,}")


def should_activate():
    """Conditional activation gate — skip hook when not needed.

    Checks (in order, cheapest first):
    1. BRAINLAYER_HOOKS_DISABLED=1 env var → skip all BrainLayer hooks
    2. Non-interactive mode (--print) → skip memory search
    3. BRAINLAYER_HOOKS_LIGHT=1 → reduce to 2 results (overnight workers)

    Returns (activate: bool, light_mode: bool).
    """
    if os.environ.get("BRAINLAYER_HOOKS_DISABLED") == "1":
        return False, False

    if os.environ.get("CLAUDE_NON_INTERACTIVE") == "1":
        return False, False

    light = os.environ.get("BRAINLAYER_HOOKS_LIGHT") == "1"

    return True, light


# Trigger words that activate deep mode (more results)
DEEP_TRIGGERS = {
    "remember",
    "last time",
    "previous",
    "previously",
    "before",
    "history",
    "earlier",
    "we discussed",
    "we decided",
    "we talked",
    "recall",
    "forgot",
    "what was",
    "what were",
    "when did",
    "how did",
    "brainlayer",
}

# Keywords that signal assumption-prone prompts (personal facts, biography, specs)
# When detected: inject a search-before-assume reminder.
# Source: Phase 3 session mining — agents assumed M4 Max (wrong), tax history (wrong),
# Avi Tour (voice transcription error). All would have been caught by brain_search.
ASSUME_TRIGGERS = {
    "hardware",
    "laptop",
    "macbook",
    "machine",
    "specs",
    "ram",
    "cpu",
    "biography",
    "background",
    "tax",
    "salary",
    "income",
    "family",
    "partner",
    "wife",
    "girlfriend",
    "husband",
    "birthday",
    "age",
    "born",
    "address",
    "apartment",
    "city",
    "neighborhood",
    "research prompt",
    "research summary",
}

# Common English stop words to skip during keyword extraction
STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "it",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "but",
    "not",
    "with",
    "this",
    "that",
    "from",
    "by",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "must",
    "need",
    "let",
    "me",
    "my",
    "i",
    "you",
    "your",
    "we",
    "our",
    "they",
    "them",
    "their",
    "he",
    "she",
    "his",
    "her",
    "its",
    "if",
    "then",
    "else",
    "when",
    "where",
    "how",
    "what",
    "which",
    "who",
    "why",
    "so",
    "just",
    "also",
    "very",
    "too",
    "up",
    "out",
    "about",
    "into",
    "over",
    "after",
    "some",
    "any",
    "all",
    "no",
    "yes",
    "ok",
    "okay",
    "please",
    "thanks",
    "thank",
    "hey",
    "hi",
    "hello",
    "sure",
    "right",
    "well",
    "now",
    "here",
    "there",
    "like",
    "want",
    "think",
    "know",
    "see",
    "look",
    "make",
    "take",
    "get",
    "go",
    "come",
    "use",
    "try",
    "help",
    "tell",
    "give",
    "show",
    "work",
    "call",
    "run",
    "set",
    "add",
    "put",
    "keep",
    "find",
    "read",
    "write",
    "create",
    "build",
    "check",
    "start",
    "stop",
    "change",
    "move",
    "open",
    "close",
    "new",
    "old",
    "good",
    "bad",
    "big",
    "small",
    "first",
    "last",
    "next",
    "more",
    "less",
    "much",
    "many",
    "each",
    "every",
    "other",
    "same",
    "different",
    "own",
    "still",
    "already",
    "again",
    "even",
    "really",
    "actually",
    "probably",
    "maybe",
    "file",
    "thing",
    "way",
    "something",
    "anything",
}

_CANONICAL_DB = os.path.expanduser("~/.local/share/brainlayer/brainlayer.db")


def get_db_path():
    env = os.environ.get("BRAINLAYER_DB")
    if env and os.path.exists(env):
        return env
    if os.path.exists(_CANONICAL_DB):
        return _CANONICAL_DB
    return None


def is_deep_mode(prompt_lower):
    for trigger in DEEP_TRIGGERS:
        if trigger in prompt_lower:
            return True
    return False


def extract_keywords(prompt):
    """Extract meaningful keywords from the prompt for FTS5 search."""
    # Remove URLs, paths, code blocks
    text = re.sub(r"https?://\S+", "", prompt)
    text = re.sub(r"[/~]\S+", "", text)
    text = re.sub(r"`[^`]+`", "", text)

    # Extract words (keep hyphens for compound terms like "6pm-mini")
    words = re.findall(r"[a-zA-Z0-9][\w-]*", text.lower())

    # Filter out stop words and short words
    keywords = []
    seen = set()
    for w in words:
        if w not in STOP_WORDS and len(w) >= 2 and w not in seen:
            keywords.append(w)
            seen.add(w)

    return keywords[:8]  # Cap at 8 keywords for FTS5 performance


def truncate(text, max_chars=200):
    # Clean up multi-line content for compact display
    text = re.sub(r"\n+", " | ", text.strip())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def elapsed_ms(start):
    return (time.monotonic() - start) * 1000


def detect_entities_in_prompt(prompt, conn):
    """Detect known KG entity names in the prompt.

    Checks bigrams and single capitalized words (3+ chars) against kg_entities.
    Returns list of dicts: {id, name, entity_type}.
    Fast: exact SQL LOWER() match, no FTS5 overhead.

    Only injects context for high-signal entity types (person, company, agent).
    Technology/concept entities are too noisy for automatic injection.
    """
    # Entity types that warrant automatic context injection
    INJECT_TYPES = {"person", "company", "agent", "project", "technology", "tool"}

    def _clean_word(w):
        """Strip trailing punctuation and possessive suffixes ('s, 's)."""
        # Remove all non-alphanumeric except hyphen (for compound words)
        cleaned = re.sub(r"[^a-zA-Z0-9-]", "", w)
        # Strip trailing possessive suffix "s" preceded by nothing (was apostrophe)
        if cleaned.endswith("s") and len(cleaned) > 2:
            # heuristic: if original had 's or 's before 's, strip the trailing s
            if re.search(r"'s?$", w):
                cleaned = cleaned[:-1]
        return cleaned

    words = prompt.split()
    cleaned_words = [_clean_word(w) for w in words]
    candidates = []

    # Bigrams: "Avi Simon", "Fedor Sidorov" etc.
    for i in range(len(cleaned_words) - 1):
        w1, w2 = cleaned_words[i], cleaned_words[i + 1]
        if not w1 or not w2:
            continue
        # At least one word must start uppercase (entities are proper nouns)
        if w1[0].isupper() or w2[0].isupper():
            candidates.append(f"{w1} {w2}")

    # Single capitalized words (4+ chars to avoid "What", "Tell", etc.)
    for w in cleaned_words:
        if len(w) >= 4 and w[0].isupper() and not w.isupper():
            candidates.append(w)

    for token in HEBREW_CANDIDATE_RE.findall(prompt):
        candidates.append(token)

    if not candidates:
        return []

    matched = []
    seen_ids = set()
    try:
        alias_table_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'kg_entity_aliases' LIMIT 1"
            ).fetchall()
        )
        for candidate in candidates:
            rows = conn.execute(
                "SELECT id, name, entity_type FROM kg_entities WHERE LOWER(name) = LOWER(?) LIMIT 1",
                (candidate,),
            ).fetchall()
            if not rows and alias_table_exists:
                rows = conn.execute(
                    """
                    SELECT e.id, e.name, e.entity_type
                    FROM kg_entity_aliases a
                    JOIN kg_entities e ON a.entity_id = e.id
                    WHERE LOWER(a.alias) = LOWER(?)
                    LIMIT 1
                    """,
                    (candidate,),
                ).fetchall()
            if not rows and alias_table_exists and looks_hebrew(candidate):
                query_tokens = phonetic_tokens(candidate)
                if query_tokens:
                    phonetic_rows = conn.execute(
                        """
                        SELECT e.id, e.name, e.entity_type, a.alias
                        FROM kg_entity_aliases a
                        JOIN kg_entities e ON a.entity_id = e.id
                        WHERE a.alias_type = 'phonetic'
                        """
                    ).fetchall()
                    query_key = phonetic_key(candidate)
                    best_row = None
                    best_score = 0.0
                    for row in phonetic_rows:
                        alias_tokens = {token for token in str(row[3]).split() if token}
                        if not alias_tokens:
                            continue
                        overlap = query_tokens & alias_tokens
                        if not overlap:
                            continue
                        score = len(overlap) / len(query_tokens | alias_tokens)
                        if row[3] == query_key:
                            score = 1.0
                        if score > best_score:
                            best_score = score
                            best_row = row
                    rows = [best_row[:3]] if best_row else []
            if rows:
                eid, name, etype = rows[0]
                if eid not in seen_ids and etype in INJECT_TYPES:
                    seen_ids.add(eid)
                    matched.append({"id": eid, "name": name, "entity_type": etype})
    except sqlite3.Error:
        pass

    return matched


def get_entity_chunks(entity_id, conn, limit=3):
    """Get top linked chunk summaries for an entity."""
    try:
        rows = conn.execute(
            """
            SELECT c.content, c.created_at, c.project
            FROM kg_entity_chunks ec
            JOIN chunks c ON c.id = ec.chunk_id
            WHERE ec.entity_id = ?
              AND COALESCE(c.project, '') != 'eval-sandbox'
              AND COALESCE(c.tags, '') NOT LIKE '%"eval-test"%'
            ORDER BY ec.relevance DESC
            LIMIT ?
            """,
            (entity_id, limit),
        ).fetchall()
        return rows
    except sqlite3.Error:
        return []


def _parse_tags(tags):
    if tags is None:
        return []
    if isinstance(tags, list):
        return [str(tag) for tag in tags]
    if isinstance(tags, str):
        try:
            parsed = json.loads(tags)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(tag) for tag in parsed]
        return [part.strip() for part in tags.split(",") if part.strip()]
    return []


def filter_pollution_rows(rows):
    """Exclude eval/test chunks from prompt injection."""
    filtered = []
    for row in rows:
        if row.get("project") == "eval-sandbox":
            continue
        if "eval-test" in _parse_tags(row.get("tags")):
            continue
        filtered.append(row)
    return filtered


def strategic_reorder(rows):
    """Place best result first and second-best last for U-shaped attention."""
    if len(rows) < 3:
        return list(rows)
    ordered = list(rows)
    best = ordered[0]
    second = ordered[1]
    middle = ordered[2:]
    return [best, *middle, second]


def select_adaptive_injection_rows(rows, entity_count=0, light_mode=False):
    """Select 0-5 rows based on RRF score thresholds."""
    del entity_count  # Reserved for future tuning of entity-card budgets.

    filtered = filter_pollution_rows(rows)
    if not filtered:
        return []

    sorted_rows = sorted(filtered, key=lambda row: row.get("rrf_score", 0.0), reverse=True)
    top_score = sorted_rows[0].get("rrf_score", 0.0)

    if top_score < LIGHT_CONFIDENCE_THRESHOLD:
        return []

    if top_score > HIGH_CONFIDENCE_THRESHOLD:
        high_confidence = [row for row in sorted_rows if row.get("rrf_score", 0.0) > HIGH_CONFIDENCE_THRESHOLD]
        selected = high_confidence[:3]
        if len(selected) < 2:
            selected = sorted_rows[: min(2, len(sorted_rows))]
    elif top_score >= MODERATE_CONFIDENCE_THRESHOLD:
        selected = [row for row in sorted_rows if row.get("rrf_score", 0.0) >= MODERATE_CONFIDENCE_THRESHOLD][
            :MAX_ADAPTIVE_INJECTION
        ]
    else:
        selected = [sorted_rows[0]]

    if light_mode:
        selected = selected[:2]

    return strategic_reorder(selected[:MAX_ADAPTIVE_INJECTION])


def _ensure_src_on_syspath():
    src_path = Path(__file__).resolve().parents[1] / "src"
    if src_path.exists():
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


def run_hybrid_search(prompt, db_path, keywords, limit):
    """Run hybrid search and return scored rows suitable for adaptive injection."""
    _ensure_src_on_syspath()

    from brainlayer._helpers import _escape_fts5_query
    from brainlayer.embeddings import get_embedding_model
    from brainlayer.vector_store import VectorStore

    query_text = " ".join(keywords) if keywords else prompt
    model = get_embedding_model()
    query_embedding = model.embed_query(prompt)

    store = VectorStore(Path(db_path))
    try:
        semantic_limit = limit * 3
        if getattr(store, "_binary_index_available", False):
            semantic = store._binary_search(query_embedding=query_embedding, n_results=semantic_limit)
            semantic = store._rerank_binary_results_with_float(query_embedding, semantic)
        else:
            semantic = store.search(query_embedding=query_embedding, n_results=semantic_limit)

        semantic_ranks = {}
        semantic_rows = {}
        for idx, chunk_id in enumerate(semantic["ids"][0]):
            if not chunk_id or chunk_id in semantic_ranks:
                continue
            meta = (semantic["metadatas"][0][idx] or {}).copy()
            semantic_ranks[chunk_id] = idx
            semantic_rows[chunk_id] = {
                "id": chunk_id,
                "content": semantic["documents"][0][idx],
                "importance": meta.get("importance"),
                "project": meta.get("project"),
                "tags": meta.get("tags"),
                "created_at": meta.get("created_at"),
            }

        fts_ranks = {}
        fts_rows = {}
        fts_query = _escape_fts5_query(query_text)
        if fts_query:
            cursor = store._read_cursor()
            fts_results = list(
                cursor.execute(
                    """
                    SELECT f.chunk_id, c.content, c.importance, c.project, c.tags, c.created_at
                    FROM chunks_fts f
                    JOIN chunks c ON c.id = f.chunk_id
                    WHERE chunks_fts MATCH ?
                      AND COALESCE(c.project, '') != 'eval-sandbox'
                      AND COALESCE(c.tags, '') NOT LIKE '%"eval-test"%'
                      AND c.superseded_by IS NULL
                      AND c.aggregated_into IS NULL
                      AND c.archived_at IS NULL
                    ORDER BY f.rank
                    LIMIT ?
                    """,
                    (fts_query, semantic_limit),
                )
            )
            for idx, row in enumerate(fts_results):
                chunk_id, content, importance, project, tags, created_at = row
                if chunk_id in fts_ranks:
                    continue
                fts_ranks[chunk_id] = idx
                fts_rows[chunk_id] = {
                    "id": chunk_id,
                    "content": content,
                    "importance": importance,
                    "project": project,
                    "tags": tags,
                    "created_at": created_at,
                }

        merged = []
        for chunk_id in set(semantic_ranks) | set(fts_ranks):
            score = 0.0
            if chunk_id in semantic_ranks:
                score += 1.0 / (RRF_K + semantic_ranks[chunk_id])
            if chunk_id in fts_ranks:
                score += 1.0 / (RRF_K + fts_ranks[chunk_id])

            row = semantic_rows.get(chunk_id, fts_rows.get(chunk_id))
            if row is None:
                continue
            merged.append(
                {
                    **row,
                    "rrf_score": score,
                }
            )

        return sorted(filter_pollution_rows(merged), key=lambda row: row["rrf_score"], reverse=True)
    finally:
        store.close()


def run_fts_search(db_path, keywords, limit):
    """Current FTS-only fallback path."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1)
    try:
        conn.execute("PRAGMA busy_timeout=1000")
        conn.execute("PRAGMA query_only=true")
        fts_query = " OR ".join(f'"{kw}"' for kw in keywords)
        rows = conn.execute(
            """
            SELECT c.id, c.content, c.importance, c.project, c.tags, c.created_at
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.chunk_id
            WHERE chunks_fts MATCH ?
              AND COALESCE(c.project, '') != 'eval-sandbox'
              AND COALESCE(c.tags, '') NOT LIKE '%"eval-test"%'
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        return [
            {
                "id": chunk_id,
                "content": content,
                "importance": importance,
                "project": project,
                "tags": tags,
                "created_at": created_at,
                "rrf_score": 0.0,
            }
            for chunk_id, content, importance, project, tags, created_at in rows
        ]
    finally:
        conn.close()


def search_prompt_chunks(prompt, db_path, keywords, limit):
    """Search with hybrid first, then fall back to FTS-only behavior."""
    try:
        return run_hybrid_search(prompt, db_path, keywords, limit), True
    except Exception:
        return run_fts_search(db_path, keywords, limit), False


def record_injection_event(db_path, session_id, prompt, chunk_ids, token_count):
    """Best-effort write of an injection event for BrainBar's live viewer."""
    if not db_path or not session_id or not chunk_ids:
        return

    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=2)
        conn.execute("PRAGMA busy_timeout=2000")
        conn.execute(
            """
            INSERT INTO injection_events (session_id, query, chunk_ids, token_count)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, prompt[:1000], json.dumps(chunk_ids), token_count),
        )
        conn.commit()
    except sqlite3.Error:
        pass
    finally:
        if conn is not None:
            conn.close()


def main():
    start = time.monotonic()

    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    activate, light_mode = should_activate()
    if not activate:
        sys.exit(0)

    prompt = hook_input.get("prompt", "")
    if not prompt or len(prompt) < MIN_PROMPT_LENGTH:
        sys.exit(0)

    prompt_lower = prompt.lower()

    # Skip if prompt is a slash command
    if prompt.strip().startswith("/"):
        sys.exit(0)

    # Handoff detection: skip auto-search to avoid duplicate injection
    # (SessionStart already injected handoff context)
    try:
        from dedup_coordination import is_handoff_prompt

        if is_handoff_prompt(prompt):
            print("[BrainLayer: Handoff prompt detected. Skipping automatic search to avoid duplicate injection.]")
            sys.exit(0)
    except ImportError:
        pass

    deep = is_deep_mode(prompt_lower)
    keywords = extract_keywords(prompt)

    if not keywords:
        sys.exit(0)

    db_path = get_db_path()
    if not db_path:
        sys.exit(0)

    # Load already-injected chunk IDs from coordination file
    already_injected = set()
    session_id = hook_input.get("session_id", "")
    try:
        from dedup_coordination import get_injected_ids

        if session_id:
            already_injected = get_injected_ids(session_id)
    except ImportError:
        pass

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1)
        # AIDEV-NOTE: Do NOT set journal_mode=WAL on readonly — it requires a write lock.
        # WAL is already set by the Python writer. Just set query_only for safety.
        conn.execute("PRAGMA busy_timeout=1000")
        conn.execute("PRAGMA query_only=true")
    except sqlite3.Error:
        sys.exit(0)

    fallback_limit = 2 if light_mode else (8 if deep else 3)
    search_limit = MAX_HYBRID_CANDIDATES + len(already_injected) if already_injected else MAX_HYBRID_CANDIDATES

    lines = []
    new_chunk_ids = []
    new_briefs = []
    try:
        # Phase A: Entity routing — detect known entity names in prompt
        # and inject entity profile before FTS results.
        if elapsed_ms(start) < DEADLINE_MS:
            entities = detect_entities_in_prompt(prompt, conn)
            for entity in entities[:2]:  # at most 2 entities per prompt
                etype = entity["entity_type"]
                ename = entity["name"]
                lines.append(f"[Entity: {ename} — {etype}]")
                # Get entity-linked chunks for context
                entity_chunks = get_entity_chunks(entity["id"], conn, limit=2)
                for content, created_at, project in entity_chunks:
                    date = created_at[:10] if created_at else "?"
                    proj = f" ({project})" if project else ""
                    lines.append(f"- [{date}{proj}] {truncate(content, max_chars=150)}")

        if elapsed_ms(start) < DEADLINE_MS:
            rows, used_hybrid = search_prompt_chunks(
                prompt=prompt,
                db_path=db_path,
                keywords=keywords,
                limit=search_limit,
            )

            deduped_rows = []
            for row in rows:
                if row["id"] in already_injected:
                    continue
                deduped_rows.append(row)

            if used_hybrid:
                selected_rows = select_adaptive_injection_rows(
                    deduped_rows,
                    entity_count=len(entities) if "entities" in locals() else 0,
                    light_mode=light_mode,
                )
                mode_label = "adaptive"
            else:
                selected_rows = deduped_rows[:fallback_limit]
                mode_label = "deep" if deep else "auto"

            if selected_rows:
                if lines:
                    lines.append(f"[BrainLayer {mode_label}] Memories matching your prompt:")
                else:
                    lines.append(f"[BrainLayer {mode_label}] Memories matching your prompt:")
                for row in selected_rows:
                    date = row.get("created_at", "")[:10] if row.get("created_at") else "?"
                    importance = row.get("importance")
                    project = row.get("project")
                    imp = f" imp:{importance:.0f}" if importance else ""
                    proj = f" ({project})" if project else ""
                    lines.append(f"- [{date}{imp}{proj}] {truncate(row['content'])}")
                    new_chunk_ids.append(row["id"])
                    new_briefs.append(truncate(row["content"], max_chars=80))

                if used_hybrid and selected_rows[0].get("rrf_score") is not None:
                    lines.append(f"(Adaptive RRF confidence: {selected_rows[0]['rrf_score']:.3f} top score)")
                elif not deep:
                    lines.append("(Use brain_search for deeper results.)")
    except sqlite3.Error:
        pass
    finally:
        conn.close()

    # Inject search-before-assume reminder when prompt contains assumption-prone keywords
    assume_detected = any(re.search(r"\b" + re.escape(trigger) + r"\b", prompt_lower) for trigger in ASSUME_TRIGGERS)
    if assume_detected:
        lines.append(
            "⚠️ SEARCH-BEFORE-ASSUME: This prompt mentions personal/biographical facts. "
            "Run brain_search() to verify before stating any personal details (hardware, history, names)."
        )

    if lines:
        print("\n".join(lines))

    # Register newly injected chunks in coordination file
    if session_id and new_chunk_ids:
        token_estimate = sum(len(b) // 4 for b in new_briefs)
        try:
            from dedup_coordination import register_chunks

            register_chunks(
                session_id=session_id,
                chunk_ids=new_chunk_ids,
                source_hook="UserPromptSubmit",
                briefs=new_briefs,
                token_estimate=token_estimate,
            )
        except Exception:
            pass  # Best-effort coordination

        record_injection_event(
            db_path=db_path,
            session_id=session_id,
            prompt=prompt,
            chunk_ids=new_chunk_ids,
            token_count=token_estimate,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
