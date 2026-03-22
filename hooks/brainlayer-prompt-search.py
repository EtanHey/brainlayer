#!/usr/bin/env python3
"""
BrainLayer UserPromptSubmit Hook — auto-searches memories relevant to the user's prompt.

Extracts keywords from the prompt, runs FTS5 search against BrainLayer.
Two modes:
  - Light (default): top 3 results, ~300 tokens
  - Deep (triggered by memory words): top 8 results, ~800 tokens

Output: plain text to stdout (injected as Claude context).
Target: <500ms total.
"""

import json
import os
import re
import sqlite3
import sys
import time

DEADLINE_MS = 450

# Prompts shorter than this are probably greetings/commands — skip search
MIN_PROMPT_LENGTH = 15

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
    "code",
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
        if w not in STOP_WORDS and len(w) > 2 and w not in seen:
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
    INJECT_TYPES = {"person", "company", "agent"}

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

    if not candidates:
        return []

    matched = []
    seen_ids = set()
    try:
        for candidate in candidates:
            rows = conn.execute(
                "SELECT id, name, entity_type FROM kg_entities WHERE LOWER(name) = LOWER(?) LIMIT 1",
                (candidate,),
            ).fetchall()
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
            ORDER BY ec.relevance DESC
            LIMIT ?
            """,
            (entity_id, limit),
        ).fetchall()
        return rows
    except sqlite3.Error:
        return []


def main():
    start = time.monotonic()

    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    prompt = hook_input.get("prompt", "")
    if not prompt or len(prompt) < MIN_PROMPT_LENGTH:
        sys.exit(0)

    prompt_lower = prompt.lower()

    # Skip if prompt is a slash command
    if prompt.strip().startswith("/"):
        sys.exit(0)

    deep = is_deep_mode(prompt_lower)
    keywords = extract_keywords(prompt)

    if not keywords:
        sys.exit(0)

    db_path = get_db_path()
    if not db_path:
        sys.exit(0)

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA query_only=true")
    except sqlite3.Error:
        sys.exit(0)

    limit = 8 if deep else 3

    # Build FTS5 query: join keywords with OR for broader matching
    fts_query = " OR ".join(f'"{kw}"' for kw in keywords)

    lines = []
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
            rows = conn.execute(
                """
                SELECT c.content, c.importance, c.project, c.tags, c.created_at
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()

            if rows:
                mode_label = "deep" if deep else "auto"
                if lines:
                    # Entity section already started — add separator
                    lines.append(f"[BrainLayer {mode_label}] Memories matching your prompt:")
                else:
                    lines.append(f"[BrainLayer {mode_label}] Memories matching your prompt:")
                for content, importance, project, tags, created_at in rows:
                    date = created_at[:10] if created_at else "?"
                    imp = f" imp:{importance:.0f}" if importance else ""
                    proj = f" ({project})" if project else ""
                    lines.append(f"- [{date}{imp}{proj}] {truncate(content)}")

                if not deep:
                    lines.append("(Use brain_search for deeper results.)")
    except sqlite3.Error:
        pass
    finally:
        conn.close()

    if lines:
        print("\n".join(lines))

    sys.exit(0)


if __name__ == "__main__":
    main()
