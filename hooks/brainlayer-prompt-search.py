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
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

from brainlayer import paths
from brainlayer.classify import classify_prompt
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
ENTITY_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[\u0590-\u05FF]+")
_ENTITY_CACHE = None
_ENTITY_CACHE_DB_PATH = None


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


HEBREW_WORD_RE = re.compile(r"[\u0590-\u05FF]{2,}")
HEBREW_PREFIXES = set("בלמשהוכ")


def strip_hebrew_prefix(text):
    """Strip common Hebrew prefixes for alias-friendly matching."""
    if not text or len(text) < 3:
        return text
    if text[0] in HEBREW_PREFIXES:
        return text[1:]
    return text


def extract_hebrew_keywords(prompt):
    """Extract Hebrew tokens and prefix-stripped variants for recall."""
    keywords = []
    seen = set()
    for word in HEBREW_WORD_RE.findall(prompt):
        for candidate in (word, strip_hebrew_prefix(word)):
            if len(candidate) >= 2 and candidate not in seen:
                keywords.append(candidate)
                seen.add(candidate)
    return keywords[:8]


def truncate(text, max_chars=200):
    # Clean up multi-line content for compact display
    text = re.sub(r"\n+", " | ", text.strip())
    if len(text) <= max_chars:
        return text
    candidate = text[:max_chars]
    search_start = max(0, max_chars - 40)
    for sep in (". ", "! ", "? ", "| "):
        idx = candidate.rfind(sep, search_start)
        if idx > 0:
            return candidate[: idx + len(sep) - 1] + "..."
    return candidate.rsplit(" ", 1)[0] + "..."


def elapsed_ms(start):
    return (time.monotonic() - start) * 1000


def _get_connection_cache_key(conn):
    if conn is None:
        return str(paths.get_db_path())

    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except sqlite3.Error:
        rows = []

    for row in rows:
        if len(row) >= 3 and row[2]:
            return row[2]

    return f"conn:{id(conn)}"


def _load_entity_cache(conn=None):
    global _ENTITY_CACHE, _ENTITY_CACHE_DB_PATH

    inject_types = ("person", "company", "agent", "project", "technology", "tool")
    cache_key = _get_connection_cache_key(conn)
    if _ENTITY_CACHE is not None and _ENTITY_CACHE_DB_PATH == cache_key:
        return _ENTITY_CACHE

    close_conn = False
    if conn is None:
        conn = sqlite3.connect(paths.get_db_path())
        close_conn = True

    try:
        entities_by_name = {}
        aliases_by_name = {}
        max_name_tokens = 1
        max_alias_tokens = 1

        for entity_id, name, entity_type in conn.execute(
            """
            SELECT id, name, entity_type
            FROM kg_entities
            WHERE entity_type IN (?, ?, ?, ?, ?, ?)
            """,
            inject_types,
        ).fetchall():
            normalized = " ".join(str(name).lower().split())
            if not normalized:
                continue
            entities_by_name.setdefault(normalized, []).append(
                {"id": entity_id, "name": name, "entity_type": entity_type}
            )
            max_name_tokens = max(max_name_tokens, len(normalized.split()))

        alias_table_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'kg_entity_aliases' LIMIT 1"
            ).fetchall()
        )
        if alias_table_exists:
            for alias, entity_id, name, entity_type in conn.execute(
                """
                SELECT a.alias, e.id, e.name, e.entity_type
                FROM kg_entity_aliases a
                JOIN kg_entities e ON a.entity_id = e.id
                WHERE e.entity_type IN (?, ?, ?, ?, ?, ?)
                """,
                inject_types,
            ).fetchall():
                normalized = " ".join(str(alias).lower().split())
                if not normalized:
                    continue
                aliases_by_name.setdefault(normalized, []).append(
                    {"id": entity_id, "name": name, "entity_type": entity_type}
                )
                max_alias_tokens = max(max_alias_tokens, len(normalized.split()))

        _ENTITY_CACHE = {
            "entities_by_name": entities_by_name,
            "aliases_by_name": aliases_by_name,
            "max_name_tokens": max_name_tokens,
            "max_alias_tokens": max_alias_tokens,
        }
        _ENTITY_CACHE_DB_PATH = cache_key
    except sqlite3.Error:
        _ENTITY_CACHE = {
            "entities_by_name": {},
            "aliases_by_name": {},
            "max_name_tokens": 1,
            "max_alias_tokens": 1,
        }
    finally:
        if close_conn:
            conn.close()

    return _ENTITY_CACHE


def _iter_prompt_tokens(prompt):
    return [(match.group(0), match.group(0).lower()) for match in ENTITY_TOKEN_RE.finditer(prompt)]


def _match_entity_spans(tokens, candidates_by_name, max_tokens, inject_types, seen_ids):
    matched = []
    index = 0

    while index < len(tokens):
        found = None
        max_span = min(max_tokens, len(tokens) - index)
        for span in range(max_span, 0, -1):
            normalized = " ".join(token[1] for token in tokens[index : index + span])
            for candidate in candidates_by_name.get(normalized, []):
                if candidate["entity_type"] not in inject_types or candidate["id"] in seen_ids:
                    continue
                found = candidate
                break
            if found:
                index += span
                break

        if found:
            seen_ids.add(found["id"])
            matched.append(found)
            continue

        index += 1

    return matched


def detect_entities_in_prompt(prompt, conn=None):
    """Detect known KG entity names in the prompt.

    Loads KG entity names into a module-level cache and performs
    case-insensitive hash-table lookups over contiguous prompt token spans.
    Multi-word names are matched longest-first.
    """
    inject_types = {"person", "company", "agent", "project", "technology", "tool"}
    tokens = _iter_prompt_tokens(prompt)
    if not tokens:
        return []

    cache = _load_entity_cache(conn)
    seen_ids = set()

    matched = _match_entity_spans(
        tokens,
        cache["entities_by_name"],
        cache["max_name_tokens"],
        inject_types,
        seen_ids,
    )
    alias_matches = _match_entity_spans(
        tokens,
        cache["aliases_by_name"],
        cache["max_alias_tokens"],
        inject_types,
        seen_ids,
    )
    matched.extend(alias_matches)

    if conn is None:
        return matched

    try:
        phonetic_rows = conn.execute(
            """
            SELECT e.id, e.name, e.entity_type, a.alias
            FROM kg_entity_aliases a
            JOIN kg_entities e ON a.entity_id = e.id
            WHERE a.alias_type = 'phonetic'
            """
        ).fetchall()
    except sqlite3.Error:
        return matched

    for candidate, _ in tokens:
        if not looks_hebrew(candidate):
            continue
        query_tokens = phonetic_tokens(candidate)
        if not query_tokens:
            continue
        query_key = phonetic_key(candidate)
        best_row = None
        best_score = 0.0
        for row in phonetic_rows:
            entity_id, name, entity_type, alias = row
            if entity_type not in inject_types or entity_id in seen_ids:
                continue
            alias_tokens = {token for token in str(alias).split() if token}
            if not alias_tokens:
                continue
            overlap = query_tokens & alias_tokens
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens | alias_tokens)
            if alias == query_key:
                score = 1.0
            if score > best_score:
                best_score = score
                best_row = row
        if best_row:
            entity_id, name, entity_type, _ = best_row
            seen_ids.add(entity_id)
            matched.append({"id": entity_id, "name": name, "entity_type": entity_type})

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


def _ensure_injection_event_schema(conn):
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(injection_events)").fetchall()}
    for column_name, definition in (
        ("latency_ms", "INTEGER NOT NULL DEFAULT 0"),
        ("mode", "TEXT NOT NULL DEFAULT 'normal'"),
        ("entities_detected", "INTEGER NOT NULL DEFAULT 0"),
    ):
        if column_name in existing_columns:
            continue
        try:
            conn.execute(f"ALTER TABLE injection_events ADD COLUMN {column_name} {definition}")
        except sqlite3.Error:
            pass


def record_injection_event(
    db_path,
    session_id,
    prompt,
    chunk_ids,
    token_count,
    *,
    latency_ms=0,
    mode="normal",
    entities_detected=0,
):
    """Best-effort write of an injection event for BrainBar's live viewer."""
    if not db_path or not session_id:
        return

    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=2)
        conn.execute("PRAGMA busy_timeout=2000")
        _ensure_injection_event_schema(conn)
        conn.execute(
            """
            INSERT INTO injection_events (
                session_id, query, chunk_ids, token_count, latency_ms, mode, entities_detected
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                prompt[:1000],
                json.dumps(chunk_ids or []),
                token_count,
                int(latency_ms),
                mode,
                int(entities_detected),
            ),
        )
        if hasattr(conn, "commit"):
            conn.commit()
    except (sqlite3.Error, AttributeError):
        pass
    finally:
        if conn is not None:
            conn.close()


def record_prompt_classification(session_id, prompt, classification):
    """Best-effort JSONL logging for route analysis without DB writes."""
    if not prompt:
        return

    log_path = os.environ.get(
        "BRAINLAYER_PROMPT_CLASSIFICATION_LOG",
        os.path.expanduser("~/.local/share/brainlayer/prompt_classification_events.jsonl"),
    )
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "session_id": session_id,
                        "prompt_hash": sha256(prompt.encode("utf-8")).hexdigest(),
                        "classification": classification,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                + "\n"
            )
    except OSError:
        pass


def build_fts_query(keywords):
    return " OR ".join(f'"{kw}"' for kw in keywords)


def fetch_matching_chunks(conn, fts_query, limit, date_from=None):
    extra_sql = ""
    params = [fts_query]
    if date_from:
        extra_sql = "AND c.created_at >= ?"
        params.append(date_from)
    params.append(limit)

    return conn.execute(
        f"""
        SELECT c.id, c.content, c.importance, c.project, c.tags, c.created_at
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.chunk_id
        WHERE chunks_fts MATCH ?
        {extra_sql}
        ORDER BY rank
        LIMIT ?
        """,
        params,
    ).fetchall()


def inject_entity_context(lines, prompt, conn):
    entities = detect_entities_in_prompt(prompt, conn)
    for entity in entities[:2]:
        etype = entity["entity_type"]
        ename = entity["name"]
        lines.append(f"[Entity: {ename} — {etype}]")
        entity_chunks = get_entity_chunks(entity["id"], conn, limit=2)
        for content, created_at, project in entity_chunks:
            date = created_at[:10] if created_at else "?"
            proj = f" ({project})" if project else ""
            lines.append(f"- [{date}{proj}] {truncate(content, max_chars=150)}")
    return entities


def inject_search_results(lines, rows, deep, label="auto"):
    chunk_ids = []
    briefs = []
    if not rows:
        return chunk_ids, briefs

    mode_label = "deep" if deep else label
    lines.append(f"[BrainLayer {mode_label}] Memories matching your prompt:")
    for chunk_id, content, importance, project, tags, created_at in rows:
        date = created_at[:10] if created_at else "?"
        imp = f" imp:{importance:.0f}" if importance else ""
        proj = f" ({project})" if project else ""
        lines.append(f"- [{date}{imp}{proj}] {truncate(content)}")
        chunk_ids.append(chunk_id)
        briefs.append(truncate(content, max_chars=80))

    if not deep and label in {"auto", "follow_up", "hebrew"}:
        lines.append("(Use brain_search for deeper results.)")

    return chunk_ids, briefs


def main():
    start = time.monotonic()
    db_path = None
    prompt = ""
    session_id = ""
    telemetry_mode = "skip"
    new_chunk_ids = []
    new_briefs = []
    entities_detected = 0

    def finalize_and_exit(*, mode=None):
        final_mode = mode or telemetry_mode
        if session_id and prompt and db_path:
            token_estimate = sum(len(b) // 4 for b in new_briefs)
            record_injection_event(
                db_path=db_path,
                session_id=session_id,
                prompt=prompt,
                chunk_ids=new_chunk_ids,
                token_count=token_estimate,
                latency_ms=elapsed_ms(start),
                mode=final_mode,
                entities_detected=entities_detected,
            )
        sys.exit(0)

    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    prompt = hook_input.get("prompt", "")
    session_id = hook_input.get("session_id", "")
    db_path = get_db_path()

    activate, light_mode = should_activate()
    if not activate:
        finalize_and_exit(mode="skip")

    if not prompt:
        finalize_and_exit(mode="skip")

    classification = classify_prompt(prompt)
    record_prompt_classification(session_id=session_id, prompt=prompt, classification=classification)
    if classification in {"command", "casual_chat"}:
        finalize_and_exit(mode="skip")

    prompt_lower = prompt.lower()

    # Handoff detection: skip auto-search to avoid duplicate injection
    # (SessionStart already injected handoff context)
    try:
        from dedup_coordination import is_handoff_prompt

        if is_handoff_prompt(prompt):
            print("[BrainLayer: Handoff prompt detected. Skipping automatic search to avoid duplicate injection.]")
            finalize_and_exit(mode="skip")
    except ImportError:
        pass

    if not db_path:
        finalize_and_exit(mode="skip")

    # Load already-injected chunk IDs from coordination file
    already_injected = set()
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
        finalize_and_exit(mode="skip")

    detected_entities = []
    if classification == "knowledge_question":
        detected_entities = detect_entities_in_prompt(prompt, conn)
        entities_detected = len(detected_entities)
        classification = classify_prompt(prompt, detected_entities=detected_entities)
        record_prompt_classification(session_id=session_id, prompt=prompt, classification=classification)

    deep = is_deep_mode(prompt_lower)
    if classification == "entity_lookup":
        telemetry_mode = "entity"
    elif deep:
        telemetry_mode = "deep"
    else:
        telemetry_mode = "normal"

    if classification == "hebrew_query":
        keywords = extract_hebrew_keywords(prompt) or extract_keywords(prompt)
    else:
        keywords = extract_keywords(prompt)

    if not keywords and classification != "entity_lookup":
        conn.close()
        finalize_and_exit(mode="skip")

    # Over-fetch to compensate for dedup removals
    # Light mode: cap at 2 results to reduce token cost for workers
    if light_mode:
        base_limit = 2
    else:
        base_limit = 8 if deep else 3
    limit = base_limit + len(already_injected) if already_injected else base_limit

    fts_query = build_fts_query(keywords)

    lines = []
    try:
        if classification == "entity_lookup" and elapsed_ms(start) < DEADLINE_MS:
            detected_entities = inject_entity_context(lines, prompt, conn)
            entities_detected = len(detected_entities)
        elif classification in {"knowledge_question", "follow_up", "hebrew_query"} and elapsed_ms(start) < DEADLINE_MS:
            if detected_entities and classification == "knowledge_question":
                inject_entity_context(lines, prompt, conn)

            date_from = None
            label = "auto"
            if classification == "follow_up":
                date_from = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
                label = "follow_up"
            elif classification == "hebrew_query":
                label = "hebrew"

            rows = fetch_matching_chunks(conn, fts_query, limit, date_from=date_from)
            filtered_rows = []
            for row in rows:
                chunk_id = row[0]
                if chunk_id not in already_injected:
                    filtered_rows.append(row)
                if len(filtered_rows) >= base_limit:
                    break

            if classification == "follow_up" and not filtered_rows:
                rows = fetch_matching_chunks(conn, fts_query, limit)
                for row in rows:
                    chunk_id = row[0]
                    if chunk_id not in already_injected:
                        filtered_rows.append(row)
                    if len(filtered_rows) >= base_limit:
                        break

            new_chunk_ids, new_briefs = inject_search_results(lines, filtered_rows, deep, label=label)
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
        try:
            from dedup_coordination import register_chunks

            register_chunks(
                session_id=session_id,
                chunk_ids=new_chunk_ids,
                source_hook="UserPromptSubmit",
                briefs=new_briefs,
                token_estimate=sum(len(b) // 4 for b in new_briefs),
            )
        except Exception:
            pass  # Best-effort coordination

    finalize_and_exit()


if __name__ == "__main__":
    main()
