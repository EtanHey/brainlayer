#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from brainlayer.paths import get_db_path

DB_PATH = get_db_path()
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}")
PROFANITY_RE = re.compile(r"\b(fuck(?:ing|er|ers)?|motherfuck(?:er|ers)?|wtf)\b", re.IGNORECASE)
STOPWORDS = {
    "about", "after", "again", "and", "app", "are", "but", "can", "dont", "for", "from", "get", "got", "have",
    "just", "lets", "like", "need", "not", "now", "our", "out", "really", "still", "that", "the", "their", "them",
    "then", "this", "tonight", "very", "want", "were", "what", "when", "with", "you", "your",
}

def log(message):
    print(f"[cleanup-profanity] {message}", file=sys.stderr)

def compact(text, limit=96, sanitize=False):
    text = re.sub(r"\s+", " ", text or "").strip()
    if sanitize:
        text = PROFANITY_RE.sub("[redacted]", text)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."

def parse_tags(raw):
    try:
        value = json.loads(raw or "[]")
        return [tag for tag in value if isinstance(tag, str)]
    except json.JSONDecodeError:
        return []

def tokenize(text, include_profanity=False):
    terms = []
    for token in TOKEN_RE.findall((text or "").lower()):
        token = token.strip("'")
        if len(token) < 3 or token in STOPWORDS:
            continue
        if not include_profanity and PROFANITY_RE.fullmatch(token):
            continue
        terms.append(token)
    return terms

def fetch_rows(conn):
    return conn.execute(
        """
        SELECT id, content, metadata, source_file, project, content_type, tags, importance, created_at, conversation_id
        FROM chunks
        WHERE id LIKE 'rt-%'
          AND content_type = 'user_message'
          AND archived = 0
          AND aggregated_into IS NULL
          AND archived_at IS NULL
          AND lower(content) GLOB '*fuck*'
        ORDER BY created_at, id
        """
    ).fetchall()

def parse_metadata(raw):
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}

def session_id_for_row(row):
    return row["conversation_id"] or parse_metadata(row["metadata"]).get("session_id") or "unknown"

def cluster_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        day = (row["created_at"] or "")[:10]
        session_id = session_id_for_row(row)
        grouped[(row["project"] or "unknown", day, session_id)].append(row)
    return {key: group for key, group in grouped.items() if len(group) >= 3}

def build_doc_freq(rows):
    df = Counter()
    for row in rows:
        for token in set(tokenize(row["content"])):
            df[token] += 1
    return len(rows), df

def top_terms(cluster, total_docs, df):
    tf = Counter()
    for row in cluster:
        tf.update(tokenize(row["content"]))
        tf.update(tokenize(row["project"] or ""))
    ranked = sorted(
        ((term, count * (math.log((1 + total_docs) / (1 + df.get(term, 0))) + 1.0), count) for term, count in tf.items()),
        key=lambda item: (-item[1], -item[2], item[0]),
    )
    return [term for term, _, _ in ranked[:3]] or ["frustration", "repeat", "user"]

def build_aggregate(cluster, total_docs, df):
    cluster = sorted(cluster, key=lambda row: (row["created_at"] or "", row["id"]))
    project, day, session_id = cluster[0]["project"] or "unknown", (cluster[0]["created_at"] or "")[:10], session_id_for_row(cluster[0])
    keywords = top_terms(cluster, total_docs, df)
    keyword_text = ", ".join(keywords)
    first_raw = compact(cluster[0]["content"])
    last_raw = compact(cluster[-1]["content"])
    anchor_row = next((row for row in cluster if re.search(r"\b(ship|shipped|shipping|build|deploy|deployed|merge|merged)\b", row["content"] or "", re.IGNORECASE)), cluster[len(cluster) // 2])
    anchor_raw = compact(anchor_row["content"])
    content = (
        f"[USER FRUSTRATION {len(cluster)}x on {day}] Top themes: {keyword_text}. "
        f"First: {compact(cluster[0]['content'], sanitize=True)}. Last: {compact(cluster[-1]['content'], sanitize=True)}."
    )
    raw_terms = Counter(token for row in cluster for token in tokenize(row["content"], include_profanity=True))
    focus = [token for token in tokenize(anchor_row["content"], include_profanity=True) if token in {"ship", "shipped", "shipping", "build", "deploy", "deployed", "merge", "merged"}]
    raw_search = " ".join(dict.fromkeys(["fucking", *re.findall(r"[a-zA-Z]+", project.lower()), *focus, *[term for term, _ in raw_terms.most_common(8)]]))
    tags = sorted({tag for row in cluster for tag in parse_tags(row["tags"])} | {"user-frustration-aggregated"})
    agg_id = "agg-frustration-" + hashlib.sha1(f"{project}|{day}|{session_id}".encode()).hexdigest()[:16]
    metadata = json.dumps(
        {"aggregation": "profanity-cleanup-v1", "session_id": session_id, "aggregated_from": [row["id"] for row in cluster], "count": len(cluster)},
        ensure_ascii=False,
    )
    return {
        "id": agg_id,
        "content": content,
        "metadata": metadata,
        "source_file": cluster[0]["source_file"],
        "project": project,
        "content_type": "user_message",
        "char_count": len(content),
        "tags": json.dumps(tags, ensure_ascii=False),
        "summary": content[:200],
        "importance": max((row["importance"] or 0) for row in cluster),
        "created_at": cluster[0]["created_at"],
        "conversation_id": session_id,
        "resolved_query": raw_search,
        "resolved_queries": json.dumps([raw_search, first_raw, anchor_raw, last_raw], ensure_ascii=False),
        "content_hash": hashlib.sha256(f"{agg_id}|{content}".encode()).hexdigest(),
    }

def archive_originals(conn, cluster, agg_id, now_iso):
    for row in cluster:
        conn.execute(
            """
            UPDATE chunks
            SET aggregated_into = ?, archived = 1, archived_at = ?, value_type = 'ARCHIVED'
            WHERE id = ?
            """,
            (agg_id, now_iso, row["id"]),
        )

def run_cleanup(db_path, dry_run=False):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = fetch_rows(conn)
    clusters = cluster_rows(rows)
    stats = {"clusters": len(clusters), "aggregated": sum(len(group) for group in clusters.values()), "archived": sum(len(group) for group in clusters.values())}
    log(f"clusters={stats['clusters']} aggregated={stats['aggregated']} archived={stats['archived']} dry_run={str(dry_run).lower()} rows={len(rows)}")
    if dry_run or not clusters:
        conn.close()
        return stats
    total_docs, df = build_doc_freq(rows)
    now_iso = datetime.now(timezone.utc).isoformat()
    with conn:
        for cluster in clusters.values():
            aggregate = build_aggregate(cluster, total_docs, df)
            conn.execute(
                """
                INSERT INTO chunks (
                    id, content, metadata, source_file, project, content_type, char_count, tags,
                    summary, importance, created_at, conversation_id, resolved_query, resolved_queries, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    aggregate["id"], aggregate["content"], aggregate["metadata"], aggregate["source_file"], aggregate["project"],
                    aggregate["content_type"], aggregate["char_count"], aggregate["tags"], aggregate["summary"], aggregate["importance"],
                    aggregate["created_at"], aggregate["conversation_id"], aggregate["resolved_query"], aggregate["resolved_queries"],
                    aggregate["content_hash"],
                ),
            )
            archive_originals(conn, cluster, aggregate["id"], now_iso)
    conn.close()
    return stats

def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate repeated profane raw-user chunks into sanitized frustration summaries.")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite DB path")
    parser.add_argument("--dry-run", action="store_true", help="Report clusters without writing")
    args = parser.parse_args(argv)
    run_cleanup(args.db, dry_run=args.dry_run)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
