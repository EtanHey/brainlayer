#!/usr/bin/env python3
"""Offline literal-recall diagnosis. No models, BrainLayer imports, or write DBs.

snapshot freezes audit identities; export performs bounded explicit readonly reads;
analyze ranks only supplied documents in an in-memory FTS index (run on M1).
Private inputs/outputs belong in .build.noindex, never public fixtures.
"""

import argparse
import ast
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ("content", "summary", "tags", "resolved_query", "key_facts", "resolved_queries")
FROZEN = (
    "questions.json",
    "groundtruth-freeze.json",
    "corpus-manifest.json",
    "brain-benchmark.json",
    "brain-source-scoring.json",
    "brain-guided.json",
    "brain-alias-verification.json",
    "structural-linked-benchmark.json",
    "REPORT.md",
)
STOP = frozenset(
    "what did want to become why the which a in where come from and were how could run or through without for require".split()
)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def python_builder(repo):
    path = repo / "src/brainlayer/_helpers.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_escape_fts5_query")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def swift_mirror(query):
    """Mirror BrainDatabase.sanitizeFTS5Query; not execution of the Swift binary."""
    terms = [w.replace('"', "").replace("*", "").strip() for w in query.split(" ")]
    terms = [f'"{w}"*' for w in terms if w]
    return (" OR " if len(terms) >= 4 else " ").join(terms) or '""'


def text(result):
    return "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")


def ids(result):
    return re.findall(r"^- ID: (.*?)\s*(?:\||$)", text(result), re.M)


def snapshot(audit):
    questions = read(audit / "questions.json")
    assert questions == read(audit / "groundtruth-freeze.json")["questions"]
    scores = read(audit / "brain-source-scoring.json")["scores"]
    runs = read(audit / "brain-benchmark.json")
    grep = [r for r in read(audit / "structural-linked-benchmark.json") if r["arm"] == "rg"]
    rows = []
    for q in questions:
        matching = [r for r in runs if r["id"] == q["id"]]
        score = next(s for s in scores if s["id"] == q["id"] and s["arm"] == "literal")
        hits = sorted(set(score["units"]) & set(q["relevant"]))
        assert hits == sorted(score["hits"])
        assert len(hits) / len(q["relevant"]) == score["recall_at_5"]
        assert len(matching) == 5 and all(r["question"] == q["question"] for r in matching)
        assert all(not r.get("error") for r in matching)
        rows.append(
            dict(
                id=q["id"],
                gold=q["relevant"],
                hits=hits,
                recall=len(hits) / len(q["relevant"]),
                grep_recall=next(r["recall_at_5"] for r in grep if r["id"] == q["id"]),
                top5=ids(matching[0]["result"]),
                stable_ids=all(ids(r["result"]) == ids(matching[0]["result"]) for r in matching),
            )
        )
    manifest = read(audit / "corpus-manifest.json")
    sources = []
    for unit in manifest:
        copy = audit / "corpus" / unit["corpus_file"]
        assert digest(copy) == unit["sha256"], f"Corpus drift: {unit['id']}"
        original = Path(unit["source"])
        sources.append(
            dict(
                id=unit["id"],
                original=str(original),
                exists=original.is_file(),
                original_sha256=digest(original) if original.is_file() else None,
                corpus_sha256=digest(copy),
            )
        )
    return dict(
        inputs={n: digest(audit / n) for n in FROZEN},
        sources=sources,
        questions=rows,
        macro_recall=sum(r["recall"] for r in rows) / len(rows),
        grep_macro_recall=sum(r["grep_recall"] for r in rows) / len(rows),
        note="Recomputes historical adjudication, not independent relevance scoring.",
    )


def readonly(path):
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2)
    conn.execute("PRAGMA query_only=ON")
    return conn


def export(audit, database, extra_ids=(), needles=()):
    """At most 80 primary-key rows + 12 rows per rare indexed FTS probe; no scans."""
    frozen = snapshot(audit)
    wanted = list(dict.fromkeys(cid for q in frozen["questions"] for cid in q["top5"]))
    wanted += list(extra_ids)
    assert len(wanted) <= 80
    conn = readonly(database)
    conn.row_factory = sqlite3.Row
    rows, probes = {}, []
    columns = (
        "id,"
        + ",".join(FIELDS)
        + ",source_file,source,project,created_at,source_class,content_class,superseded_by,aggregated_into,archived_at"
    )
    for cid in wanted:
        row = conn.execute(f"SELECT {columns} FROM chunks WHERE id=?", (cid,)).fetchone()
        if row:
            rows[cid] = dict(row)
    assert len(needles) <= 4
    for needle in needles:
        started = time.monotonic()
        conn.set_progress_handler(lambda: int(time.monotonic() - started > 0.5), 1000)
        try:
            found = conn.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 12", (needle,)
            ).fetchall()
            probe = dict(query=needle, ids=[r[0] for r in found], cap=12, error=None)
        except sqlite3.Error as exc:
            probe = dict(query=needle, ids=[], cap=12, error=str(exc))
        finally:
            conn.set_progress_handler(None, 0)
        probes.append(probe)
        for cid in probe["ids"]:
            row = conn.execute(f"SELECT {columns} FROM chunks WHERE id=?", (cid,)).fetchone()
            if row:
                rows[cid] = dict(row)
    conn.close()
    return dict(
        database=str(database),
        access="mode=ro; query_only=ON; bounded exact IDs/FTS",
        probes=probes,
        rows=list(rows.values()),
        missing_ids=[cid for cid in wanted if cid not in rows],
        note="Current bounded sample, not a complete coverage census or historical DB snapshot.",
    )


def fts_index(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE VIRTUAL TABLE f USING fts5("
        + ",".join(FIELDS)
        + ",id UNINDEXED, tokenize='unicode61 remove_diacritics 2', prefix='2 3 4')"
    )
    for row in rows:
        conn.execute("INSERT INTO f VALUES (?,?,?,?,?,?,?)", [row.get(k) or "" for k in FIELDS] + [row["id"]])
    return conn


def ranked(db, query):
    return [r[0] for r in db.execute("SELECT id FROM f WHERE f MATCH ? ORDER BY rank, id", (query,))]


def membership(repo, sample, chunk_ids, query):
    """Tiny single-row membership fixtures, not a ranking experiment."""
    assert 1 <= len(chunk_ids) <= 4
    escape = python_builder(repo)
    by_id = {r["id"]: r for r in read(sample)["rows"]}
    result = []
    for cid in chunk_ids:
        db = fts_index([by_id[cid]])

        def matches(fts):
            return bool(db.execute("SELECT id FROM f WHERE f MATCH ?", (fts,)).fetchall())

        fields = {w: {k: matches(k + " : " + escape(w)) for k in FIELDS} for w in query.split()}
        result.append(
            dict(
                id=cid,
                python_candidate=matches(escape(query)),
                swift_candidate=matches(swift_mirror(query)),
                fields_matching_terms=fields,
            )
        )
        db.close()
    return dict(
        rows=result,
        query=query,
        sample_sha256=digest(sample),
        method="Independent single-row FTS membership fixtures; no ranking or model",
    )


def analyze(audit, repo, sample):
    escape = python_builder(repo)
    corpus = [
        dict(id=u["id"], content=(audit / "corpus" / u["corpus_file"]).read_text())
        for u in read(audit / "corpus-manifest.json")
    ]
    cohorts = {"curated_documents": corpus}
    if sample:
        cohorts["exported_chunk_sample"] = read(sample)["rows"]
    output = {}
    for name, rows in cohorts.items():
        db = fts_index(rows)
        results = []
        for q in read(audit / "questions.json"):
            query = q["question"]
            terms = [w.replace('"', "") for w in query.split()]
            variants = {
                "python_actual": escape(query),
                "swift_mirror": swift_mirror(query),
                "forced_and_counterfactual": " AND ".join(f'"{w}"' for w in terms),
                "remove_function_words": escape(" ".join(w for w in query.split() if w.lower() not in STOP)),
                "graphify_anchor_counterfactual": f'"graphify" AND ({escape(query)})',
            }
            arms = {}
            for arm, fts in variants.items():
                try:
                    candidates = ranked(db, fts)
                    arms[arm] = dict(fts=fts, candidates=candidates, top5=candidates[:5], error=None)
                except sqlite3.Error as exc:
                    arms[arm] = dict(fts=fts, candidates=[], top5=[], error=str(exc))
            results.append(dict(id=q["id"], arms=arms))
        output[name] = dict(rows=len(rows), queries=results)
        db.close()
    return dict(
        cohorts=output,
        helper_sha256=digest(repo / "src/brainlayer/_helpers.py"),
        swift_sha256=digest(repo / "brain-bar/Sources/BrainBar/BrainDatabase.swift"),
        sample_sha256=digest(sample) if sample else None,
        caveat="Lexical mechanism ablations only. Different corpus statistics, chunking and no vectors/production reranker; no uplift estimate.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["snapshot", "export", "membership", "analyze"])
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--database", type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--chunk-id", action="append", default=[])
    parser.add_argument("--needle", action="append", default=[])
    parser.add_argument("--query")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "export" and args.database is None:
        parser.error("export requires an explicit --database (always opened readonly)")
    if args.mode == "membership" and (not args.sample or not args.query or not args.chunk_id):
        parser.error("membership requires --sample, --query and 1-4 --chunk-id arguments")
    if args.mode == "analyze":
        result = analyze(args.audit, args.repo, args.sample)
    elif args.mode == "membership":
        result = membership(args.repo, args.sample, args.chunk_id, args.query)
    elif args.mode == "export":
        result = export(args.audit, args.database, args.chunk_id, args.needle)
    else:
        result = snapshot(args.audit)
    result["captured_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("output exists; choose a new receipt path to preserve prior evidence")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(mode=args.mode, output=str(args.output), sha256=digest(args.output))))


if __name__ == "__main__":
    main()
