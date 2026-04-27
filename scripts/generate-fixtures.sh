#!/usr/bin/env bash
set -euo pipefail

if ! command -v uvx &> /dev/null; then
    echo "ERROR: uvx (from uv package manager) is required but not found in PATH" >&2
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_PATH="${1:-$ROOT_DIR/tests/fixtures/stale_index_query.json}"

mkdir -p "$(dirname "$FIXTURE_PATH")"

cd "$ROOT_DIR"
export FIXTURE_PATH

uv run python3 - <<'PY'
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from brainlayer._helpers import serialize_f32
from brainlayer.embeddings import EMBEDDING_DIM, get_embedding_model
from brainlayer.vector_store import VectorStore

fixture_path = Path(os.environ["FIXTURE_PATH"])
query_match = "apple AND machine"
sample_text = "apple machine retrieval baseline for deterministic fixture verification"

seed_chunks = [
    {
        "id": "orchard-ml-001",
        "content": (
            "apple machine apple machine retrieval baseline for orchard robotics. "
            "apple machine notes focus on ranking stability and deterministic fixtures."
        ),
        "summary": "apple machine retrieval baseline",
        "tags": ["search", "fixture", "apple-machine"],
        "resolved_query": "apple machine retrieval baseline",
        "key_facts": [
            "apple and machine both appear multiple times in the content",
            "document is intentionally concise to rank strongly",
        ],
        "resolved_queries": ["apple machine", "deterministic retrieval baseline"],
    },
    {
        "id": "orchard-ml-002",
        "content": (
            "machine learning notes for apple sorting lines and orchard quality control. "
            "This memory mentions apple once and machine once with less dense overlap."
        ),
        "summary": "machine learning for apple sorting",
        "tags": ["orchard", "ml"],
        "resolved_query": "apple sorting machine learning",
        "key_facts": ["lower keyword density than orchard-ml-001"],
        "resolved_queries": ["apple machine"],
    },
    {
        "id": "orchard-ml-003",
        "content": (
            "machine maintenance checklist for conveyors, sensors, and cooling fans. "
            "An apple crate jam triggered the maintenance runbook once."
        ),
        "summary": "machine maintenance with one apple mention",
        "tags": ["maintenance"],
        "resolved_query": "machine maintenance apple crate",
        "key_facts": ["contains both query terms but weaker topical focus"],
        "resolved_queries": ["machine maintenance"],
    },
    {
        "id": "orchard-ml-004",
        "content": (
            "apple orchard tasting notes and seasonal harvest planning without robotics keywords."
        ),
        "summary": "apple only control document",
        "tags": ["orchard", "taste"],
        "resolved_query": "apple harvest planning",
        "key_facts": ["control row should not satisfy the two-term boolean search"],
        "resolved_queries": ["apple harvest"],
    },
]

with tempfile.TemporaryDirectory(prefix="brainlayer-fixture-") as tmpdir:
    db_path = Path(tmpdir) / "stale-index.db"
    store = VectorStore(db_path)
    try:
        model = get_embedding_model()
        encoder = model._load_model()
        chunk_embeddings = encoder.encode(
            [chunk["content"] for chunk in seed_chunks],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

        cursor = store.conn.cursor()
        for chunk, embedding in zip(seed_chunks, chunk_embeddings):
            cursor.execute(
                """
                INSERT INTO chunks (
                    id, content, metadata, source_file, project, content_type, value_type,
                    char_count, source, summary, tags, resolved_query, key_facts,
                    resolved_queries, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["id"],
                    chunk["content"],
                    "{}",
                    "tests/fixtures/stale_index_query.json",
                    "ecosystem-regression-harness",
                    "assistant_text",
                    "MEDIUM",
                    len(chunk["content"]),
                    "fixture-generator",
                    chunk["summary"],
                    json.dumps(chunk["tags"]),
                    chunk["resolved_query"],
                    json.dumps(chunk["key_facts"]),
                    json.dumps(chunk["resolved_queries"]),
                    "2026-04-27T00:00:00Z",
                ),
            )
            cursor.execute(
                "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                (chunk["id"], serialize_f32([float(value) for value in embedding])),
            )

        fts_sql = (
            "SELECT chunk_id, bm25(chunks_fts) AS rank "
            "FROM chunks_fts "
            f"WHERE chunks_fts MATCH '{query_match}' "
            "ORDER BY bm25(chunks_fts), chunk_id"
        )
        proc = subprocess.run(
            [
                "uvx",
                "--from",
                "sqlite-utils",
                "sqlite-utils",
                "query",
                str(db_path),
                fts_sql,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        ranked_rows = json.loads(proc.stdout)

        baseline_embedding = [float(value) for value in model.embed_query(sample_text)]

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "scripts/generate-fixtures.sh",
            "sqlite_snapshot": {
                "db_backend": "sqlite",
                "fts_table": "chunks_fts",
                "seed_chunk_count": len(seed_chunks),
                "query_sql": fts_sql,
            },
            "embedding_model": {
                "name": model.model_name,
                "dimension": EMBEDDING_DIM,
                "dtype": "float32",
            },
            "query": {
                "match": query_match,
                "expected_ids": [row["chunk_id"] for row in ranked_rows],
                "baseline_rows": ranked_rows,
            },
            "sample_text": {
                "text": sample_text,
                "baseline_embedding": baseline_embedding,
                "min_cosine_similarity": 0.999,
            },
            "chunks": [
                {
                    **chunk,
                    "embedding": [float(value) for value in embedding],
                }
                for chunk, embedding in zip(seed_chunks, chunk_embeddings)
            ],
        }
    finally:
        store.close()

fixture_path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Wrote fixture to {fixture_path}")
PY
