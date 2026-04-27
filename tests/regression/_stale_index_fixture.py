"""Helpers for the stale index regression fixture."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from brainlayer.embeddings import get_embedding_model

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "stale_index_query.json"


def load_fixture() -> dict[str, Any]:
    """Load the seeded stale-index regression fixture."""
    return json.loads(FIXTURE_PATH.read_text())


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity without adding another numeric dependency."""
    if len(left) != len(right):
        raise ValueError(f"embedding length mismatch: {len(left)} != {len(right)}")

    dot_product = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot_product += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value

    return dot_product / ((left_norm**0.5) * (right_norm**0.5))


def create_fixture_db(db_path: Path) -> None:
    """Seed a temporary SQLite FTS table from the fixture chunks."""
    fixture = load_fixture()
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
              content,
              summary,
              tags,
              resolved_query,
              key_facts,
              resolved_queries,
              chunk_id UNINDEXED
            );
            """
        )
        insert_sql = """
            INSERT INTO chunks_fts(
              content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        for chunk in fixture["chunks"]:
            connection.execute(
                insert_sql,
                (
                    chunk["content"],
                    chunk.get("summary"),
                    json.dumps(chunk["tags"]) if chunk.get("tags") else None,
                    chunk.get("resolved_query"),
                    json.dumps(chunk["key_facts"]) if chunk.get("key_facts") else None,
                    json.dumps(chunk["resolved_queries"]) if chunk.get("resolved_queries") else None,
                    chunk["id"],
                ),
            )
        connection.commit()
    finally:
        connection.close()


def current_embedding_rows() -> list[list[float]]:
    """Re-embed the fixture corpus with the current model."""
    fixture = load_fixture()
    model = get_embedding_model()
    encoder = model._load_model()
    chunk_embeddings = encoder.encode(
        [chunk["content"] for chunk in fixture["chunks"]],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()
    sample_embedding = model.embed_query(fixture["sample_text"]["text"])
    return [[float(value) for value in row] for row in chunk_embeddings] + [sample_embedding]


def baseline_embedding_rows() -> list[list[float]]:
    """Return the baseline embedding matrix stored in the fixture."""
    fixture = load_fixture()
    chunk_embeddings = [[float(value) for value in chunk["embedding"]] for chunk in fixture["chunks"]]
    sample_embedding = [float(value) for value in fixture["sample_text"]["baseline_embedding"]]
    return chunk_embeddings + [sample_embedding]


def write_expected_ranking_json(output_path: Path) -> None:
    """Write the baseline FTS rows to a normalized JSON file."""
    fixture = load_fixture()
    output_path.write_text(json.dumps(fixture["query"]["baseline_rows"], indent=2, sort_keys=True) + "\n")


def create_temp_fixture_db() -> Path:
    """Create a temporary seeded fixture DB and return its path."""
    temp_file = NamedTemporaryFile(prefix="brainlayer-stale-index-", suffix=".db", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    create_fixture_db(temp_path)
    return temp_path
