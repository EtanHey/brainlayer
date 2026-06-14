"""Seeded isolation-proof harness for BrainLayer consumer scoping.

The harness deliberately uses deterministic local embeddings so it can run in
tests, CI, and cmux smoke probes without starting an embedder.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from ._helpers import serialize_f32
from .chunk_origin import CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT
from .scoping import ConsumerScope, resolve_consumer_scope
from .search_repo import clear_hybrid_search_cache
from .vector_store import VectorStore

QUERY_TEXT = "happy camper isolation proof sentinel"

BASIC_PROOF_EXPECTATIONS: dict[str, set[str]] = {
    "worker-repo-a": {"repo-a-main-proof"},
    "orchestrator": {
        "repo-a-main-proof",
        "repo-a-worktree-proof",
        "repo-a-checkpoint-proof",
        "repo-b-main-proof",
        "repo-b-worktree-proof",
        "repo-b-checkpoint-proof",
        "personal-checkpoint-proof",
        "null-user-local-proof",
    },
    "coach": {"personal-checkpoint-proof", "null-user-local-proof"},
}

EXTENSION_PROOF_EXPECTATIONS: dict[str, set[str]] = {
    "lead-repo-a": {
        "repo-a-main-proof",
        "repo-a-worktree-proof",
        "repo-b-main-proof",
        "repo-b-worktree-proof",
    },
    "worker-repo-a-main": {"repo-a-main-proof"},
    "worker-repo-a-worktree": {"repo-a-main-proof", "repo-a-worktree-proof"},
}


@dataclass(frozen=True)
class IsolationProofFixture:
    db_path: Path
    seeded_ids: set[str]


@dataclass(frozen=True)
class ScopeProbe:
    name: str
    consumer: str
    project: str | None
    include_checkpoints: bool = False
    project_filters: tuple[str, ...] = ()


@dataclass(frozen=True)
class IsolationProofReport:
    visible_ids_by_probe: dict[str, set[str]]
    failures: list[str]

    def as_jsonable(self) -> dict:
        return {
            "visible_ids_by_probe": {
                name: sorted(visible_ids) for name, visible_ids in sorted(self.visible_ids_by_probe.items())
            },
            "failures": list(self.failures),
        }


def _embed(text: str) -> list[float]:
    seed = (sum(ord(character) for character in text[:80]) % 97) / 1000.0
    return [seed + (index / 10000.0) for index in range(1024)]


def _insert_proof_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    project: str | None,
    content: str,
    created_at: str,
    chunk_origin: str | None = None,
) -> None:
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, created_at, chunk_origin
        ) VALUES (?, ?, ?, 'happy-camper-isolation-proof.jsonl', ?, 'assistant_text', ?, 'claude_code', ?, ?)""",
        (
            chunk_id,
            content,
            json.dumps({"proof_fixture": "happy-camper-isolation"}),
            project,
            len(content),
            created_at,
            chunk_origin,
        ),
    )
    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_f32(_embed(QUERY_TEXT))),
    )


def seed_isolation_proof_db(db_path: str | Path) -> IsolationProofFixture:
    """Create a self-contained DB with repo, worktree, local, and checkpoint rows."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    seeded_ids = {
        "repo-a-main-proof",
        "repo-a-worktree-proof",
        "repo-a-checkpoint-proof",
        "repo-b-main-proof",
        "repo-b-worktree-proof",
        "repo-b-checkpoint-proof",
        "personal-checkpoint-proof",
        "null-user-local-proof",
    }
    store = VectorStore(path)
    store._binary_index_available = False
    try:
        _insert_proof_chunk(
            store,
            chunk_id="repo-a-main-proof",
            project="repo-a",
            content=f"{QUERY_TEXT} repo A main branch memory",
            created_at="2026-06-14T00:00:00Z",
        )
        _insert_proof_chunk(
            store,
            chunk_id="repo-a-worktree-proof",
            project="repo-a.worktree.feature-x",
            content=f"{QUERY_TEXT} repo A feature worktree memory",
            created_at="2026-06-14T00:01:00Z",
        )
        _insert_proof_chunk(
            store,
            chunk_id="repo-a-checkpoint-proof",
            project="repo-a",
            content=f"[precompact checkpoint] {QUERY_TEXT} repo A checkpoint memory",
            created_at="2026-06-14T00:02:00Z",
            chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        )
        _insert_proof_chunk(
            store,
            chunk_id="repo-b-main-proof",
            project="repo-b",
            content=f"{QUERY_TEXT} repo B main branch memory",
            created_at="2026-06-14T00:03:00Z",
        )
        _insert_proof_chunk(
            store,
            chunk_id="repo-b-worktree-proof",
            project="repo-b.worktree.parallel",
            content=f"{QUERY_TEXT} repo B parallel worktree memory",
            created_at="2026-06-14T00:04:00Z",
        )
        _insert_proof_chunk(
            store,
            chunk_id="repo-b-checkpoint-proof",
            project="repo-b",
            content=f"[precompact checkpoint] {QUERY_TEXT} repo B checkpoint memory",
            created_at="2026-06-14T00:05:00Z",
            chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        )
        _insert_proof_chunk(
            store,
            chunk_id="personal-checkpoint-proof",
            project="personal",
            content=f"[precompact checkpoint] {QUERY_TEXT} personal coach checkpoint memory",
            created_at="2026-06-14T00:06:00Z",
            chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        )
        _insert_proof_chunk(
            store,
            chunk_id="null-user-local-proof",
            project=None,
            content=f"{QUERY_TEXT} user-local null-project memory",
            created_at="2026-06-14T00:07:00Z",
        )
    finally:
        store.close()
    clear_hybrid_search_cache(path)
    return IsolationProofFixture(db_path=path, seeded_ids=seeded_ids)


def _scope_for_probe(probe: ScopeProbe) -> ConsumerScope:
    if probe.project_filters:
        return ConsumerScope(
            role=probe.consumer,
            project_filter=probe.project,
            project_filters=probe.project_filters,
            include_checkpoints=probe.include_checkpoints,
            allow_null_project=False,
            deny_all=probe.project is None,
        )
    return resolve_consumer_scope(
        project=probe.project,
        consumer=probe.consumer,
        include_checkpoints=probe.include_checkpoints,
    )


def run_basic_isolation_proof(
    db_path: str | Path,
    probes: list[ScopeProbe] | None = None,
    expectations: dict[str, set[str]] | None = None,
) -> IsolationProofReport:
    """Run role probes against the seeded DB and compare them with expected IDs."""
    selected_probes = probes or [
        ScopeProbe(name="worker-repo-a", consumer="worker", project="repo-a"),
        ScopeProbe(name="orchestrator", consumer="orchestrator", project=None, include_checkpoints=True),
        ScopeProbe(name="coach", consumer="coach", project=None),
    ]
    expected_by_probe = expectations or {**BASIC_PROOF_EXPECTATIONS, **EXTENSION_PROOF_EXPECTATIONS}
    visible_ids_by_probe: dict[str, set[str]] = {}
    failures: list[str] = []

    store = VectorStore(Path(db_path), readonly=True)
    store._binary_index_available = False
    try:
        for probe in selected_probes:
            scope = _scope_for_probe(probe)
            results = store.hybrid_search(
                query_embedding=_embed(QUERY_TEXT),
                query_text=QUERY_TEXT,
                n_results=20,
                consumer_scope=scope,
            )
            visible_ids = set(results["ids"][0])
            visible_ids_by_probe[probe.name] = visible_ids
            expected = expected_by_probe.get(probe.name)
            if expected is not None and visible_ids != expected:
                failures.append(f"{probe.name}: expected {sorted(expected)}, got {sorted(visible_ids)}")
    finally:
        store.close()

    return IsolationProofReport(visible_ids_by_probe=visible_ids_by_probe, failures=failures)


def run_extension_isolation_proof(db_path: str | Path) -> IsolationProofReport:
    return run_basic_isolation_proof(
        db_path,
        probes=[
            ScopeProbe(
                name="lead-repo-a",
                consumer="lead",
                project="repo-a",
                project_filters=(
                    "repo-a",
                    "repo-a.worktree.feature-x",
                    "repo-b",
                    "repo-b.worktree.parallel",
                ),
            ),
            ScopeProbe(name="worker-repo-a-main", consumer="worker", project="repo-a"),
            ScopeProbe(
                name="worker-repo-a-worktree",
                consumer="worker",
                project="repo-a.worktree.feature-x",
                project_filters=("repo-a.worktree.feature-x", "repo-a"),
            ),
        ],
        expectations=EXTENSION_PROOF_EXPECTATIONS,
    )


def run_full_isolation_proof(db_path: str | Path) -> IsolationProofReport:
    basic = run_basic_isolation_proof(db_path)
    extension = run_extension_isolation_proof(db_path)
    return IsolationProofReport(
        visible_ids_by_probe={**basic.visible_ids_by_probe, **extension.visible_ids_by_probe},
        failures=[*basic.failures, *extension.failures],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and run the Happy Camper BrainLayer isolation proof DB.")
    parser.add_argument("--db", type=Path, required=True, help="Path to write the seeded proof SQLite DB")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    fixture = seed_isolation_proof_db(args.db)
    report = run_full_isolation_proof(fixture.db_path)
    payload = {
        "db_path": str(fixture.db_path),
        "seeded_ids": sorted(fixture.seeded_ids),
        **report.as_jsonable(),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"DB: {fixture.db_path}")
        print(f"Seeded: {', '.join(payload['seeded_ids'])}")
        for name, visible_ids in payload["visible_ids_by_probe"].items():
            print(f"{name}: {', '.join(visible_ids)}")
        print("PASS" if not report.failures else "FAIL")
        for failure in report.failures:
            print(f"- {failure}")
    return 0 if not report.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
