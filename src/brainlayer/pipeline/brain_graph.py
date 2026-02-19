"""
Brain Graph Pipeline — Transform BrainLayer chunks into a visualization-ready graph.json.

Takes 240K+ chunks, aggregates to session level, computes hybrid similarity,
runs Leiden community detection, and generates a 500-2K node graph with
pre-computed 3D coordinates via UMAP.

Usage:
    brainlayer brain-export [--output PATH] [--project NAME]
"""

import json
import logging
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path.home() / ".brainlayer-brain"
MIN_CHUNKS_PER_SESSION = 3  # Skip tiny sessions
MAX_SESSIONS = 5000  # Cap for performance
KNN_NEIGHBORS = 15  # Each node connects to its K nearest neighbors
LEIDEN_RESOLUTIONS = [0.3, 0.8, 2.0]  # Coarse → medium → fine


# ─── Data Loading ───────────────────────────────────────────────────


def load_sessions(db_path: str, project: Optional[str] = None) -> list[dict]:
    """Load session data from BrainLayer DB, using source_file as the session unit.

    Each source_file (JSONL conversation) becomes one node in the graph.
    Falls back to session_context if available, but builds from chunks otherwise.
    """
    import apsw
    import sqlite_vec

    conn = apsw.Connection(db_path, flags=apsw.SQLITE_OPEN_READONLY)
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    cursor = conn.cursor()

    # Build session_context lookup (sparse — only recent sessions have this)
    context_by_sid = {}
    for row in cursor.execute(
        "SELECT session_id, project, branch, pr_number, files_changed, "
        "started_at, ended_at, plan_name, plan_phase FROM session_context"
    ):
        sid, proj, branch, pr, files_json, started, ended, plan, phase = row
        context_by_sid[sid] = {
            "project": proj or "",
            "branch": branch or "",
            "pr_number": pr,
            "files": json.loads(files_json) if files_json else [],
            "started_at": started or "",
            "ended_at": ended or "",
            "plan_name": plan or "",
            "plan_phase": phase or "",
        }

    # Get all source_files with their chunk counts and metadata
    query = """
        SELECT source_file, project, COUNT(*) as chunk_count,
               GROUP_CONCAT(DISTINCT content_type) as types,
               GROUP_CONCAT(DISTINCT intent) as intents,
               AVG(CASE WHEN importance IS NOT NULL THEN importance END) as avg_importance,
               (SELECT source FROM chunks c2
                WHERE c2.source_file = chunks.source_file AND c2.source IS NOT NULL
                GROUP BY source ORDER BY COUNT(*) DESC LIMIT 1) as dominant_source
        FROM chunks
    """
    params = []
    if project:
        query += " WHERE project = ?"
        params.append(project)
    query += " GROUP BY source_file HAVING chunk_count >= ? ORDER BY chunk_count DESC"
    params.append(MIN_CHUNKS_PER_SESSION)

    source_files = list(cursor.execute(query, params))
    logger.info(f"Found {len(source_files)} source files with >= {MIN_CHUNKS_PER_SESSION} chunks")

    # Cap to MAX_SESSIONS by chunk count (biggest sessions = most informative)
    if len(source_files) > MAX_SESSIONS:
        source_files = source_files[:MAX_SESSIONS]
        logger.info(f"Capped to {MAX_SESSIONS} source files")

    # Build sessions from source files
    sessions = []
    embed_dim = None
    EMBED_SAMPLE = 20  # Sample up to 20 embeddings per session for mean-pooling

    for i, (src_file, proj, chunk_count, types_str, intents_str, avg_imp, dominant_source) in enumerate(source_files):
        if i % 200 == 0 and i > 0:
            logger.info(f"Processing session {i}/{len(source_files)}...")

        # Extract session ID from source_file path (UUID part)
        fname = Path(src_file).stem
        sid = fname if len(fname) <= 40 else fname[:36]  # UUID or truncated

        # Try to match with session_context
        ctx = None
        for ctx_sid, ctx_data in context_by_sid.items():
            if ctx_sid[:8] in fname:
                ctx = ctx_data
                break

        # Get sample of content for TF-IDF (first 20 chunks)
        text_rows = list(cursor.execute(
            "SELECT content FROM chunks WHERE source_file = ? AND content IS NOT NULL "
            "ORDER BY ROWID ASC LIMIT 20",
            (src_file,),
        ))
        text = " ".join(row[0][:2000] for row in text_rows if row[0])

        # Get sampled embeddings (evenly spaced for representative mean)
        embed_rows = list(cursor.execute(
            "SELECT cv.embedding FROM chunk_vectors cv "
            "JOIN chunks c ON cv.chunk_id = c.id "
            "WHERE c.source_file = ? ORDER BY c.ROWID ASC",
            (src_file,),
        ))

        embedding = None
        if embed_rows:
            # Sample evenly from available embeddings
            step = max(1, len(embed_rows) // EMBED_SAMPLE)
            sampled = embed_rows[::step][:EMBED_SAMPLE]
            vecs = []
            for row in sampled:
                if row[0] and len(row[0]) >= 4:
                    vec = np.array(struct.unpack(f"{len(row[0]) // 4}f", row[0]))
                    vecs.append(vec)
                    if embed_dim is None:
                        embed_dim = len(vec)
            if vecs:
                embedding = np.mean(vecs, axis=0)

        if embedding is None:
            continue

        content_types = Counter(t.strip() for t in (types_str or "").split(",") if t.strip())
        intents = Counter(t.strip() for t in (intents_str or "").split(",") if t.strip())

        sessions.append({
            "id": sid,
            "source_file": src_file,
            "project": (ctx["project"] if ctx else proj) or "",
            "branch": ctx["branch"] if ctx else "",
            "pr_number": ctx["pr_number"] if ctx else None,
            "files": ctx["files"] if ctx else [],
            "started_at": ctx["started_at"] if ctx else "",
            "ended_at": ctx["ended_at"] if ctx else "",
            "plan_name": ctx["plan_name"] if ctx else "",
            "plan_phase": ctx["plan_phase"] if ctx else "",
            "chunk_count": chunk_count,
            "content_types": content_types,
            "intents": intents,
            "importance": float(avg_imp) if avg_imp is not None else 5.0,
            "source": dominant_source or "claude_code",
            "text": text,
            "embedding": embedding,
        })

    logger.info(f"Built {len(sessions)} sessions with embeddings")
    return sessions


# ─── Similarity Matrix ──────────────────────────────────────────────


def compute_similarity_matrix(sessions: list[dict]) -> np.ndarray:
    """Build hybrid similarity: 40% semantic + 35% file overlap + 15% temporal + 10% branch."""
    n = len(sessions)
    logger.info(f"Computing {n}x{n} similarity matrix...")
    t0 = time.time()

    # 1. Semantic similarity (cosine on mean embeddings) — 40%
    embeddings = np.array([s["embedding"] for s in sessions])
    semantic_sim = cosine_similarity(embeddings)
    np.fill_diagonal(semantic_sim, 0)

    # 2. File overlap (Jaccard) — 35%
    file_sim = np.zeros((n, n))
    file_sets = [set(s["files"]) for s in sessions]
    for i in range(n):
        if not file_sets[i]:
            continue
        for j in range(i + 1, n):
            if not file_sets[j]:
                continue
            intersection = len(file_sets[i] & file_sets[j])
            union = len(file_sets[i] | file_sets[j])
            if union > 0:
                sim = intersection / union
                file_sim[i, j] = sim
                file_sim[j, i] = sim

    # 3. Temporal proximity — 15%
    temporal_sim = np.zeros((n, n))
    timestamps = []
    for s in sessions:
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(s["started_at"].replace("Z", "+00:00"))
            timestamps.append(ts.timestamp())
        except (ValueError, AttributeError):
            timestamps.append(0)

    for i in range(n):
        if not timestamps[i]:
            continue
        for j in range(i + 1, n):
            if not timestamps[j]:
                continue
            hours_apart = abs(timestamps[i] - timestamps[j]) / 3600
            # Decay: sessions within 24h are similar, beyond 7 days → 0
            sim = max(0, 1 - hours_apart / 168)  # 168 hours = 7 days
            temporal_sim[i, j] = sim
            temporal_sim[j, i] = sim

    # 4. Branch/PR similarity — 10%
    branch_sim = np.zeros((n, n))
    for i in range(n):
        if not sessions[i]["branch"]:
            continue
        for j in range(i + 1, n):
            if sessions[i]["branch"] == sessions[j]["branch"]:
                branch_sim[i, j] = 1.0
                branch_sim[j, i] = 1.0
            elif sessions[i]["plan_name"] and sessions[i]["plan_name"] == sessions[j]["plan_name"]:
                branch_sim[i, j] = 0.7
                branch_sim[j, i] = 0.7

    # Combine with weights
    hybrid = (
        0.40 * semantic_sim
        + 0.35 * file_sim
        + 0.15 * temporal_sim
        + 0.10 * branch_sim
    )

    elapsed = time.time() - t0
    logger.info(f"Similarity matrix computed in {elapsed:.1f}s")
    return hybrid


# ─── Community Detection ────────────────────────────────────────────


def detect_communities(
    similarity: np.ndarray,
    sessions: list[dict],
) -> dict[str, list[int]]:
    """Run hierarchical Leiden community detection at multiple resolutions.

    Uses KNN graph instead of threshold — each node connects to its K nearest
    neighbors, guaranteeing every node has connections. This avoids the
    threshold cliff problem (similarity distribution has a sharp drop-off).
    """
    import igraph as ig
    import leidenalg

    n = len(sessions)
    k = min(KNN_NEIGHBORS, n - 1)

    # Build symmetric KNN graph
    edge_set: set[tuple[int, int]] = set()
    weight_map: dict[tuple[int, int], float] = {}

    for i in range(n):
        # Get top-K most similar nodes for this node
        sims = similarity[i].copy()
        sims[i] = -1  # Exclude self
        top_k = np.argpartition(sims, -k)[-k:]
        for j_idx in top_k:
            j = int(j_idx)
            edge = (min(i, j), max(i, j))
            if edge not in edge_set:
                edge_set.add(edge)
                weight_map[edge] = float(similarity[i, j])
            else:
                weight_map[edge] = max(weight_map[edge], float(similarity[i, j]))

    edges = list(edge_set)
    weights = [max(0.001, weight_map[e]) for e in edges]  # Leiden rejects negatives

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights

    logger.info(f"KNN graph (k={k}): {g.vcount()} nodes, {g.ecount()} edges")

    hierarchy = {}
    for resolution in LEIDEN_RESOLUTIONS:
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights if weights else None,
            resolution_parameter=resolution,
            n_iterations=-1,  # Run until convergence
        )
        label = (
            "coarse" if resolution == LEIDEN_RESOLUTIONS[0]
            else "fine" if resolution == LEIDEN_RESOLUTIONS[-1]
            else "medium"
        )
        hierarchy[label] = partition.membership
        n_communities = len(set(partition.membership))
        logger.info(f"Leiden {label} (res={resolution}): {n_communities} communities, modularity={partition.modularity:.3f}")

    return hierarchy


# ─── Cluster Labeling ───────────────────────────────────────────────


def label_communities(
    sessions: list[dict],
    membership: list[int],
) -> dict[int, str]:
    """Generate c-TF-IDF labels for each community."""
    communities = defaultdict(list)
    for idx, comm_id in enumerate(membership):
        communities[comm_id].append(idx)

    # Build per-community documents
    docs = []
    comm_ids = sorted(communities.keys())
    for comm_id in comm_ids:
        members = communities[comm_id]
        text = " ".join(sessions[i]["text"][:5000] for i in members)
        docs.append(text)

    if not docs:
        return {}

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8,
    )
    try:
        tfidf = vectorizer.fit_transform(docs)
    except ValueError:
        return {c: f"cluster-{c}" for c in comm_ids}

    feature_names = vectorizer.get_feature_names_out()
    labels = {}
    for i, comm_id in enumerate(comm_ids):
        top_indices = tfidf[i].toarray()[0].argsort()[-3:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices if tfidf[i, idx] > 0]
        labels[comm_id] = " / ".join(top_terms[:3]) if top_terms else f"cluster-{comm_id}"

    return labels


# ─── UMAP 3D Layout ────────────────────────────────────────────────


def compute_layout(sessions: list[dict]) -> np.ndarray:
    """Compute 3D positions via UMAP on session embeddings."""
    import umap

    embeddings = np.array([s["embedding"] for s in sessions])
    n = len(sessions)

    n_neighbors = min(15, n - 1) if n > 1 else 1
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=0.3,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    # Normalize to [-50, 50] range for Three.js
    for dim in range(3):
        mn, mx = coords[:, dim].min(), coords[:, dim].max()
        if mx > mn:
            coords[:, dim] = (coords[:, dim] - mn) / (mx - mn) * 100 - 50

    logger.info(f"UMAP 3D layout computed for {n} nodes")
    return coords


# ─── Graph Export ───────────────────────────────────────────────────


def dominant_type(counter: Counter) -> str:
    """Get the most common content type or intent."""
    if not counter:
        return "unknown"
    return counter.most_common(1)[0][0]


def build_graph_json(
    sessions: list[dict],
    similarity: np.ndarray,
    hierarchy: dict[str, list[int]],
    labels: dict[int, str],
    coords: np.ndarray,
) -> dict:
    """Build the final graph.json structure."""
    medium_membership = hierarchy.get("medium", hierarchy.get("coarse", [0] * len(sessions)))

    nodes = []
    for i, s in enumerate(sessions):
        comm = medium_membership[i] if i < len(medium_membership) else 0
        nodes.append({
            "id": s["id"][:8],
            "session_id": s["id"],
            "label": labels.get(comm, f"cluster-{comm}"),
            "community": {
                level: membership[i] if i < len(membership) else 0
                for level, membership in hierarchy.items()
            },
            "x": round(float(coords[i, 0]), 2),
            "y": round(float(coords[i, 1]), 2),
            "z": round(float(coords[i, 2]), 2),
            "size": round(float(np.clip(s["importance"] / 10 * 5 + 1, 1, 8)), 2),
            "color_type": dominant_type(s["intents"]),
            "source": s.get("source", "claude_code"),
            "project": s["project"],
            "branch": s["branch"],
            "plan": s["plan_name"],
            "chunk_count": s["chunk_count"],
            "files_count": len(s["files"]),
            "started_at": s["started_at"],
            "importance": round(float(s["importance"]), 2),
        })

    # Build edges (top N per node to avoid clutter)
    edges = []
    n = len(sessions)
    max_edges_per_node = 5
    for i in range(n):
        # Get top connections for this node
        row = similarity[i].copy()
        row[i] = 0
        top_j = np.argsort(row)[-max_edges_per_node:]
        for j in top_j:
            if row[j] > 0 and i < j:
                edges.append({
                    "source": sessions[i]["id"][:8],
                    "target": sessions[j]["id"][:8],
                    "weight": round(float(row[j]), 3),
                })

    # Build hierarchy info
    hierarchy_info = {}
    for level, membership in hierarchy.items():
        communities = defaultdict(list)
        for idx, comm_id in enumerate(membership):
            communities[comm_id].append(sessions[idx]["id"][:8])
        hierarchy_info[level] = {
            str(comm_id): {
                "label": labels.get(comm_id, f"cluster-{comm_id}"),
                "members": members,
                "size": len(members),
            }
            for comm_id, members in communities.items()
        }

    return {
        "nodes": nodes,
        "edges": edges,
        "hierarchy": hierarchy_info,
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_count": len(sessions),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "community_counts": {
                level: len(set(m)) for level, m in hierarchy.items()
            },
        },
    }


# ─── Main Pipeline ──────────────────────────────────────────────────


DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db"


def generate_brain_graph(
    db_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    project: Optional[str] = None,
) -> Path:
    """Run the full brain graph pipeline."""
    db = db_path or str(DEFAULT_DB_PATH)
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Step 1: Load sessions
    logger.info("Step 1: Loading sessions...")
    sessions = load_sessions(db, project)
    if len(sessions) < 2:
        raise ValueError(f"Need at least 2 sessions, found {len(sessions)}")

    # Step 2: Compute similarity matrix
    logger.info("Step 2: Computing similarity matrix...")
    similarity = compute_similarity_matrix(sessions)

    # Step 3: Community detection
    logger.info("Step 3: Running Leiden community detection...")
    hierarchy = detect_communities(similarity, sessions)

    # Step 4: Label communities
    logger.info("Step 4: Generating community labels...")
    medium = hierarchy.get("medium", hierarchy.get("coarse", []))
    labels = label_communities(sessions, medium)

    # Step 5: UMAP 3D layout
    logger.info("Step 5: Computing UMAP 3D layout...")
    coords = compute_layout(sessions)

    # Step 6: Build and export graph
    logger.info("Step 6: Building graph.json...")
    graph = build_graph_json(sessions, similarity, hierarchy, labels, coords)

    graph_path = out / "graph.json"
    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=2)

    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(graph["meta"], f, indent=2)

    elapsed = time.time() - t0
    logger.info(
        f"Brain graph exported to {graph_path} "
        f"({graph['meta']['node_count']} nodes, {graph['meta']['edge_count']} edges) "
        f"in {elapsed:.1f}s"
    )

    return graph_path
