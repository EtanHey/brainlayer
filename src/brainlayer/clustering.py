"""Hierarchical clustering pipeline for BrainLayer chunks.

Builds a 3-level cluster hierarchy using Recursive Leiden on a Faiss KNN graph.
Stores results in SQLite tables (clusters, chunk_clusters, vec_cluster_centroids).

Architecture:
    245K chunk embeddings (sqlite-vec, 1024 dims)
      → L2-normalize
        → Faiss IndexFlatIP k=30 KNN graph (~2 GB, 5-15 min)
          → igraph conversion
            → Recursive Leiden at 3 resolutions
              Level 0: ~40 clusters (resolution ~0.005)
              Level 1: ~400 clusters (~10 per L0, resolution ~0.05)
              Level 2: ~4000 clusters (~10 per L1, resolution ~0.5)
                → Centroids + materialized paths → SQLite

Usage:
    python3 -m brainlayer.clustering [--db-path PATH] [--k 30] [--dry-run]
"""

import argparse
import json
import logging
import struct
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import apsw
import faiss
import igraph as ig
import leidenalg
import numpy as np
import sqlite_vec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DB = Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db"
EMBEDDING_DIM = 1024

# Target cluster counts at each level
LEVEL_TARGETS = [40, 10, 10]  # L0: ~40 top, L1: ~10 per L0, L2: ~10 per L1


def serialize_f32(vector) -> bytes:
    """Serialize float32 vector to bytes for sqlite-vec.

    Note: Duplicated from vector_store.serialize_f32 because clustering.py
    runs as a standalone script (python -m brainlayer.clustering) and
    importing vector_store triggers heavy dependencies. Kept intentionally.
    """
    return struct.pack(f"{len(vector)}f", *vector)


# ─── Step 1: Extract Embeddings ─────────────────────────────────


def extract_embeddings(db_path: str):
    """Batch-read all embeddings from sqlite-vec.

    Returns:
        chunk_ids: list of chunk IDs (same order as embeddings)
        embeddings: numpy array of shape (N, 1024)
    """
    conn = apsw.Connection(db_path, flags=apsw.SQLITE_OPEN_READONLY)
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    cursor = conn.cursor()

    total = list(cursor.execute("SELECT COUNT(*) FROM chunk_vectors"))[0][0]
    logger.info(f"Extracting {total} embeddings...")

    # Read all chunk IDs first (ordered by rowid for deterministic ordering)
    logger.info("  Reading chunk IDs...")
    all_chunk_ids = [row[0] for row in cursor.execute("SELECT id FROM chunks ORDER BY rowid")]
    logger.info(f"  Got {len(all_chunk_ids)} chunk IDs")

    # Build a set for fast lookup + ordered list
    chunk_id_set = set(all_chunk_ids)

    # Read all vectors — vec0 doesn't support ORDER BY or OFFSET,
    # so we read everything and match by chunk_id
    logger.info("  Reading all vectors from chunk_vectors...")
    chunk_ids = []
    embeddings = []
    # AIDEV-NOTE: vec0 virtual tables return rows in insertion order (not guaranteed sorted).
    # We read all and re-order to match chunks.rowid order.
    vec_map = {}
    for chunk_id, emb_bytes in cursor.execute("SELECT chunk_id, embedding FROM chunk_vectors"):
        if emb_bytes and len(emb_bytes) == EMBEDDING_DIM * 4 and chunk_id in chunk_id_set:
            vec = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            vec_map[chunk_id] = vec

    # Re-order to match chunks.rowid order
    for cid in all_chunk_ids:
        if cid in vec_map:
            chunk_ids.append(cid)
            embeddings.append(vec_map[cid])

    del vec_map  # Free memory
    logger.info(f"  Matched {len(chunk_ids)}/{len(all_chunk_ids)} chunks with vectors")

    conn.close()
    embeddings_array = np.vstack(embeddings) if embeddings else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    logger.info(f"Extracted {len(chunk_ids)} embeddings, shape: {embeddings_array.shape}")
    return chunk_ids, embeddings_array


# ─── Step 2: Build KNN Graph ────────────────────────────────────


def build_knn_graph(embeddings: np.ndarray, k: int = 30):
    """Build KNN graph using Faiss IndexFlatIP after L2 normalization.

    L2 normalization converts dot product (IP) → cosine similarity.

    Returns:
        distances: (N, k) array of cosine similarities
        indices: (N, k) array of neighbor indices
    """
    n, d = embeddings.shape
    logger.info(f"L2-normalizing {n} embeddings...")
    faiss.normalize_L2(embeddings)  # in-place

    logger.info(f"Building Faiss IndexFlatIP for KNN (k={k})...")
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    logger.info(f"Searching {k} nearest neighbors for {n} vectors...")
    t0 = time.time()
    # k+1 because the first result is always the point itself
    distances, indices = index.search(embeddings, k + 1)
    elapsed = time.time() - t0
    logger.info(f"KNN search completed in {elapsed:.1f}s")

    # Remove self-matches (first column)
    return distances[:, 1:], indices[:, 1:]


# ─── Step 3: Convert to igraph ──────────────────────────────────


def knn_to_igraph(indices: np.ndarray, distances: np.ndarray, n: int):
    """Convert KNN results to a weighted undirected igraph graph.

    Collapses directed KNN edges into undirected with max weight.
    """
    logger.info(f"Building igraph from KNN ({n} nodes)...")

    edge_set = set()
    weight_map = {}

    for i in range(n):
        for j_pos in range(indices.shape[1]):
            j = int(indices[i, j_pos])
            if j < 0 or j >= n or j == i:
                continue
            w = float(distances[i, j_pos])
            edge = (min(i, j), max(i, j))
            if edge not in edge_set:
                edge_set.add(edge)
                weight_map[edge] = w
            else:
                weight_map[edge] = max(weight_map[edge], w)

    edges = list(edge_set)
    weights = [max(0.001, weight_map[e]) for e in edges]

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights

    logger.info(f"igraph: {g.vcount()} nodes, {g.ecount()} edges")
    return g


# ─── Step 4: Resolution Binary Search ───────────────────────────


def find_resolution_for_target(
    graph: ig.Graph,
    weights: list,
    target: int,
    lo: float = 0.0001,
    hi: float = 5.0,
    tolerance: float = 0.2,
    max_iters: int = 20,
) -> float:
    """Binary search for Leiden resolution that gives ~target clusters.

    tolerance: fraction of target we accept (e.g., 0.2 = within 20%)
    """
    best_res = (lo + hi) / 2
    best_diff = float("inf")

    for iteration in range(max_iters):
        mid = (lo + hi) / 2
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=mid,
            n_iterations=3,  # Fewer iterations for search
            seed=42,
        )
        n_clusters = len(set(partition.membership))

        diff = abs(n_clusters - target) / target
        if diff < best_diff:
            best_diff = diff
            best_res = mid

        if diff < tolerance:
            logger.info(f"  Resolution {mid:.6f} → {n_clusters} clusters (target: {target}, diff: {diff:.1%})")
            return mid

        if n_clusters > target:
            hi = mid  # Too many clusters → lower resolution
        else:
            lo = mid  # Too few → higher resolution

    logger.info(f"  Best resolution {best_res:.6f} after {max_iters} iters (diff: {best_diff:.1%})")
    return best_res


# ─── Step 5: Recursive Leiden ────────────────────────────────────


def recursive_leiden(
    graph: ig.Graph,
    node_indices: np.ndarray,
    level_targets: list,
    embeddings: np.ndarray,
    level: int = 0,
    parent_id: Optional[str] = None,
    parent_path: str = "",
):
    """Run Leiden recursively to build a guaranteed nested hierarchy.

    At each level, runs Leiden on the subgraph to split into ~target clusters.
    Then recurses into each cluster for the next level.

    Returns:
        list of cluster dicts: {id, level, parent_id, path, node_indices, centroid}
    """
    if level >= len(level_targets):
        return []

    target = level_targets[level]
    n = graph.vcount()

    if n < 3:
        # Too small to cluster further
        cluster_id = str(uuid.uuid4())[:12]
        path = f"{parent_path}/{cluster_id}" if parent_path else cluster_id
        centroid = embeddings[node_indices].mean(axis=0) if len(node_indices) > 0 else np.zeros(EMBEDDING_DIM)
        return [
            {
                "id": cluster_id,
                "level": level,
                "parent_id": parent_id,
                "path": path,
                "node_indices": node_indices,
                "centroid": centroid,
                "chunk_count": len(node_indices),
            }
        ]

    # Get weights for this subgraph
    weights = graph.es["weight"] if graph.es else None

    # Find resolution for target cluster count
    actual_target = min(target, max(2, n // 3))  # Don't try to make more clusters than we have points / 3
    resolution = find_resolution_for_target(graph, weights, actual_target)

    # Run final Leiden with more iterations
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        n_iterations=-1,  # Until convergence
        seed=42,
    )

    membership = partition.membership
    communities = defaultdict(list)
    for local_idx, comm_id in enumerate(membership):
        communities[comm_id].append(local_idx)

    n_clusters = len(communities)
    logger.info(f"Level {level}: {n_clusters} clusters from {n} nodes (res={resolution:.6f})")

    all_clusters = []

    for comm_id in sorted(communities.keys()):
        local_indices = communities[comm_id]
        global_indices = np.array([node_indices[i] for i in local_indices])

        cluster_id = str(uuid.uuid4())[:12]
        path = f"{parent_path}/{cluster_id}" if parent_path else cluster_id
        centroid = embeddings[global_indices].mean(axis=0)

        cluster = {
            "id": cluster_id,
            "level": level,
            "parent_id": parent_id,
            "path": path,
            "node_indices": global_indices,
            "centroid": centroid,
            "chunk_count": len(global_indices),
        }
        all_clusters.append(cluster)

        # Recurse into sub-clusters if we have enough points and more levels
        if level + 1 < len(level_targets) and len(local_indices) >= 6:
            subgraph = graph.subgraph(local_indices)
            sub_clusters = recursive_leiden(
                subgraph,
                global_indices,
                level_targets,
                embeddings,
                level=level + 1,
                parent_id=cluster_id,
                parent_path=path,
            )
            all_clusters.extend(sub_clusters)

    return all_clusters


# ─── Step 6: Write to SQLite ────────────────────────────────────


def create_cluster_schema(conn: apsw.Connection):
    """Create clustering tables in the BrainLayer DB."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id TEXT PRIMARY KEY,
            level INTEGER NOT NULL,
            parent_id TEXT,
            path TEXT NOT NULL,
            label TEXT,
            ctfidf_label TEXT,
            chunk_count INTEGER DEFAULT 0,
            silhouette_score REAL,
            avg_intra_dist REAL,
            created_at TEXT,
            updated_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunk_clusters (
            chunk_id TEXT NOT NULL,
            cluster_id TEXT NOT NULL,
            level INTEGER NOT NULL,
            dist_to_centroid REAL,
            assignment_method TEXT DEFAULT 'initial',
            assigned_at TEXT,
            PRIMARY KEY (chunk_id, level)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_clusters_cluster ON chunk_clusters(cluster_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_clusters_level ON chunk_clusters(level)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_clusters_level ON clusters(level)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_clusters_parent ON clusters(parent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_clusters_path ON clusters(path)")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clustering_runs (
            id TEXT PRIMARY KEY,
            started_at TEXT,
            completed_at TEXT,
            status TEXT DEFAULT 'running',
            total_chunks INTEGER,
            level_counts TEXT,
            params TEXT,
            silhouette_scores TEXT
        )
    """)

    # vec_cluster_centroids — virtual table for centroid KNN search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_cluster_centroids USING vec0(
            cluster_id TEXT PRIMARY KEY,
            centroid FLOAT[1024]
        )
    """)

    logger.info("Cluster schema created/verified")


def write_clusters(
    conn: apsw.Connection,
    clusters: list,
    chunk_ids: list,
    embeddings: np.ndarray,
    run_id: str,
):
    """Write cluster hierarchy and chunk assignments to SQLite.

    Wrapped in a single transaction — either all clusters are written or none
    (prevents partial state if an insert fails after clearing old data).
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA busy_timeout = 5000")
    now = datetime.now(timezone.utc).isoformat()

    cursor.execute("BEGIN")
    try:
        # Clear previous data
        cursor.execute("DELETE FROM clusters")
        cursor.execute("DELETE FROM chunk_clusters")
        cursor.execute("DELETE FROM vec_cluster_centroids")

        logger.info(f"Writing {len(clusters)} clusters...")

        for cluster in clusters:
            # Write cluster
            cursor.execute(
                "INSERT INTO clusters (id, level, parent_id, path, chunk_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cluster["id"],
                    cluster["level"],
                    cluster["parent_id"],
                    cluster["path"],
                    cluster["chunk_count"],
                    now,
                    now,
                ),
            )

            # Write centroid (L2-normalize for cosine similarity via sqlite-vec match)
            centroid = cluster["centroid"].copy()
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            cursor.execute(
                "INSERT INTO vec_cluster_centroids (cluster_id, centroid) VALUES (?, ?)",
                (cluster["id"], serialize_f32(centroid.tolist())),
            )

            # Write chunk assignments
            for global_idx in cluster["node_indices"]:
                cid = chunk_ids[global_idx]
                dist = float(np.linalg.norm(embeddings[global_idx] - cluster["centroid"]))
                cursor.execute(
                    "INSERT OR REPLACE INTO chunk_clusters (chunk_id, cluster_id, level, dist_to_centroid, assignment_method, assigned_at) VALUES (?, ?, ?, ?, 'initial', ?)",
                    (cid, cluster["id"], cluster["level"], dist, now),
                )

        cursor.execute("COMMIT")
        logger.info(f"Written {len(clusters)} clusters to DB")
    except Exception:
        cursor.execute("ROLLBACK")
        logger.error("Failed to write clusters — rolled back")
        raise


# ─── Step 7: c-TF-IDF Labeling ──────────────────────────────────


def generate_ctfidf_labels(
    conn: apsw.Connection,
    clusters: list,
    chunk_ids: list,
):
    """Generate c-TF-IDF labels for all clusters.

    For each cluster, gets the top-3 discriminative terms from its member chunks.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    cursor = conn.cursor()
    cursor.execute("PRAGMA busy_timeout = 5000")

    # Build a mapping from global index to chunk content (sample for efficiency)
    logger.info("Loading chunk content for labeling...")
    content_map = {}
    for row in cursor.execute("SELECT id, content FROM chunks"):
        content_map[row[0]] = row[1] or ""

    # Group clusters by level
    by_level = defaultdict(list)
    for c in clusters:
        by_level[c["level"]].append(c)

    for level in sorted(by_level.keys()):
        level_clusters = by_level[level]
        logger.info(f"Labeling {len(level_clusters)} clusters at level {level}...")

        # Build per-cluster documents
        docs = []
        cluster_ids_for_docs = []
        for c in level_clusters:
            # Sample up to 50 chunks per cluster for labeling
            indices = c["node_indices"][:50] if len(c["node_indices"]) > 50 else c["node_indices"]
            texts = []
            for idx in indices:
                cid = chunk_ids[idx]
                text = content_map.get(cid, "")
                if text:
                    texts.append(text[:2000])
            doc = " ".join(texts)
            if doc.strip():
                docs.append(doc)
                cluster_ids_for_docs.append(c["id"])

        if not docs:
            continue

        vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
        )

        try:
            tfidf = vectorizer.fit_transform(docs)
        except ValueError:
            continue

        feature_names = vectorizer.get_feature_names_out()

        for i, cluster_id in enumerate(cluster_ids_for_docs):
            top_indices = tfidf[i].toarray()[0].argsort()[-5:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices if tfidf[i, idx] > 0]
            label = " / ".join(top_terms[:3]) if top_terms else f"cluster-{cluster_id}"

            cursor.execute(
                "UPDATE clusters SET ctfidf_label = ?, label = ? WHERE id = ?",
                (label, label, cluster_id),
            )

    logger.info("c-TF-IDF labeling complete")


# ─── Step 8: Silhouette Score ────────────────────────────────────


def compute_silhouette_sample(
    embeddings: np.ndarray,
    clusters: list,
    chunk_ids: list,
    sample_size: int = 5000,
):
    """Compute silhouette score on a sample for each level.

    Returns dict of {level: silhouette_score}.
    """
    from sklearn.metrics import silhouette_score

    # Build level-specific label arrays
    by_level = defaultdict(list)
    for c in clusters:
        by_level[c["level"]].append(c)

    scores = {}
    for level in sorted(by_level.keys()):
        level_clusters = by_level[level]

        # Build label array (global_idx → cluster_id)
        label_map = {}
        for c in level_clusters:
            for idx in c["node_indices"]:
                label_map[idx] = c["id"]

        indices = sorted(label_map.keys())
        if len(indices) < 10:
            continue

        # Sample for efficiency
        if len(indices) > sample_size:
            rng = np.random.default_rng(42)
            indices = sorted(rng.choice(indices, sample_size, replace=False))

        X = embeddings[indices]
        labels = [label_map[i] for i in indices]

        # Need at least 2 unique labels
        if len(set(labels)) < 2:
            continue

        try:
            score = silhouette_score(X, labels, metric="cosine", sample_size=min(len(X), sample_size))
            scores[level] = float(score)
            logger.info(f"Silhouette score L{level}: {score:.4f}")
        except Exception as e:
            logger.warning(f"Silhouette failed for L{level}: {e}")

    return scores


# ─── Main Pipeline ───────────────────────────────────────────────


def run_clustering(
    db_path: str = str(DEFAULT_DB),
    k: int = 30,
    level_targets: list = None,
    dry_run: bool = False,
):
    """Run the full clustering pipeline."""
    if level_targets is None:
        level_targets = LEVEL_TARGETS

    run_id = str(uuid.uuid4())[:12]
    t0 = time.time()

    # Step 1: Extract
    logger.info("=" * 60)
    logger.info("STEP 1: Extracting embeddings")
    logger.info("=" * 60)
    chunk_ids, embeddings = extract_embeddings(db_path)
    n = len(chunk_ids)

    if dry_run:
        logger.info(f"DRY RUN: {n} chunks extracted. Would cluster with k={k}, targets={level_targets}")
        return

    # Step 2: KNN
    logger.info("=" * 60)
    logger.info("STEP 2: Building KNN graph")
    logger.info("=" * 60)
    distances, indices = build_knn_graph(embeddings, k=k)

    # Step 3: igraph
    logger.info("=" * 60)
    logger.info("STEP 3: Converting to igraph")
    logger.info("=" * 60)
    graph = knn_to_igraph(indices, distances, n)

    # Step 4+5: Recursive Leiden
    logger.info("=" * 60)
    logger.info("STEP 4: Running recursive Leiden clustering")
    logger.info("=" * 60)
    all_indices = np.arange(n)
    clusters = recursive_leiden(graph, all_indices, level_targets, embeddings)

    # Report cluster counts
    by_level = defaultdict(int)
    for c in clusters:
        by_level[c["level"]] += 1
    level_counts = {f"L{lv}": cnt for lv, cnt in sorted(by_level.items())}
    logger.info(f"Cluster counts: {level_counts}")

    # Step 6: Write to DB
    logger.info("=" * 60)
    logger.info("STEP 5: Writing to SQLite")
    logger.info("=" * 60)
    conn = apsw.Connection(db_path)
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)

    create_cluster_schema(conn)
    write_clusters(conn, clusters, chunk_ids, embeddings, run_id)

    # Step 7: c-TF-IDF labels
    logger.info("=" * 60)
    logger.info("STEP 6: Generating c-TF-IDF labels")
    logger.info("=" * 60)
    generate_ctfidf_labels(conn, clusters, chunk_ids)

    # Step 8: Silhouette
    logger.info("=" * 60)
    logger.info("STEP 7: Computing silhouette scores")
    logger.info("=" * 60)
    silhouette_scores = compute_silhouette_sample(embeddings, clusters, chunk_ids)

    # Record run
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO clustering_runs (id, started_at, completed_at, status, total_chunks, level_counts, params, silhouette_scores) VALUES (?, ?, ?, 'completed', ?, ?, ?, ?)",
        (
            run_id,
            datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
            now,
            n,
            json.dumps(level_counts),
            json.dumps({"k": k, "level_targets": level_targets}),
            json.dumps(silhouette_scores),
        ),
    )

    conn.close()

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"DONE in {elapsed / 60:.1f} minutes")
    logger.info(f"  Chunks: {n}")
    logger.info(f"  Clusters: {level_counts}")
    logger.info(f"  Silhouette: {silhouette_scores}")
    logger.info(f"  Run ID: {run_id}")
    logger.info("=" * 60)

    return {
        "run_id": run_id,
        "total_chunks": n,
        "level_counts": level_counts,
        "silhouette_scores": silhouette_scores,
        "elapsed_minutes": elapsed / 60,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hierarchical clustering on BrainLayer chunks")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--k", type=int, default=30, help="KNN neighbors")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--l0", type=int, default=40, help="Target L0 clusters")
    parser.add_argument("--l1", type=int, default=10, help="Target L1 per L0")
    parser.add_argument("--l2", type=int, default=10, help="Target L2 per L1")
    args = parser.parse_args()

    run_clustering(
        db_path=args.db_path,
        k=args.k,
        level_targets=[args.l0, args.l1, args.l2],
        dry_run=args.dry_run,
    )
