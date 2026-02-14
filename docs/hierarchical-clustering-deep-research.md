# Hierarchical knowledge clustering for Zikaron's personal knowledge graph

**Recursive Leiden community detection on a Faiss-built KNN graph is the optimal algorithm for Zikaron's 245K chunks**, running entirely on an M1 Pro in under 30 minutes at ~3–4 GB peak memory. This approach delivers a clean 3-level hierarchy with direct control over cluster counts, supports incremental updates, and leverages the user's existing Leiden experience. The biggest risk isn't the clustering itself but the Hebrew embeddings: bge-large-en-v1.5 produces near-random vectors for Hebrew text, meaning **16K WhatsApp chunks need re-embedding with BGE-M3** before clustering can work cross-lingually.

This report provides a complete implementation blueprint: algorithm with parameters, memory plan, Python code outline, SQLite schema, incremental strategy, search integration, labeling pipeline, content automation hooks, visualization approach, migration plan, and risk assessment.

---

## 1. Recursive Leiden wins on every constraint that matters

**HDBSCAN is infeasible at 245K × 1024 dimensions without dimensionality reduction.** The generic algorithm (the only option at 1024 dims, since ball trees degrade past ~50 dims) needs the full pairwise distance matrix: 245K² × 8 bytes ≈ **480 GB**. Even with UMAP reducing to 50 dimensions first, HDBSCAN still requires ~6–15 GB and takes 1–2 hours—and critically, its condensed tree doesn't give direct control over target cluster counts. You'd need to binary-search over `min_cluster_size` or `cut_distance` to approximate 30–50 / 300–500 / 3000–5000 clusters.

**Faiss K-Means is fast (~5 minutes) but assumes spherical clusters**, which poorly fits semantic embedding spaces where clusters have irregular shapes. It also produces wildly uneven cluster sizes—some Voronoi cells getting 50K points while others get 100.

**Recursive Leiden on a KNN graph is the clear winner** for four reasons:

- **Memory**: ~3–4 GB peak (embeddings + Faiss index + igraph + partitions), easily fits 32 GB
- **Speed**: 15–30 minutes total on M1 Pro
- **Hierarchy control**: Resolution parameter directly tunes cluster count; recursive application guarantees perfect nesting
- **Familiarity**: The user already runs Leiden at 3 resolutions on session-level aggregates in `brain_graph.py`

The pipeline: L2-normalize all 245K embeddings → build a **k=30 KNN graph via Faiss IndexFlatIP** (5–15 minutes, ~2 GB) → convert to igraph → run Leiden recursively. Level 0 uses resolution ~0.005 for 30–50 clusters, then each Level 0 subgraph runs at ~0.05 for Level 1, then each Level 1 subgraph at ~0.5 for Level 2. These resolution values are starting points requiring binary-search tuning on the actual data, but the Leiden call itself takes only 1–3 minutes per level.

| Criterion | Leiden | HDBSCAN (w/ UMAP) | Faiss K-Means | Birch |
|---|---|---|---|---|
| Memory (32 GB Mac) | **~3–4 GB** | ~8–15 GB | ~2 GB | ~10 GB |
| Time | **15–30 min** | 1–2 hrs | 5 min | 30 min |
| 3-level hierarchy | **Resolution control** | Condensed tree (indirect) | Forced nesting | Not natural |
| Cluster quality | **Excellent** | Excellent | Spherical assumption | CF tree limits |
| Incremental updates | **Add nodes/edges** | Must re-fit | Re-assign only | partial_fit |

**Optional RunPod validation** (~$0.50): Run UMAP + HDBSCAN on an RTX 3090 instance (125 GB RAM, ~$0.25/hr) to get an independent clustering for comparison. Use HDBSCAN's condensed tree to identify noise points that Leiden may have forced into clusters. This is a useful quality check, not a requirement.

---

## 2. Memory and compute plan: everything runs locally

The entire pipeline fits comfortably on the M1 Pro with 32 GB unified memory. Before starting, **kill Ollama, any running Claude sessions, and heavy processes** to free ~20–24 GB for the pipeline.

**Step-by-step execution:**

| Step | Operation | Memory | Time | Machine |
|---|---|---|---|---|
| 1 | Extract embeddings from sqlite-vec (batched 10K reads) | ~1 GB (float32 matrix) | 2–3 min | Local |
| 2 | L2-normalize embeddings (in-place) | +0 MB | Seconds | Local |
| 3 | Build Faiss IndexFlatIP, search k=30 | ~2 GB (index + results) | 5–15 min | Local |
| 4 | Construct igraph from KNN edges | ~400 MB | 1–2 min | Local |
| 5 | Recursive Leiden at 3 levels | ~200 MB overhead | 5–10 min | Local |
| 6 | Compute centroids per cluster | ~50 MB | 1 min | Local |
| 7 | Write hierarchy to SQLite | Minimal | 1–2 min | Local |
| 8 | c-TF-IDF labeling (all clusters) | ~500 MB | 2–5 min | Local |
| 9 | LLM labeling (top 2 levels, ~300–550 clusters) | 19 GB (Ollama) | 20–37 min | Local |
| **Peak** | **Steps 1–5 concurrent** | **~3.6 GB** | | |
| **Total** | | | **~45–80 min** | |

Steps 1–7 run without Ollama. After the hierarchy is built, unload the Faiss index and igraph (freeing ~2.5 GB), then start Ollama for LLM labeling. The two phases never compete for memory.

**If RunPod is used** for UMAP + HDBSCAN validation: provision an RTX 3090 community instance ($0.25/hr), upload the ~1 GB embedding matrix via SCP, run UMAP (1024→50, ~30 min) + HDBSCAN (~30 min), download results. Total cost: **~$0.50**, well within the $5–20 budget.

---

## 3. Python code outline with key functions and data flow

```python
# === PHASE 1: EXTRACT & PREPARE ===

def extract_embeddings(db_path: str, batch_size: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """Read all embeddings from sqlite-vec into numpy arrays.
    Returns (embeddings: float32[N, 1024], chunk_ids: int64[N])"""
    conn = apsw.Connection(db_path)
    all_embeddings, all_ids = [], []
    offset = 0
    while True:
        rows = conn.execute(
            "SELECT chunk_id, embedding FROM vec_chunks LIMIT ? OFFSET ?",
            (batch_size, offset)
        ).fetchall()
        if not rows:
            break
        for cid, blob in rows:
            all_ids.append(cid)
            all_embeddings.append(np.frombuffer(blob, dtype=np.float32))
        offset += batch_size
    return np.vstack(all_embeddings), np.array(all_ids, dtype=np.int64)


def build_knn_graph(embeddings: np.ndarray, k: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Build k-NN graph using Faiss. Returns (indices[N, k], distances[N, k])."""
    import faiss
    n, d = embeddings.shape
    faiss.normalize_L2(embeddings)  # in-place for cosine similarity via IP
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    distances, indices = index.search(embeddings, k + 1)
    return indices[:, 1:], distances[:, 1:]  # drop self-match


def knn_to_igraph(indices: np.ndarray, distances: np.ndarray, n: int) -> ig.Graph:
    """Convert KNN results to weighted undirected igraph."""
    import igraph as ig
    k = indices.shape[1]
    sources = np.repeat(np.arange(n), k)
    targets = indices.flatten()
    weights = distances.flatten()
    edges = list(zip(sources.tolist(), targets.tolist()))
    g = ig.Graph(n=n, edges=edges, directed=True)
    g.es['weight'] = weights.tolist()
    g.to_undirected(mode='collapse', combine_edges={'weight': 'max'})
    return g


# === PHASE 2: RECURSIVE LEIDEN CLUSTERING ===

def find_resolution_for_target(graph: ig.Graph, target_clusters: int,
                                lo: float = 0.0001, hi: float = 5.0,
                                tolerance: int = 5, max_iter: int = 20) -> float:
    """Binary search for Leiden resolution parameter yielding target cluster count."""
    import leidenalg
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        part = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition,
            weights='weight', resolution_parameter=mid, seed=42
        )
        n_clusters = len(set(part.membership))
        if abs(n_clusters - target_clusters) <= tolerance:
            return mid
        if n_clusters < target_clusters:
            lo = mid
        else:
            hi = mid
    return mid


def recursive_leiden(graph: ig.Graph, node_indices: np.ndarray,
                     level_targets: list[int]) -> dict:
    """Run Leiden recursively for guaranteed nested hierarchy.
    level_targets: [40, 10, 10] means 40 top clusters, ~10 sub per top, ~10 per mid.
    Returns: {cluster_id: {level, parent_id, member_indices, children}}"""
    import leidenalg
    hierarchy = {}
    cluster_counter = [0]

    def _cluster_level(subgraph, parent_indices, parent_id, level):
        if level >= len(level_targets):
            return
        target = level_targets[level]
        res = find_resolution_for_target(subgraph, target)
        part = leidenalg.find_partition(
            subgraph, leidenalg.RBConfigurationVertexPartition,
            weights='weight', resolution_parameter=res, seed=42
        )
        communities = {}
        for local_idx, comm in enumerate(part.membership):
            communities.setdefault(comm, []).append(local_idx)

        for comm_id, local_members in communities.items():
            cid = cluster_counter[0]
            cluster_counter[0] += 1
            global_members = parent_indices[local_members]
            hierarchy[cid] = {
                'level': level, 'parent_id': parent_id,
                'member_indices': global_members, 'children': []
            }
            if parent_id is not None:
                hierarchy[parent_id]['children'].append(cid)
            # Recurse into subgraph
            if level + 1 < len(level_targets) and len(local_members) > 5:
                sub = subgraph.subgraph(local_members)
                _cluster_level(sub, global_members, cid, level + 1)

    _cluster_level(graph, node_indices, None, 0)
    return hierarchy


# === PHASE 3: CENTROID COMPUTATION ===

def compute_centroids(hierarchy: dict, embeddings: np.ndarray) -> dict[int, np.ndarray]:
    """Compute mean embedding for each cluster."""
    centroids = {}
    for cid, info in hierarchy.items():
        member_embs = embeddings[info['member_indices']]
        centroids[cid] = member_embs.mean(axis=0).astype(np.float32)
    return centroids


# === PHASE 4: INCREMENTAL ASSIGNMENT ===

def assign_new_chunk(embedding: np.ndarray, db_conn,
                     level_order: list[int] = [0, 1, 2]) -> dict[int, int]:
    """Top-down nearest-centroid assignment for a single new chunk."""
    assignments = {}
    parent_id = None
    for level in level_order:
        if parent_id is None:
            query = """SELECT cluster_id, distance FROM vec_cluster_centroids
                       WHERE centroid_embedding MATCH ? AND k=1 AND level=?"""
            row = db_conn.execute(query, [embedding.tobytes(), level]).fetchone()
        else:
            query = """SELECT cluster_id, distance FROM vec_cluster_centroids
                       WHERE centroid_embedding MATCH ? AND k=1
                       AND level=? AND parent_id=?"""
            row = db_conn.execute(query, [embedding.tobytes(), level, parent_id]).fetchone()
        assignments[level] = row[0]
        parent_id = row[0]
    return assignments


def update_centroid_incremental(cluster_id: int, new_embedding: np.ndarray,
                                 db_conn):
    """Update centroid as running mean: new = (old * n + new_emb) / (n + 1)."""
    row = db_conn.execute(
        "SELECT centroid_embedding, chunk_count FROM vec_cluster_centroids WHERE cluster_id=?",
        [cluster_id]
    ).fetchone()
    old_centroid = np.frombuffer(row[0], dtype=np.float32)
    n = row[1]
    updated = ((old_centroid * n) + new_embedding) / (n + 1)
    updated = updated.astype(np.float32)
    db_conn.execute(
        "UPDATE vec_cluster_centroids SET centroid_embedding=?, chunk_count=? WHERE cluster_id=?",
        [updated.tobytes(), n + 1, cluster_id]
    )
```

**Data flow**: `extract_embeddings` → `build_knn_graph` → `knn_to_igraph` → `recursive_leiden` → `compute_centroids` → write to SQLite tables + `vec_cluster_centroids`. New chunks enter via `assign_new_chunk` → `update_centroid_incremental`.

---

## 4. SQLite schema: exact DDL for hierarchy tables

```sql
-- ============================================================
-- CLUSTER HIERARCHY (adjacency list + materialized path)
-- ============================================================
CREATE TABLE clusters (
    id            INTEGER PRIMARY KEY,
    level         INTEGER NOT NULL CHECK(level IN (0, 1, 2)),
    parent_id     INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    path          TEXT NOT NULL,               -- "/0003/0051/0551" (zero-padded)
    label         TEXT,                        -- LLM-generated natural language label
    ctfidf_label  TEXT,                        -- c-TF-IDF keyword label (always present)
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    avg_intra_dist   REAL,                     -- mean cosine dist to centroid
    silhouette_score REAL,                     -- cluster quality metric
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_clusters_level ON clusters(level);
CREATE INDEX idx_clusters_parent ON clusters(parent_id);
CREATE INDEX idx_clusters_path ON clusters(path);

-- ============================================================
-- CHUNK-TO-CLUSTER MAPPING (normalized, survives re-clustering)
-- ============================================================
CREATE TABLE chunk_clusters (
    chunk_id         INTEGER NOT NULL,          -- FK to chunks table
    cluster_id       INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    level            INTEGER NOT NULL,           -- denormalized for speed
    dist_to_centroid REAL,
    assignment_method TEXT DEFAULT 'initial',    -- 'initial' | 'incremental' | 'recluster'
    assigned_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (chunk_id, level)
);
CREATE INDEX idx_cc_cluster ON chunk_clusters(cluster_id);
CREATE INDEX idx_cc_chunk ON chunk_clusters(chunk_id);

-- ============================================================
-- CLUSTER CENTROIDS (sqlite-vec, separate from chunk embeddings)
-- ============================================================
CREATE VIRTUAL TABLE vec_cluster_centroids USING vec0(
    cluster_id         INTEGER PRIMARY KEY,
    centroid_embedding float[1024],
    level              INTEGER partition key,    -- fast level-filtered KNN
    parent_id          INTEGER,                  -- metadata: filter within parent
    chunk_count        INTEGER,                  -- metadata: cluster size
    +label             TEXT,                     -- auxiliary: human-readable
    +path              TEXT                      -- auxiliary: materialized path
);

-- ============================================================
-- CLUSTERING AUDIT LOG
-- ============================================================
CREATE TABLE clustering_runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at         TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at       TEXT,
    algorithm          TEXT NOT NULL,
    parameters_json    TEXT,
    total_chunks       INTEGER,
    num_clusters_l0    INTEGER,
    num_clusters_l1    INTEGER,
    num_clusters_l2    INTEGER,
    avg_silhouette     REAL,
    status             TEXT DEFAULT 'running'
);
```

**Design rationale**: The `chunk_clusters` mapping table is preferred over adding `cluster_l0/l1/l2` columns to the chunks table because it cleanly separates concerns—the chunks table stays immutable during re-clustering, and you can atomically swap cluster assignments in a transaction. The materialized `path` column (zero-padded: `"/0003/0051/0551"`) enables fast subtree queries (`WHERE path LIKE '/0003/%'`) while `parent_id` handles direct parent lookups. The `vec_cluster_centroids` table uses **`level` as a partition key**, meaning `WHERE level = 0` only searches the ~40 centroids at Level 0—instant even by brute force.

---

## 5. Incremental update strategy

**Daily new chunk assignment** uses top-down nearest-centroid routing. For each new chunk: find nearest Level 0 centroid → find nearest Level 1 centroid within that parent → find nearest Level 2 centroid within that parent. This requires **5,550 distance computations total** (50 + 500 + 5000), which takes milliseconds. After assignment, update the leaf centroid incrementally: `new_centroid = (old × n + embedding) / (n + 1)`.

**Quality safeguard**: Also run KNN voting (find 10 nearest existing chunks, check their cluster assignments). If KNN vote disagrees with centroid assignment, flag the chunk in a `staging` table for manual review or next re-clustering batch.

**Cluster health monitoring** runs as a daily cron job checking three conditions:

- **Size split**: Leaf cluster exceeds **3× the average leaf size** (~147 chunks at initial scale) → run 2-means on that cluster
- **Variance split**: Cluster's average intra-distance exceeds **2× the global average** → BIC test for 1 vs 2 clusters
- **Merge detection**: Two sibling clusters whose centroids are closer than cosine distance 0.1 → merge and relabel

**Re-clustering schedule at ~750 chunks/day growth**:

| Action | Frequency | Trigger |
|---|---|---|
| Centroid assignment + update | Per-chunk (real-time) | Every new chunk ingestion |
| Split/merge health check | Daily | Cron job |
| Local re-clustering (affected clusters) | Weekly | >5% of leaf clusters flagged |
| Full hierarchical re-clustering | Every 4–6 weeks | Silhouette score drops >10% from baseline |

At 750 chunks/day, the dataset grows ~3% per week. Empirically, incremental centroid assignment degrades gracefully for 4–6 weeks before requiring a full rebuild. Track a **sampled silhouette score** (5K random chunks, weekly) as the quality metric that triggers re-clustering.

---

## 6. Search integration: concrete changes to the existing pipeline

The current BM25 (FTS5) + semantic (sqlite-vec cosine) → rerank pipeline should adopt a **search-first with cluster expansion** strategy. This preserves existing search quality while adding cluster context.

**Modified search flow:**

1. **Existing search** — BM25 + semantic retrieval produces top-K candidates (unchanged)
2. **Cluster annotation** — For each result, look up its cluster path via `chunk_clusters` JOIN `clusters`. Attach the path as metadata: `"deployment / railway / dockerfile-optimization"`
3. **Sibling expansion** — For the top-3 results, retrieve **5 highest-cohesion siblings** from the same leaf cluster (chunks closest to the cluster centroid, excluding already-returned results). This adds semantically related chunks the user didn't directly query for
4. **Cluster-boosted reranking** — When multiple results share a leaf cluster, boost their collective relevance (cluster coherence as a ranking signal). Replace the static importance score with a dynamic **cluster relevance score**: `0.3 × log(cluster_size) + 0.3 × cohesion + 0.2 × (1 - dist_to_centroid) + 0.2 × freshness`

**Cluster-first search** (find nearest cluster → search within) is useful as an **alternative mode for browsing**, not as the default search path. The cluster hypothesis holds well for semantic embeddings, but boundary documents between clusters get missed. Offer it as `?mode=browse` for when users want to explore a topic rather than find a specific answer.

**RAPTOR-style enhancement** (future phase): Generate LLM summaries for each cluster and store them as searchable nodes. Queries can then match leaf chunks for factual retrieval OR cluster summaries for thematic retrieval—different abstraction levels serve different query types. The RAPTOR paper showed **20% absolute accuracy improvement** on the QuALITY benchmark using this approach.

```sql
-- Cluster annotation query (fast, indexed)
SELECT c.path, c.label, c.chunk_count
FROM chunk_clusters cc
JOIN clusters c ON cc.cluster_id = c.id
WHERE cc.chunk_id = ? AND cc.level = 2;

-- Sibling expansion query
SELECT cc2.chunk_id, cc2.dist_to_centroid
FROM chunk_clusters cc1
JOIN chunk_clusters cc2 ON cc1.cluster_id = cc2.cluster_id
WHERE cc1.chunk_id = ? AND cc1.level = 2
  AND cc2.chunk_id != ?
ORDER BY cc2.dist_to_centroid ASC
LIMIT 5;
```

---

## 7. Labeling 5,000 clusters efficiently with a tiered hybrid approach

A pure LLM approach for all 5,000 clusters takes 5.5 hours at 4 seconds/call—feasible overnight but unnecessarily expensive for leaf clusters. The optimal strategy is **tiered**:

**Tier 1: c-TF-IDF keywords for all 5,000 clusters** (2–5 minutes, no LLM). Treat each cluster's chunks as a single document, compute class-based TF-IDF, extract top-5 keywords. Use BERTopic's enhancements: `bm25_weighting=True` and `reduce_frequent_words=True` to improve keyword quality. This produces labels like `"docker_railway_deploy_config_dockerfile"`.

**Tier 2: LLM hybrid labeling for Level 0 + Level 1** (~300–550 clusters, 20–37 minutes). For each cluster, pass the c-TF-IDF keywords + 5 representative chunks (nearest to centroid) + any existing enrichment tags to GLM-4.7-Flash with this prompt:

```
I have a topic described by these keywords: {ctfidf_keywords}
Enrichment tags found in this cluster: {enrichment_tags}

Representative content samples:
{5_closest_to_centroid_chunks, truncated_to_200_chars_each}

Provide a short descriptive label (2-5 words) for this topic.
topic:
```

Parse everything after `topic:` as the label. Research shows that **including key terms plus representative snippets produces the best labels**—better than either alone, and better than human naming in blind evaluations.

**Labels should be hierarchical paths**: `"Deployment / Railway / Dockerfile Config"`. Level 0 labels are the broad domain, Level 1 the sub-area, Level 2 uses c-TF-IDF keywords formatted as a short phrase. This supports breadcrumb navigation in the UI and filter-by-level in search.

**Parallelizing Ollama**: Set `OLLAMA_NUM_PARALLEL=2` to halve labeling time. Each parallel slot increases VRAM usage proportionally, but with a single 19 GB model and no other workloads, 2 slots should fit in 32 GB. This brings Level 0+1 labeling down to **~10–18 minutes**. For the full 5,000 clusters, overnight batch at 2 parallel slots takes ~2.75 hours.

**Existing enrichment data** (tags, intents) provides a powerful quality signal. Feed cluster-level tag distributions into the LLM prompt for better labels. Additionally, if any chunks already have manually assigned topic labels, use BERTopic's semi-supervised mode (pass labels as the `y` parameter) to steer the UMAP dimensionality reduction toward known categories.

---

## 8. Content automation: MCP tools, n8n hooks, and topic suggestions

**Five MCP tools** expose the cluster hierarchy to AI agents in the content pipeline:

- **`get_topic_clusters(level, parent_id?, min_chunks?)`** — Browse hierarchy. Level 0 returns ~40 top topics with labels and chunk counts. Pass `parent_id` to drill down.
- **`get_cluster_details(cluster_id, include_chunks?)`** — Full cluster info: label, children, cohesion score, representative chunks. The content pipeline's Claude agent uses this to understand what a topic actually covers.
- **`find_relevant_clusters(query, limit?)`** — Semantic search over cluster centroids via `vec_cluster_centroids`. Returns ranked clusters. Powers "find me everything related to Railway deployment."
- **`get_expert_topics(min_chunks?, sort_by?)`** — Returns clusters where the user has demonstrable expertise, sorted by a composite score.
- **`suggest_content_topics(type?, limit?)`** — Analyzes the knowledge graph to surface content opportunities across four categories.

**Content suggestion categories** the `suggest_content_topics` tool should implement:

- **Authority content**: Clusters with >50 chunks, high cohesion (silhouette >0.5), multiple source types → "You have deep expertise in TypeScript monorepo patterns—write a technical guide"
- **Timely content**: Clusters with spiking chunk velocity (>5× their 30-day average) → "Railway deployment activity surged this week—share your setup"
- **Unique angles**: Chunks bridging two distant clusters (high betweenness centrality) → "You've connected Telegram bots with real-estate scraping—that's a unique perspective"
- **Content gaps**: Clusters with high query frequency but low chunk count → topics the user searches for but hasn't documented

**Expertise scoring formula**:
```
expertise = 0.30 × log(chunk_count)/log(max_count)  // depth
          + 0.20 × silhouette_score                   // focus
          + 0.20 × source_diversity/max_diversity      // breadth
          + 0.15 × temporal_span/365                   // sustained
          + 0.15 × exp(-0.01 × days_since_last)        // fresh
```

**n8n integration**: Wrap the cluster SQLite queries in a lightweight **FastAPI service** (5 endpoints matching the MCP tools above). n8n's HTTP Request node calls these endpoints. Example workflow: Schedule Trigger (Monday 9am) → HTTP Request (`GET /api/suggest-content?type=all`) → Code Node (format suggestions) → Slack Node (post to #content-ideas). For Remotion/ComfyUI pipeline triggers, n8n Webhook nodes receive cluster data and fan out to the appropriate rendering pipeline based on content type.

---

## 9. Visualization: Plotly treemap plus Three.js hull overlays

**Start with a Plotly.js zoomable treemap** — it handles 5,000 nodes natively, requires ~50 lines of code, and provides built-in click-to-zoom with a pathbar breadcrumb. The treemap is the highest insight-per-effort visualization: it fills 100% of screen space, rectangle sizes immediately communicate cluster importance, and drilling down reveals the topic hierarchy intuitively.

```javascript
// Plotly treemap data: three flat arrays from your clusters table
Plotly.newPlot('treemap', [{
  type: 'treemap',
  labels: clusterLabels,      // ["All", "Deployment", "Railway", ...]
  parents: clusterParents,    // ["", "All", "Deployment", ...]
  values: chunkCounts,        // [245817, 5000, 1200, ...]
  textinfo: 'label+value',
  branchvalues: 'total'
}], { margin: { t: 30, l: 0, r: 0, b: 0 } });
```

**Add a sunburst chart as an alternative view** — same data structure, different visual. Sunbursts work especially well for exactly 3 levels: the inner ring shows broad domains, middle ring shows sub-topics, outer ring shows leaf clusters. Toggle between treemap and sunburst with a single button.

**Integrate with the existing Three.js 3D brain graph** by adding semi-transparent convex hull overlays per cluster. Use Three.js's built-in `ConvexGeometry` to compute hulls from member chunk 3D positions, then render with `MeshBasicMaterial({ transparent: true, opacity: 0.15 })`. For performance, only render hulls at the current zoom level—Level 0 hulls (~40 meshes) at default zoom, Level 1 hulls (~500) when zoomed into a cluster. UMAP 3D coordinates from the existing brain graph can be reused directly; compute cluster-level positions as the mean of member coordinates.

**Linked views**: Clicking a cluster in the treemap sidebar flies the 3D camera to that cluster's centroid and renders its hull. Hovering a chunk in the 3D graph highlights its cluster path in the treemap. This dual-view approach gives structural overview (treemap) plus spatial exploration (3D graph).

---

## 10. Migration plan: flat enrichment to hierarchical, step by step

**Phase 1 — Hebrew re-embedding (Day 1, ~2–4 hours)**

bge-large-en-v1.5 produces **near-random vectors for Hebrew** because its BERT tokenizer has virtually zero Hebrew tokens—characters decompose to `[UNK]` with no learned semantics. The 16K WhatsApp Hebrew embeddings are effectively noise in the current vector space.

**The practical fix**: Re-embed at minimum all 16K WhatsApp chunks with **BGE-M3** (1024 dims, 100+ languages, MIRACL SOTA). On a RunPod T4 instance ($0.20/hr), this takes ~10 minutes. Ideally, re-embed the full 245K with BGE-M3 in a single RunPod session (~1–2 hours on an A100, ~$2). BGE-M3 matches bge-large-en-v1.5 on English while adding real Hebrew support and handling short-to-long text via its multi-granularity architecture.

**Critical**: Do not mix embeddings from different models in the same vector index. Either re-embed everything with BGE-M3, or maintain two separate `vec0` tables and cluster each independently, linking cross-source via temporal/entity overlap.

**Phase 2 — Initial clustering (Day 1–2, ~1 hour)**

1. Kill Ollama and heavy processes
2. Run `extract_embeddings()` → `build_knn_graph()` → `knn_to_igraph()` → `recursive_leiden()`
3. Run `compute_centroids()`
4. Execute the DDL to create `clusters`, `chunk_clusters`, `vec_cluster_centroids` tables
5. Populate all three tables from the clustering results
6. Verify: `SELECT level, COUNT(*) FROM clusters GROUP BY level` should return ~40/~400/~4000

**Phase 3 — Labeling (Day 2, ~1 hour)**

1. Generate c-TF-IDF labels for all clusters (BERTopic vectorizer, 2–5 minutes)
2. Start Ollama with GLM-4.7-Flash, `OLLAMA_NUM_PARALLEL=2`
3. LLM-label Level 0 + Level 1 clusters (~300–550 calls, ~15–30 minutes)
4. Write all labels to `clusters.label` and `clusters.ctfidf_label`

**Phase 4 — Search integration (Day 3)**

1. Modify the search reranker to JOIN `chunk_clusters` and annotate results with cluster paths
2. Add sibling expansion: for top-3 results, query 5 siblings from the same leaf cluster
3. Replace static importance score with cluster-derived relevance score

**Phase 5 — Visualization + MCP tools (Week 2)**

1. Add Plotly treemap to Ops Dashboard (generates from `clusters` table)
2. Implement the 5 MCP tools as FastAPI endpoints
3. Add convex hull overlays to existing Three.js brain graph

**Phase 6 — Content automation (Week 3+)**

1. Build expertise scoring query
2. Implement content suggestion categories
3. Create n8n workflows connecting to FastAPI endpoints
4. Wire into Remotion/ComfyUI pipeline selection logic

---

## 11. Risk assessment and what could go wrong

**Hebrew embedding quality is the highest-impact risk.** bge-large-en-v1.5 fundamentally cannot process Hebrew—this isn't a "degraded quality" situation, it's near-random output. If Hebrew WhatsApp chunks aren't re-embedded with a multilingual model, they'll cluster randomly, corrupt any cross-source topic detection, and make the 16K WhatsApp chunks essentially invisible to the knowledge graph. **Mitigation**: Re-embed with BGE-M3. If full re-embedding is too costly, at minimum re-embed WhatsApp chunks and keep them in a separate vector index.

**Short WhatsApp messages (5–20 words) produce noisier embeddings** than long code blocks (200+ tokens). Even with BGE-M3, expect WhatsApp clusters to be looser with wider variance. **Mitigation**: Use HDBSCAN-style `min_cluster_size` thresholds during Leiden subclustering of WhatsApp-heavy subgraphs. Accept that some WhatsApp messages will land in catch-all clusters—this is inherent to short-text clustering, not a system failure.

**Leiden resolution tuning is empirical, not deterministic.** The resolution values (0.005 / 0.05 / 0.5) are starting estimates. The actual values for your data's graph structure could differ by 10×. **Mitigation**: The binary-search `find_resolution_for_target()` function automates this, but verify cluster quality manually for the first run. Inspect the 5 largest and 5 smallest clusters at each level.

**Centroid drift during incremental updates** is a slow-acting risk. The running-mean centroid update (`(old × n + new) / (n + 1)`) biases the centroid toward the temporal distribution of new chunks, which may not represent the cluster's semantic center. After 4–6 weeks of 750 chunks/day, cluster boundaries may no longer reflect the actual data distribution. **Mitigation**: Weekly silhouette sampling detects this early. Re-compute true centroids (mean of all member embeddings) monthly rather than relying solely on the running mean.

**sqlite-vec partition key filtering** in v0.1.6 works for the centroid table's ~5,500 rows but may behave unexpectedly for complex compound filters (`level AND parent_id`). The partition key creates separate internal indices per level, but metadata filtering happens post-KNN-search. With only ~50 centroids per Level 0 partition, this is fine—but test the `AND parent_id = ?` filter carefully. **Mitigation**: If compound filtering fails, fall back to fetching top-K per level and filtering in Python.

**228K code chunks dominate the embedding space.** With 93% of chunks being code conversations, the Level 0 clusters will likely all be code-related, with WhatsApp/YouTube squeezed into 1–2 clusters. **Mitigation**: Consider source-weighted sampling during KNN graph construction (e.g., upweight WhatsApp edges) or run a preliminary source-aware split before unified clustering. Alternatively, accept the imbalance and use the existing `content_type` metadata to surface non-code clusters in the UI regardless of their relative size.

**SSD space is tight** at 28 GB free. The clustering pipeline temporarily needs ~1 GB for the embedding matrix + ~200 MB for intermediate arrays. The new SQLite tables add ~50 MB (clusters) + ~25 MB (centroid vec0) + ~30 MB (chunk_clusters mapping). Total new storage: **~100–150 MB permanent**. This is negligible, but if a full BGE-M3 re-embedding is done, the new vec0 table replaces the old one (same 1024 dims, ~1 GB). **No net SSD impact** from re-embedding.

**Code-switching in WhatsApp messages** (Hebrew words mixed with English technical terms like "deploy," "push," "build") is actually a partial advantage: the English terms provide semantic signal even with the English-only model. But for pure-Hebrew messages (social, personal), the signal is zero. BGE-M3 handles code-switching natively via its XLM-RoBERTa backbone, which was trained on multilingual web data including mixed-language text.