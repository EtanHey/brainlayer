"""Cluster-based sampling for representative message selection."""

from typing import List, Optional, Protocol, TypeVar

T = TypeVar("T")


class HasText(Protocol):
    """Protocol for objects with .text attribute."""
    text: str


def cluster_sample_messages(
    messages: List[T],  # Must have .text attribute
    embeddings: list[list[float]],
    k: Optional[int] = None,
    samples_per_cluster: int = 3,
    max_total: int = 250,
) -> List[T]:
    """
    Sample representative messages via K-means clustering on embeddings.

    Selects messages nearest to each cluster centroid for diversity.

    Args:
        messages: Messages (same order as embeddings)
        embeddings: Corresponding embedding vectors
        k: Number of clusters (default: min(8, len(messages)//10))
        samples_per_cluster: Messages to pick per cluster
        max_total: Cap total samples

    Returns:
        Sampled items (messages or text wrappers) for LLM prompt.
        Input items must have .text attribute.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    n = len(messages)
    if n == 0 or n != len(embeddings):
        return messages[:max_total] if messages else []

    if k is None:
        k = min(8, max(2, n // 10))
    k = min(k, n)

    X = np.array(embeddings, dtype=np.float32)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # For each cluster, get indices of points nearest to centroid
    sampled_indices = []
    for c in range(k):
        mask = labels == c
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        # Distances to centroid
        dists = np.linalg.norm(X[indices] - centroids[c], axis=1)
        # Take closest samples_per_cluster
        closest = indices[np.argsort(dists)[:samples_per_cluster]]
        sampled_indices.extend(closest.tolist())

    # Deduplicate and cap
    seen = set()
    result = []
    for i in sampled_indices:
        if i not in seen and len(result) < max_total:
            seen.add(i)
            result.append(messages[i])
    return result
