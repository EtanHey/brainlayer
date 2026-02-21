"""Hybrid search engine combining BM25 and semantic search."""

import logging
import math
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..vector_store import VectorStore


class BM25:
    """Simple BM25 implementation for keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = []
        self.idf = {}
        self.vocab = set()

    def fit(self, documents: List[str]):
        """Fit BM25 on document corpus."""
        self.documents = documents
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if documents else 0

        # Calculate document frequencies
        self.doc_freqs = []
        vocab_counter = Counter()

        for doc in documents:
            words = doc.lower().split()
            word_counts = Counter(words)
            self.doc_freqs.append(word_counts)
            vocab_counter.update(set(words))

        self.vocab = set(vocab_counter.keys())

        # Calculate IDF
        n_docs = len(documents)
        for word in self.vocab:
            df = sum(1 for doc_freq in self.doc_freqs if word in doc_freq)
            self.idf[word] = math.log((n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for query against document."""
        if doc_idx >= len(self.doc_freqs):
            return 0.0

        # Guard against divide-by-zero
        if self.avg_doc_length == 0:
            return 0.0

        query_words = query.lower().split()
        doc_freq = self.doc_freqs[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        score = 0.0
        for word in query_words:
            if word in doc_freq:
                tf = doc_freq[word]
                idf = self.idf.get(word, 0)

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

                if denominator > 0:
                    score += idf * (numerator / denominator)

        return score

    def search(self, query: str, n_results: int = 10) -> List[Tuple[int, float]]:
        """Search documents and return (doc_idx, score) pairs."""
        scores = []
        for i in range(len(self.documents)):
            score = self.score(query, i)
            if score > 0:
                scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_results]


class HybridSearchEngine:
    """Hybrid search combining BM25 keyword search with semantic search."""

    def __init__(self):
        self.bm25 = BM25()
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.is_fitted = False
        self._embedding_model = None

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Lazy load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel()
        return self._embedding_model

    def fit_store(self, vector_store: "VectorStore"):
        """Fit search engine on VectorStore (sqlite-vec)."""
        try:
            # Get sample of documents for BM25 fitting
            # Note: This is a simplified approach - for large DBs, sample instead
            all_data = vector_store.get_all_chunks(limit=10000)

            self.documents = [d["content"] for d in all_data]
            self.metadatas = [d["metadata"] for d in all_data]
            self.ids = [d["id"] for d in all_data]

            if self.documents:
                self.bm25.fit(self.documents)
                self.is_fitted = True

        except Exception as e:
            logger.warning("Error fitting search engine: %s", e)
            self.is_fitted = False

    def search(
        self,
        vector_store: "VectorStore",
        query: str,
        n_results: int = 10,
        project_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Hybrid search using RRF (Reciprocal Rank Fusion).

        Args:
            vector_store: VectorStore instance
            query: Search query
            n_results: Number of results to return
            project_filter: Filter by project name
            content_type_filter: Filter by content type
            alpha: Weight for combining scores (0.5 = equal weight)

        Returns:
            Search results dict with documents, metadatas, distances
        """
        if vector_store is None:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        if not self.is_fitted:
            self.fit_store(vector_store)

        if not self.is_fitted or not self.documents:
            # Fallback to semantic search only
            return self._semantic_search_only(vector_store, query, n_results, project_filter, content_type_filter)

        try:
            # 1. BM25 keyword search
            bm25_results = self.bm25.search(query, n_results * 2)

            # 2. Semantic search via VectorStore
            query_embedding = self.embedding_model.embed_query(query)
            semantic_results = vector_store.search(
                query_embedding=query_embedding,
                n_results=n_results * 2,
                project_filter=project_filter,
                content_type_filter=content_type_filter,
            )

            # 3. Reciprocal Rank Fusion (RRF)
            rrf_scores = defaultdict(float)
            k = 60  # RRF parameter

            # Add BM25 scores
            for rank, (doc_idx, score) in enumerate(bm25_results):
                if doc_idx < len(self.ids):
                    doc_id = self.ids[doc_idx]
                    rrf_scores[doc_id] += alpha / (k + rank + 1)

            # Add semantic scores
            semantic_docs = semantic_results.get("documents", [[]])[0]
            semantic_metas = semantic_results.get("metadatas", [[]])[0]
            semantic_distances = semantic_results.get("distances", [[]])[0]

            for rank, (doc, meta, distance) in enumerate(zip(semantic_docs, semantic_metas, semantic_distances)):
                # Use metadata to find ID
                doc_id = meta.get("source_file", "") + ":" + str(rank)
                rrf_scores[doc_id] += (1 - alpha) / (k + rank + 1)

            # 4. Sort by combined RRF score and return top results
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]

            # 5. Build result structure from semantic results (more reliable)
            return {
                "documents": [semantic_docs[:n_results]],
                "metadatas": [semantic_metas[:n_results]],
                "distances": [semantic_distances[:n_results]],
            }

        except Exception as e:
            logger.warning("Hybrid search error: %s", e)
            return self._semantic_search_only(vector_store, query, n_results, project_filter, content_type_filter)

    def _semantic_search_only(
        self,
        vector_store: "VectorStore",
        query: str,
        n_results: int,
        project_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fallback to semantic search only."""
        try:
            query_embedding = self.embedding_model.embed_query(query)
            return vector_store.search(
                query_embedding=query_embedding,
                n_results=n_results,
                project_filter=project_filter,
                content_type_filter=content_type_filter,
            )
        except Exception as e:
            logger.warning("Semantic search error: %s", e)
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
