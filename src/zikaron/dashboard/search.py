"""Hybrid search engine combining BM25 and semantic search."""

import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer

from ..pipeline.embed import embed_query


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
    
    def fit_collection(self, collection: chromadb.Collection):
        """Fit search engine on ChromaDB collection."""
        try:
            # Get all documents from collection
            all_data = collection.get(
                limit=10000,  # Reasonable limit for Phase 1
                include=["documents", "metadatas", "ids"]
            )
            
            self.documents = all_data.get("documents", [])
            self.metadatas = all_data.get("metadatas", [])
            self.ids = all_data.get("ids", [])
            
            if self.documents:
                self.bm25.fit(self.documents)
                self.is_fitted = True
            
        except Exception as e:
            print(f"Error fitting search engine: {e}")
            self.is_fitted = False
    
    def search(
        self,
        collection: chromadb.Collection,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> Dict[str, Any]:
        """
        Hybrid search using RRF (Reciprocal Rank Fusion).
        
        Args:
            collection: ChromaDB collection
            query: Search query
            n_results: Number of results to return
            where: Metadata filters
            alpha: Weight for combining scores (0.5 = equal weight)
        
        Returns:
            Search results in ChromaDB format
        """
        if not self.is_fitted:
            self.fit_collection(collection)
        
        if not self.is_fitted or not self.documents:
            # Fallback to semantic search only
            return self._semantic_search_only(collection, query, n_results, where)
        
        try:
            # 1. BM25 keyword search
            bm25_results = self.bm25.search(query, n_results * 2)  # Get more for fusion
            bm25_scores = {idx: score for idx, score in bm25_results}
            
            # 2. Semantic search
            query_embedding = embed_query(query)
            semantic_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # 3. Reciprocal Rank Fusion (RRF)
            rrf_scores = defaultdict(float)
            k = 60  # RRF parameter
            
            # Add BM25 scores
            for rank, (doc_idx, score) in enumerate(bm25_results):
                if doc_idx < len(self.ids):
                    doc_id = self.ids[doc_idx]
                    rrf_scores[doc_id] += alpha / (k + rank + 1)
            
            # Add semantic scores (convert distances to similarity)
            semantic_docs = semantic_results.get("documents", [[]])[0]
            semantic_metas = semantic_results.get("metadatas", [[]])[0]
            semantic_distances = semantic_results.get("distances", [[]])[0]
            
            for rank, (doc, meta, distance) in enumerate(zip(semantic_docs, semantic_metas, semantic_distances)):
                # Find document ID from metadata or content
                doc_id = None
                for i, (stored_doc, stored_meta) in enumerate(zip(self.documents, self.metadatas)):
                    if stored_doc == doc and stored_meta == meta:
                        doc_id = self.ids[i]
                        break
                
                if doc_id:
                    rrf_scores[doc_id] += (1 - alpha) / (k + rank + 1)
            
            # 4. Sort by combined RRF score and return top results
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
            
            # 5. Build result structure
            result_docs = []
            result_metas = []
            result_distances = []
            
            for doc_id, rrf_score in sorted_results:
                # Find document by ID
                for i, stored_id in enumerate(self.ids):
                    if stored_id == doc_id:
                        result_docs.append(self.documents[i])
                        result_metas.append(self.metadatas[i])
                        result_distances.append(1.0 - rrf_score)  # Convert score to distance-like
                        break
            
            return {
                "documents": [result_docs],
                "metadatas": [result_metas],
                "distances": [result_distances]
            }
            
        except Exception as e:
            print(f"Hybrid search error: {e}")
            # Fallback to semantic search
            return self._semantic_search_only(collection, query, n_results, where)
    
    def _semantic_search_only(
        self,
        collection: chromadb.Collection,
        query: str,
        n_results: int,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fallback to semantic search only."""
        try:
            query_embedding = embed_query(query)
            return collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"Semantic search error: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
