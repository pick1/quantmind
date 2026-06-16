"""Vector search — ChromaDB cosine similarity retrieval of top-k candidates."""

import logging
from typing import Optional

from quantmind.config import TOP_K_CANDIDATES

logger = logging.getLogger("quantmind.retrieval.searcher")


class Searcher:
    """Retrieves top-k document chunks from ChromaDB by vector similarity."""

    def __init__(self, vec_store: Optional["VectorStore"] = None):
        from quantmind.store.chroma import VectorStore
        self._vec_store = vec_store or VectorStore()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_CANDIDATES,
    ) -> list[dict]:
        """Search ChromaDB for the top-k most similar chunks.

        Args:
            query_embedding: Vector embedding of the user's query.
            top_k: Number of candidates to retrieve (default: 20).

        Returns:
            List of dicts with keys: id, document, metadata, distance, source_type, title.
        """
        result = self._vec_store.search(
            query_embedding=query_embedding,
            n_results=top_k,
        )

        hits = []
        if not result["ids"] or not result["ids"][0]:
            logger.info("No results found in ChromaDB")
            return []

        for i in range(len(result["ids"][0])):
            meta = result["metadatas"][0][i] if result["metadatas"] else {}
            hits.append({
                "id": result["ids"][0][i],
                "document": result["documents"][0][i],
                "metadata": meta,
                "distance": result["distances"][0][i] if result["distances"] else 0.0,
                "source_type": meta.get("source_type", "unknown"),
                "title": meta.get("title", "Untitled"),
            })

        # Sort by distance ascending (lower = more similar)
        hits.sort(key=lambda h: h["distance"])

        logger.debug("Search returned %d candidates", len(hits))
        return hits
