"""Cross-encoder reranking — re-scores top candidates using ms-marco-MiniLM-L-6-v2.

Runs on CPU. For 20 candidates, benchmarks at <1s on modern i9.
The cross-encoder model is loaded lazily on first use.
"""

import logging
from typing import Optional

from quantmind.config import TOP_K_RERANK

logger = logging.getLogger("quantmind.retrieval.reranker")

# Lazy-loaded cross-encoder model
_cross_encoder_model = None


def _get_cross_encoder():
    """Load the cross-encoder model on first call."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder_model = CrossEncoder(
            "ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )
        logger.info(
            "Cross-encoder model loaded (ms-marco-MiniLM-L-6-v2, CPU)"
        )
    return _cross_encoder_model


class Reranker:
    """Reranks retrieved candidates using a cross-encoder for relevance."""

    def __init__(self, top_k: int = TOP_K_RERANK):
        self.top_k = top_k

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidates by query-chunk relevance.

        Args:
            query: The original user query.
            candidates: List of candidate dicts from Searcher.

        Returns:
            Top-k reranked candidates, each with an added 'relevance_score'.
        """
        if not candidates:
            return []

        model = _get_cross_encoder()

        # Prepare query-document pairs
        pairs = [(query, c["document"]) for c in candidates]

        try:
            scores = model.predict(pairs)
        except Exception:
            logger.exception("Cross-encoder reranking failed — falling back to vector distance")
            # Fallback: use the negative of ChromaDB distance as score
            max_dist = max(c["distance"] for c in candidates) or 1.0
            scores = [1.0 - (c["distance"] / max_dist) for c in candidates]

        # Attach scores and sort
        for i, candidate in enumerate(candidates):
            candidate["relevance_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda c: c["relevance_score"], reverse=True)

        top_k = reranked[: self.top_k]

        logger.debug(
            "Reranked %d candidates → top-%d (best score: %.4f)",
            len(candidates),
            len(top_k),
            top_k[0]["relevance_score"] if top_k else 0,
        )
        return top_k
