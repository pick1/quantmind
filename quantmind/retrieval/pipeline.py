"""Retrieval pipeline — orchestrates embed → search → rerank → synthesise.

End-to-end: user query → vector → ChromaDB search → cross-encoder rerank → LLM answer.
"""

import logging
from typing import Optional

from quantmind.llm_client import LLMClient, ProviderUnavailableError
from quantmind.retrieval.embedder import QueryEmbedder
from quantmind.retrieval.searcher import Searcher
from quantmind.retrieval.reranker import Reranker
from quantmind.retrieval.synthesizer import Synthesizer

logger = logging.getLogger("quantmind.retrieval.pipeline")


class RetrievalPipeline:
    """End-to-end retrieval: query → embedding → search → rerank → synthesis."""

    def __init__(
        self,
        embedder: Optional[QueryEmbedder] = None,
        searcher: Optional[Searcher] = None,
        reranker: Optional[Reranker] = None,
        synthesizer: Optional[Synthesizer] = None,
    ):
        self.embedder = embedder or QueryEmbedder()
        self.searcher = searcher or Searcher()
        self.reranker = reranker or Reranker()
        self.synthesizer = synthesizer or Synthesizer()

    def query(
        self,
        question: str,
        top_k_candidates: int = 20,
        top_k_rerank: int = 5,
    ) -> dict:
        """Execute a full retrieval-augmented generation cycle.

        Args:
            question: Natural language question.
            top_k_candidates: Candidates from vector search (default: 20).
            top_k_rerank: Candidates after reranking (default: 5).

        Returns:
            dict with keys: answer, citations, has_citations, unverified,
            used_chunks, confidence, context (the top chunks used).
        """
        logger.info("RetrievalPipeline.query: %s", question[:100])

        # 1. Embed
        try:
            query_embedding = self.embedder.embed(question)
        except ProviderUnavailableError:
            return {
                "answer": "Embedding backend unavailable. Ensure Ollama is running or OLLAMA_HOST is set correctly.",
                "citations": [],
                "has_citations": False,
                "unverified": True,
                "used_chunks": [],
                "confidence": "low",
                "context": [],
            }
        except Exception:
            logger.exception("Embedding failed")
            return {
                "answer": "Failed to embed query. Check the embedding backend.",
                "citations": [],
                "has_citations": False,
                "unverified": True,
                "used_chunks": [],
                "confidence": "low",
                "context": [],
            }

        # 2. Search
        try:
            candidates = self.searcher.search(query_embedding, top_k=top_k_candidates)
        except Exception:
            logger.exception("Vector search failed")
            return {
                "answer": "Vector search failed. Check ChromaDB status.",
                "citations": [],
                "has_citations": False,
                "unverified": True,
                "used_chunks": [],
                "confidence": "low",
                "context": [],
            }

        if not candidates:
            return {
                "answer": "No relevant documents found. Try ingesting more documents first.",
                "citations": [],
                "has_citations": False,
                "unverified": True,
                "used_chunks": [],
                "confidence": "low",
                "context": [],
            }

        # 3. Rerank
        try:
            top_chunks = self.reranker.rerank(question, candidates)
        except Exception:
            logger.warning("Reranking failed — using top candidates by vector distance")
            top_chunks = candidates[:top_k_rerank]
            for c in top_chunks:
                c["relevance_score"] = 1.0 - c.get("distance", 0)

        # 4. Synthesise
        try:
            result = self.synthesizer.synthesize(question, top_chunks)
        except Exception:
            logger.exception("Synthesis failed")
            return {
                "answer": "Failed to generate answer. The LLM provider may be unavailable.",
                "citations": [],
                "has_citations": False,
                "unverified": True,
                "used_chunks": [c["id"] for c in top_chunks],
                "confidence": "low",
                "context": top_chunks,
            }

        result["context"] = top_chunks
        return result
