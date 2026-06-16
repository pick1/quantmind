"""Query embedding — converts natural language queries to vectors via Ollama.

Reuses the LLMClient's embed() method, which prefers Ollama (nomic-embed-text)
with graceful fallback to sentence-transformers on CPU.
"""

import logging
from typing import Optional

from quantmind.llm_client import LLMClient

logger = logging.getLogger("quantmind.retrieval.embedder")


class QueryEmbedder:
    """Embeds query text using the configured LLM client."""

    def __init__(self, llm: Optional[LLMClient] = None):
        self._llm = llm or LLMClient()

    def embed(self, query: str) -> list[float]:
        """Convert a plain-text query to a vector embedding.

        Args:
            query: The user's natural language query.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            ProviderUnavailableError: If no embedding backend is available.
        """
        if not query.strip():
            raise ValueError("Cannot embed empty query")

        logger.debug("Embedding query (%d chars)", len(query))
        return self._llm.embed(query)
