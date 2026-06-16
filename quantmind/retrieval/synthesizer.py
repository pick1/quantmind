"""Synthesizer — LLM answer generation with strict citation formatting.

Takes the top-k reranked chunks and the user's query, formats them into a
structured prompt, calls the LLM, and extracts citations from the response.
"""

import logging
import re
from typing import Optional

from quantmind.llm_client import LLMClient

logger = logging.getLogger("quantmind.retrieval.synthesizer")

SYSTEM_PROMPT = """You are a financial research assistant. Answer only using the provided context.
For every claim, cite the source using this exact format: [SOURCE: {title} | "{excerpt}"]
If the context does not contain enough information, say "Insufficient context."
Never fabricate information. Never cite sources not present in the context."""

USER_PROMPT_TEMPLATE = """Context:
{formatted_chunks}

Question: {query}"""


class Synthesizer:
    """Generates a cited answer from retrieved chunks using the LLM."""

    def __init__(self, llm: Optional[LLMClient] = None):
        self._llm = llm or LLMClient()

    def synthesize(self, query: str, chunks: list[dict]) -> dict:
        """Generate an answer with citations from the top-k chunks.

        Args:
            query: The user's original query.
            chunks: Reranked list of chunk dicts (must have 'document', 'title', 'id').

        Returns:
            dict with keys:
                - answer: str — the LLM's synthesized answer
                - citations: list[dict] — parsed citations [{title, excerpt}]
                - has_citations: bool
                - unverified: bool — True if no citations found in response
                - used_chunks: list[str] — chunk IDs used
                - confidence: str — high / medium / low
        """
        formatted = self._format_chunks(chunks)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            formatted_chunks=formatted,
            query=query,
        )

        logger.debug("Synthesizing answer for query: %s", query[:80])

        try:
            response = self._llm.complete(
                system=SYSTEM_PROMPT,
                user=user_prompt,
            )
        except Exception:
            logger.exception("LLM synthesis failed")
            return {
                "answer": "Failed to generate answer due to a provider error.",
                "citations": [],
                "has_citations": False,
                "unverified": True,
                "used_chunks": [c["id"] for c in chunks],
                "confidence": "low",
            }

        citations = self._extract_citations(response)
        has_citations = len(citations) > 0
        confidence = self._estimate_confidence(chunks, has_citations)

        return {
            "answer": response,
            "citations": citations,
            "has_citations": has_citations,
            "unverified": not has_citations,
            "used_chunks": [c["id"] for c in chunks],
            "confidence": confidence,
        }

    @staticmethod
    def _format_chunks(chunks: list[dict]) -> str:
        """Format chunks into a numbered context block."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("title", "Untitled")
            source_type = chunk.get("source_type", "unknown")
            text = chunk.get("document", "")
            parts.append(
                f"[{i}] Source: {title} ({source_type})\n"
                f"    {text}\n"
            )
        return "\n".join(parts)

    @staticmethod
    def _extract_citations(response: str) -> list[dict]:
        """Parse [SOURCE: title | "excerpt"] citations from LLM response."""
        pattern = r'\[SOURCE:\s*(.*?)\s*\|\s*"(.*?)"\s*\]'
        matches = re.findall(pattern, response, re.DOTALL)
        return [
            {"title": title.strip(), "excerpt": excerpt.strip()}
            for title, excerpt in matches
        ]

    @staticmethod
    def _estimate_confidence(chunks: list[dict], has_citations: bool) -> str:
        """Estimate confidence based on retrieval quality and citations."""
        if not chunks:
            return "low"
        if not has_citations:
            return "low"

        # Check average relevance score if available
        scores = [
            c.get("relevance_score", 0.5)
            for c in chunks
            if c.get("relevance_score") is not None
        ]
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score > 0.7:
                return "high"
            elif avg_score > 0.4:
                return "medium"
            return "low"

        return "medium"
