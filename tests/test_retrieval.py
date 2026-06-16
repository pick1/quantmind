"""Tests for retrieval modules — embedder, searcher, reranker, synthesizer, pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from quantmind.retrieval.synthesizer import Synthesizer
from quantmind.retrieval.pipeline import RetrievalPipeline


class TestSynthesizer:
    def test_citation_extraction(self):
        text = "According to [SOURCE: Paper A | \"key finding\"] and [SOURCE: Paper B | \"another result\"]."
        cits = Synthesizer._extract_citations(text)
        assert len(cits) == 2
        assert cits[0] == {"title": "Paper A", "excerpt": "key finding"}
        assert cits[1] == {"title": "Paper B", "excerpt": "another result"}

    def test_citation_extraction_no_citations(self):
        assert Synthesizer._extract_citations("Plain answer without citations.") == []

    def test_citation_extraction_multiline_excerpt(self):
        text = '[SOURCE: Paper | "line1\nline2"]'
        cits = Synthesizer._extract_citations(text)
        assert len(cits) == 1
        assert "line1" in cits[0]["excerpt"]
        assert "line2" in cits[0]["excerpt"]

    def test_format_chunks(self):
        chunks = [
            {"title": "Paper Alpha", "source_type": "arxiv", "document": "Some content here."},
            {"title": "Paper Beta", "source_type": "pdf", "document": "More content."},
        ]
        result = Synthesizer._format_chunks(chunks)
        assert "[1]" in result
        assert "[2]" in result
        assert "Paper Alpha" in result
        assert "Paper Beta" in result
        assert "Some content here." in result
        assert "More content." in result

    def test_synthesize_calls_llm_with_proper_prompt(self):
        """Verify the LLM is called with the expected system + user prompt."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Answer with [SOURCE: Test | \"citation\"]."
        synth = Synthesizer(llm=mock_llm)

        chunks = [{"id": "c1", "title": "Test", "document": "Content."}]
        result = synth.synthesize("Test query", chunks)

        assert result["answer"] == "Answer with [SOURCE: Test | \"citation\"]."
        assert len(result["citations"]) == 1
        assert result["has_citations"] is True
        assert result["unverified"] is False
        assert result["used_chunks"] == ["c1"]

        # Verify the prompt included the chunk and the query
        call_kwargs = mock_llm.complete.call_args[1]
        assert "Test query" in call_kwargs["user"]
        assert "Content." in call_kwargs["user"]

    def test_synthesize_no_citations_shows_warning(self):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Answer without citations."
        synth = Synthesizer(llm=mock_llm)

        result = synth.synthesize("query", [{"id": "c1", "title": "T", "document": "C."}])
        assert result["has_citations"] is False
        assert result["unverified"] is True

    def test_synthesize_llm_failure_returns_error(self):
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("LLM down")
        synth = Synthesizer(llm=mock_llm)

        result = synth.synthesize("query", [{"id": "c1", "title": "T", "document": "C."}])
        assert "Failed to generate" in result["answer"]
        assert result["confidence"] == "low"

    def test_confidence_high(self):
        assert Synthesizer._estimate_confidence(
            [{"relevance_score": 0.85}], True
        ) == "high"

    def test_confidence_medium(self):
        assert Synthesizer._estimate_confidence(
            [{"relevance_score": 0.55}], True
        ) == "medium"

    def test_confidence_low_no_chunks(self):
        assert Synthesizer._estimate_confidence([], True) == "low"

    def test_confidence_low_no_citations(self):
        assert Synthesizer._estimate_confidence(
            [{"relevance_score": 0.9}], False
        ) == "low"


class TestRetrievalPipeline:
    def test_query_empty_result(self):
        """Pipeline returns 'No relevant documents' when search is empty."""
        with patch("quantmind.retrieval.pipeline.QueryEmbedder") as me, \
             patch("quantmind.retrieval.pipeline.Searcher") as ms, \
             patch("quantmind.retrieval.pipeline.Reranker") as mr, \
             patch("quantmind.retrieval.pipeline.Synthesizer") as msyn:

            mock_embedder = MagicMock()
            mock_embedder.embed.return_value = [0.1] * 768
            me.return_value = mock_embedder

            mock_searcher = MagicMock()
            mock_searcher.search.return_value = []
            ms.return_value = mock_searcher

            pipeline = RetrievalPipeline()
            result = pipeline.query("test question")

            assert "No relevant documents found" in result["answer"]
            assert result["confidence"] == "low"

    def test_query_full_flow(self):
        """Pipeline runs full embed→search→rerank→synthesize flow."""
        with patch("quantmind.retrieval.pipeline.QueryEmbedder") as me, \
             patch("quantmind.retrieval.pipeline.Searcher") as ms, \
             patch("quantmind.retrieval.pipeline.Reranker") as mr, \
             patch("quantmind.retrieval.pipeline.Synthesizer") as msyn:

            candidates = [
                {"id": "c1", "title": "Paper1", "document": "Content1",
                 "distance": 0.2, "source_type": "arxiv"},
                {"id": "c2", "title": "Paper2", "document": "Content2",
                 "distance": 0.3, "source_type": "arxiv"},
            ]
            top_chunks = [
                {**candidates[0], "relevance_score": 0.9},
                {**candidates[1], "relevance_score": 0.7},
            ]

            mock_embedder = MagicMock()
            mock_embedder.embed.return_value = [0.1] * 768
            me.return_value = mock_embedder

            mock_searcher = MagicMock()
            mock_searcher.search.return_value = candidates
            ms.return_value = mock_searcher

            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = top_chunks
            mr.return_value = mock_reranker

            mock_synth = MagicMock()
            mock_synth.synthesize.return_value = {
                "answer": "Answer with [SOURCE: Paper1 | \"Content1\"].",
                "citations": [{"title": "Paper1", "excerpt": "Content1"}],
                "has_citations": True,
                "unverified": False,
                "used_chunks": ["c1", "c2"],
                "confidence": "high",
            }
            msyn.return_value = mock_synth

            pipeline = RetrievalPipeline()
            result = pipeline.query("test question")

            assert result["has_citations"] is True
            assert result["confidence"] == "high"
            assert result["context"] == top_chunks
            mock_searcher.search.assert_called_once()
            mock_reranker.rerank.assert_called_once()
            mock_synth.synthesize.assert_called_once()
