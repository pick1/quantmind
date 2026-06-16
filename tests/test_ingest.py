"""Tests for ingestion modules — arXiv fetch, PDF/chunk, pipeline, tagging."""

import json
from unittest.mock import MagicMock, patch

import pytest

from quantmind.ingest.arxiv import (
    _entry_to_dict,
    _source_id,
)
from quantmind.ingest.pdf import chunk_text
from quantmind.ingest.tagger import (
    CONTROLLED_VOCABULARY,
    _parse_tags_response,
)


# ── arXiv helpers ──────────────────────────────────────────────────────────

class TestArxivHelpers:
    def test_source_id_normalisation(self):
        assert _source_id("2309.01234v2") == "2309.01234"
        assert _source_id("2309.01234V1") == "2309.01234"
        assert _source_id(" 2309.01234 ") == "2309.01234"

    def test_entry_to_dict(self):
        """Parse a minimal Atom entry into expected dict shape."""
        import xml.etree.ElementTree as ET

        xml = """<?xml version="1.0"?>
<entry xmlns="http://www.w3.org/2005/Atom">
  <id>http://arxiv.org/abs/2309.01234v1</id>
  <title>Test Paper Title</title>
  <summary>This is a test abstract.</summary>
  <author><name>Alice Smith</name></author>
  <author><name>Bob Jones</name></author>
</entry>"""
        entry = ET.fromstring(xml)
        result = _entry_to_dict(entry)

        assert result["source_type"] == "arxiv"
        assert result["source_id"] == "2309.01234"
        assert result["title"] == "Test Paper Title"
        assert result["abstract"] == "This is a test abstract."
        assert "Alice Smith" in result["authors"]
        assert "Bob Jones" in result["authors"]


# ── Chunking ───────────────────────────────────────────────────────────────

class TestChunking:
    def test_chunks_simple_text(self):
        text = "word " * 1000
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert 10 <= len(chunks) <= 20

    def test_chunks_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text("\n\n\n") == []

    def test_chunks_paragraph_preservation(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, chunk_size=50, overlap=5)
        assert len(chunks) >= 1
        # Paragraph markers should become newlines
        assert any("\n\n" in c for c in chunks)

    def test_chunks_overlap(self):
        text = "token " * 500
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        # With overlap, chunks overlap in source words
        assert len(chunks) >= 10

    def test_chunks_small_text_no_overlap(self):
        text = "short text"
        chunks = chunk_text(text, chunk_size=1000, overlap=100)
        assert len(chunks) == 1
        assert "short text" in chunks[0]


# ── Tagger ─────────────────────────────────────────────────────────────────

class TestTagger:
    def test_controlled_vocabulary_size(self):
        assert len(CONTROLLED_VOCABULARY) == 9

    def test_parse_valid_json_response(self):
        tags = _parse_tags_response('["factor-models", "ml-methods"]')
        assert tags == ["factor-models", "ml-methods"]

    def test_parse_json_with_extra_text(self):
        tags = _parse_tags_response(
            'Here are the tags: ["risk-management", "macro"] thanks'
        )
        assert "risk-management" in tags
        assert "macro" in tags

    def test_parse_fallback_word_match(self):
        tags = _parse_tags_response(
            "This paper discusses risk-management and nlp-finance techniques."
        )
        assert "risk-management" in tags
        assert "nlp-finance" in tags

    def test_parse_fallback_limits_to_3(self):
        text = " ".join(CONTROLLED_VOCABULARY)  # all 9 tags
        tags = _parse_tags_response(text)
        assert len(tags) <= 3

    def test_parse_invalid_response(self):
        tags = _parse_tags_response("I don't know what tags to assign.")
        assert tags == []

    def test_parse_empty_response(self):
        tags = _parse_tags_response("")
        assert tags == []

    @patch("quantmind.ingest.tagger.LLMClient")
    def test_tag_document_calls_llm(self, mock_llm_cls):
        from quantmind.ingest.tagger import tag_document

        mock_instance = MagicMock()
        mock_instance.provider = MagicMock(value="ollama")
        mock_instance.complete.return_value = '["factor-models", "ml-methods"]'
        mock_llm_cls.return_value = mock_instance

        tags = tag_document("Test", "Abstract", llm=mock_instance)
        assert "factor-models" in tags
        mock_instance.complete.assert_called_once()

    def test_tag_document_empty_content(self):
        from quantmind.ingest.tagger import tag_document

        tags = tag_document("", "", llm=None)
        assert tags == []


# ── Pipeline integration (mocked stores) ───────────────────────────────────

class TestPipeline:
    def test_ingest_text_flow(self):
        """IngestionPipeline.ingest_text with mocked stores."""
        from quantmind.ingest.pipeline import IngestionPipeline
        from quantmind.store.sqlite import DocumentStore

        doc_store = DocumentStore(":memory:")
        # Mock VectorStore
        with patch("quantmind.ingest.pipeline.VectorStore") as mock_vs:
            mock_vs_instance = MagicMock()
            mock_vs.return_value = mock_vs_instance

            with patch("quantmind.ingest.pipeline.LLMClient") as mock_llm_cls:
                mock_llm = MagicMock()
                mock_llm.embed.return_value = [0.5] * 768
                mock_llm_cls.return_value = mock_llm

                pipeline = IngestionPipeline(
                    doc_store=doc_store,
                    vec_store=mock_vs_instance,
                    llm=mock_llm,
                )
                result = pipeline.ingest_text(
                    "Some financial text about factor models.",
                    title="Test Document",
                )

                assert result["status"] == "ingested"
                assert result["chunk_count"] >= 1
                mock_vs_instance.add_chunks.assert_called_once()
                mock_llm.embed.assert_called()

    def test_ingest_text_duplicate_skipped(self):
        """Duplicate source_id correctly skips re-ingestion."""
        from quantmind.ingest.pipeline import IngestionPipeline
        from quantmind.store.sqlite import DocumentStore

        doc_store = DocumentStore(":memory:")
        with patch("quantmind.ingest.pipeline.VectorStore") as mock_vs, \
             patch("quantmind.ingest.pipeline.LLMClient") as mock_llm_cls:

            pipeline = IngestionPipeline(
                doc_store=doc_store,
                vec_store=MagicMock(),
                llm=MagicMock(),
            )
            # First ingest
            doc_store.add_document("text", "dup-id", title="Original")
            # Second ingest — should skip
            result = pipeline.ingest_text(
                "Some text", source_id="dup-id"
            )
            assert result["status"] == "skipped"
            assert result["reason"] == "already exists"
