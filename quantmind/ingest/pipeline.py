"""Ingestion pipeline — orchestrates fetch → chunk → embed → store.

Handles three source types: arXiv, PDF files, and plain text.
Each document is processed sequentially with progress tracking.
"""

import hashlib
import logging
from typing import Optional

from quantmind.config import CHUNK_OVERLAP, CHUNK_SIZE
from quantmind.ingest.tagger import tag_document
from quantmind.llm_client import LLMClient
from quantmind.store.sqlite import DocumentStore

logger = logging.getLogger("quantmind.ingest.pipeline")


_DOMAIN_TAGGING_TODO = (
    "TODO: add async queue (Celery/ARQ) for batch ingestion in future versions"
)


class IngestionPipeline:
    """Orchestrates a single document through the full ingestion flow."""

    def __init__(
        self,
        doc_store: Optional[DocumentStore] = None,
        vec_store: Optional["VectorStore"] = None,
        llm: Optional[LLMClient] = None,
    ):
        from quantmind.store.chroma import VectorStore

        self.doc_store = doc_store or DocumentStore()
        self.vec_store = vec_store or VectorStore()
        self.llm = llm or LLMClient()

    # ── Main entry points ──────────────────────────────────────────────────

    def ingest_arxiv(self, arxiv_id: str) -> dict:
        """Fetch an arXiv paper and ingest it."""
        from quantmind.ingest.arxiv import fetch_by_id, _source_id

        source_id = _source_id(arxiv_id)

        if self.doc_store.exists(source_id):
            logger.info("arXiv paper %s already in library — skipping", source_id)
            return {"status": "skipped", "source_id": source_id, "reason": "already exists"}

        paper = fetch_by_id(arxiv_id)
        return self._process_document(
            source_type="arxiv",
            source_id=source_id,
            title=paper["title"],
            authors=paper["authors"],
            abstract=paper["abstract"],
            full_text=paper["abstract"],  # arXiv: abstract IS the full text in v1
            source_display=f"arXiv:{source_id}",
        )

    def ingest_pdf(self, pdf_path: str) -> dict:
        """Extract text from a PDF and ingest it."""
        from quantmind.ingest.pdf import extract_and_chunk

        source_id, full_text, chunks = extract_and_chunk(
            pdf_path, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
        )

        if self.doc_store.exists(source_id):
            logger.info("PDF %s already in library — skipping", pdf_path)
            return {"status": "skipped", "source_id": source_id, "reason": "already exists"}

        # Extract title from first line or filename
        title = pdf_path.rsplit("/", 1)[-1].replace(".pdf", "").replace("_", " ").replace("-", " ").title()

        return self._process_document(
            source_type="pdf",
            source_id=source_id,
            title=title,
            authors="",
            abstract="",
            full_text=full_text,
            source_display=pdf_path,
        )

    def ingest_url(self, url: str) -> dict:
        """Ingest a document from a URL.

        Auto-detects arXiv links, PDF links, and generic article pages.
        """
        import httpx
        import tempfile
        from urllib.parse import urlparse

        url = url.strip()
        logger.info("Ingesting from URL: %s", url)

        # 1. Detect arXiv links (arxiv.org/abs/XXXX.XXXXX or /pdf/XXXX.XXXXX)
        arxiv_id = self._parse_arxiv_url(url)
        if arxiv_id:
            return self.ingest_arxiv(arxiv_id)

        # 2. Detect PDF links by extension or path pattern
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        if path_lower.endswith(".pdf") or "/pdf/" in path_lower:
            return self._ingest_pdf_from_url(url)

        # 3. Check content-type via HEAD request
        try:
            head = httpx.head(url, timeout=10, follow_redirects=True)
            content_type = head.headers.get("content-type", "").lower()
            if "pdf" in content_type:
                return self._ingest_pdf_from_url(url)
        except Exception:
            pass

        # 4. Fallback: fetch as HTML/article and extract text
        return self._ingest_webpage(url)

    # ── URL helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_arxiv_url(url: str) -> Optional[str]:
        """Extract arXiv ID from various arXiv URL formats.

        Handles:
          https://arxiv.org/abs/2309.01234
          https://arxiv.org/abs/2309.01234v3
          https://arxiv.org/pdf/2309.01234.pdf
          http://arxiv.org/abs/2309.01234
        """
        import re

        # Match the arXiv ID pattern in common URL forms
        m = re.search(
            r"(?:arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?)",
            url,
            re.IGNORECASE,
        )
        if m:
            return m.group(1)
        return None

    def _ingest_pdf_from_url(self, url: str) -> dict:
        """Download a PDF from a URL and ingest it."""
        import httpx
        import tempfile

        logger.info("Downloading PDF from URL: %s", url)
        try:
            resp = httpx.get(url, timeout=60, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.exception("Failed to download PDF from %s", url)
            return {"status": "error", "source_id": url, "reason": f"download_failed: {e}"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        try:
            return self.ingest_pdf(tmp_path)
        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _ingest_webpage(self, url: str) -> dict:
        """Fetch a webpage, extract readable text, and ingest as text."""
        import httpx
        import re

        logger.info("Fetching webpage: %s", url)
        try:
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.exception("Failed to fetch webpage %s", url)
            return {"status": "error", "source_id": url, "reason": f"fetch_failed: {e}"}

        html = resp.text

        # Simple HTML-to-text extraction (strip tags, preserve paragraphs)
        # Remove script/style blocks
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Replace block-level tags with newlines
        for tag in ("</p>", "</div>", "</section>", "</article>", "</li>", "</h[1-6]>", "<br\\s*/?>", "<hr\\s*/?>"):
            text = re.sub(tag, "\n", text, flags=re.IGNORECASE)

        # Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode common entities
        text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)

        # Collapse whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"  +", " ", text)
        text = text.strip()

        if not text:
            return {"status": "error", "source_id": url, "reason": "no_content_extracted"}

        # Try to extract a title from <title> tag
        title = self._extract_title(html) or url

        source_id = hashlib.sha256(url.encode()).hexdigest()[:16]

        if self.doc_store.exists(source_id):
            logger.info("URL %s already in library — skipping", url)
            return {"status": "skipped", "source_id": source_id, "reason": "already exists"}

        return self._process_document(
            source_type="web",
            source_id=source_id,
            title=title,
            authors="",
            abstract="",
            full_text=text,
            source_display=url,
        )

    def ingest_gmail_alerts(
        self,
        max_emails: int = 10,
        max_links_per_email: int = 5,
        re_process: bool = False,
    ) -> dict:
        """Fetch Google Scholar Alert emails and ingest any new article links.

        See quantmind/ingest/gmail_alerts.py for the full implementation.
        Requires Gmail OAuth setup (run once):
          python3 -m quantmind.ingest.gmail_alerts setup
        """
        from quantmind.ingest.gmail_alerts import fetch_and_ingest

        return fetch_and_ingest(
            pipeline=self,
            max_emails=max_emails,
            max_links_per_email=max_links_per_email,
            re_process=re_process,
        )

    @staticmethod
    def _extract_title(html: str) -> Optional[str]:
        """Extract the <title> from an HTML document."""
        import re

        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            title = m.group(1).strip()
            # Remove trailing site name separators like " — SiteName" or " | SiteName"
            title = re.sub(r"\s+[—\-|:].*$", "", title).strip()
            return title or None
        return None

    def ingest_text(
        self,
        text: str,
        source_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> dict:
        """Ingest plain text content."""
        # Hash the text for dedup
        sid = source_id or hashlib.sha256(text.encode()).hexdigest()[:16]

        if self.doc_store.exists(sid):
            logger.info("Text source %s already in library — skipping", sid)
            return {"status": "skipped", "source_id": sid, "reason": "already exists"}

        return self._process_document(
            source_type="text",
            source_id=sid,
            title=title or f"Text document {sid[:8]}",
            authors="",
            abstract="",
            full_text=text,
            source_display=title or sid[:16],
        )

    # ── Core processing ────────────────────────────────────────────────────

    def _process_document(
        self,
        source_type: str,
        source_id: str,
        title: str,
        authors: str,
        abstract: str,
        full_text: str,
        source_display: str,
    ) -> dict:
        """Shared processing: chunk → embed → store."""
        logger.info("Ingesting %s: %s (%s)", source_type, source_display, source_id)

        # Chunk
        text_for_chunking = abstract if source_type == "arxiv" and not full_text else full_text
        from quantmind.ingest.pdf import chunk_text

        chunks = chunk_text(text_for_chunking, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        if not chunks:
            logger.warning("No chunks generated for %s — skipping", source_id)
            return {"status": "error", "source_id": source_id, "reason": "no content"}

        # Embed
        chunk_ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
        embeddings = []
        for i, chunk_text_ in enumerate(chunks):
            try:
                emb = self.llm.embed(chunk_text_)
                embeddings.append(emb)
            except Exception:
                logger.exception("Embedding failed for chunk %d of %s", i, source_id)
                return {"status": "error", "source_id": source_id, "reason": "embedding failed"}

        # Store vectors
        metadatas = [
            {
                "doc_id": source_id,
                "source_type": source_type,
                "title": title,
                "chunk_index": i,
                "token_count": len(chunk_text_.split()),
            }
            for i, chunk_text_ in enumerate(chunks)
        ]

        try:
            self.vec_store.add_chunks(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )
        except Exception:
            logger.exception("ChromaDB insert failed for %s", source_id)
            return {"status": "error", "source_id": source_id, "reason": "chroma_write_failed"}

        # Store metadata in SQLite
        doc_id = self.doc_store.add_document(
            source_type=source_type,
            source_id=source_id,
            title=title,
            authors=authors,
            abstract=abstract,
            chunk_count=len(chunks),
        )

        # Domain tagging
        try:
            tags = tag_document(title, abstract, llm=self.llm)
            if tags:
                self.doc_store.update_tags(doc_id, tags)
        except Exception:
            logger.exception("Domain tagging failed for %s — continuing", source_id)

        logger.info(
            "Ingested %s: %d chunks, %d tags",
            source_id,
            len(chunks),
            len(tags) if tags else 0,
        )
        return {
            "status": "ingested",
            "doc_id": doc_id,
            "source_id": source_id,
            "source_type": source_type,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "content_preview": chunks[0][:500] if chunks else "",
            "chunk_count": len(chunks),
            "tags": tags or [],
        }
