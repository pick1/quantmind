"""PDF text extraction and chunking.

Uses pdfplumber for text extraction from PDF files.
Splits extracted text into ~512-token chunks with configurable overlap.
Figures and tables are extracted as text only (tables as tab-separated rows;
figures are skipped — no image captioning in v1).
"""

import hashlib
import logging
import re
from pathlib import Path

from quantmind.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger("quantmind.ingest.pdf")


def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF using pdfplumber."""
    import pdfplumber

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        with pdfplumber.open(str(path)) as pdf:
            pages_text = []
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                pages_text.append(f"[Page {page_num}]\n{text}")
            return "\n\n".join(pages_text)
    except Exception as e:
        raise ValueError(f"Failed to extract text from {pdf_path}: {e}")


def file_hash(pdf_path: str) -> str:
    """SHA-256 hash of the PDF file contents for duplicate detection."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks of approximately *chunk_size* tokens.

    Token count is approximated as word count (simple split on whitespace).
    Overlap ensures the LLM has context across chunk boundaries.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    words = []
    for para in paragraphs:
        para_words = para.split()
        if para_words:
            words.extend(para_words)
            words.append("¶")  # paragraph marker

    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        chunk_str = " ".join(chunk_words)
        chunk_str = chunk_str.replace(" ¶ ", "\n\n").replace("¶", "\n\n").strip()
        if chunk_str:
            chunks.append(chunk_str)

        step = max(chunk_size - overlap, 1)
        start += step

    return chunks


def extract_and_chunk(
    pdf_path: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[str, str, list[str]]:
    """Extract text from PDF and chunk it in one call.

    Returns:
        (source_id, full_text, list_of_chunks)
    """
    source_id = file_hash(pdf_path)
    text = extract_text(pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return source_id, text, chunks
