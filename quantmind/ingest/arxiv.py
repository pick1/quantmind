"""arXiv API client — fetch papers by ID and keyword search.

Uses the arXiv API (export.arxiv.org/api/query) with Atom XML parsing.
Implements exponential backoff with configurable retries.
"""

import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET

import httpx

from quantmind.config import (
    ARXIV_DEFAULT_CATEGORY,
    ARXIV_MAX_RETRIES,
    ARXIV_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger("quantmind.ingest.arxiv")

#ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivFetchError(Exception):
    """Raised when arXiv API returns an error or rate-limits us."""


def _source_id(arxiv_id: str) -> str:
    """Normalise an arXiv ID to a clean source_id."""
    # Strip version suffixes like v1, v2
    return re.sub(r"v\d+$", "", arxiv_id.strip()).lower()


def fetch_by_id(arxiv_id: str) -> dict:
    """Fetch a single paper by its arXiv ID.

    Returns dict with keys: source_id, title, authors, abstract, source_type='arxiv'.
    """
    normalized = _source_id(arxiv_id)
    url = f"{ARXIV_API_BASE}?id_list={normalized}"
    data = _fetch_with_retry(url)
    entry = _parse_single_entry(data, normalized)
    return entry


def search_by_keyword(
    query: str,
    max_results: int = 10,
    category: str = ARXIV_DEFAULT_CATEGORY,
) -> list[dict]:
    """Search arXiv by keyword, optionally filtered by category.

    Returns a list of paper dicts (same shape as fetch_by_id).
    """
    #safe_query = httpx.utils.quote(query)
    from urllib.parse import quote
    safe_query = quote(query)
    cat_filter = f" AND cat:{category}" if category else ""
    url = (
        f"{ARXIV_API_BASE}?search_query=all:{safe_query}{cat_filter}"
        f"&start=0&max_results={max_results}&sortBy=relevance"
    )
    data = _fetch_with_retry(url)
    return _parse_multi_entry(data)


def _fetch_with_retry(url: str) -> str:
    """GET the arXiv API with exponential backoff retry."""
    last_error = None
    delay = ARXIV_RETRY_DELAY_SECONDS

    for attempt in range(ARXIV_MAX_RETRIES):
        try:
            r = httpx.get(url, timeout=15)
            if r.status_code == 200:
                return r.text
            elif r.status_code == 403:
                raise ArxivFetchError("arXiv API returned 403 (rate limited).")
            else:
                last_error = ArxivFetchError(
                    f"arXiv API returned HTTP {r.status_code}"
                )
        except httpx.TimeoutException as e:
            last_error = ArxivFetchError(f"arXiv API timeout: {e}")
        except httpx.RequestError as e:
            last_error = ArxivFetchError(f"arXiv API request failed: {e}")

        if attempt < ARXIV_MAX_RETRIES - 1:
            logger.warning(
                "arXiv API attempt %d/%d failed, retrying in %.1fs...",
                attempt + 1, ARXIV_MAX_RETRIES, delay,
            )
            time.sleep(delay)
            delay *= 2  # exponential backoff

    raise last_error or ArxivFetchError("Unknown arXiv fetch error")


def _parse_single_entry(xml_text: str, expected_id: str) -> dict:
    """Parse a single Atom entry from arXiv response."""
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", ARXIV_NS)
    if not entries:
        raise ArxivFetchError(f"No arXiv entry found for {expected_id}")

    entry = entries[0]
    return _entry_to_dict(entry)


def _parse_multi_entry(xml_text: str) -> list[dict]:
    """Parse multiple Atom entries from arXiv response."""
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", ARXIV_NS)
    return [_entry_to_dict(e) for e in entries]


def _entry_to_dict(entry: ET.Element) -> dict:
    """Convert an Atom <entry> to a flat dict."""
    title_el = entry.find("atom:title", ARXIV_NS)
    summary_el = entry.find("atom:summary", ARXIV_NS)
    id_el = entry.find("atom:id", ARXIV_NS)

    # Extract arXiv ID from the URL-like id (e.g. http://arxiv.org/abs/2309.01234v1)
    raw_id = id_el.text.strip() if id_el is not None else ""
    arxiv_id_match = re.search(r"/(\d+\.\d+)(v\d+)?", raw_id)
    source_id = arxiv_id_match.group(1) if arxiv_id_match else raw_id

    # Authors
    authors = []
    for author_el in entry.findall("atom:author", ARXIV_NS):
        name_el = author_el.find("atom:name", ARXIV_NS)
        if name_el is not None:
            authors.append(name_el.text.strip())

    title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
    abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""

    return {
        "source_type": "arxiv",
        "source_id": source_id,
        "title": title,
        "authors": "; ".join(authors),
        "abstract": abstract,
    }
