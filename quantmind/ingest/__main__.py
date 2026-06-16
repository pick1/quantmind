#!/usr/bin/env python3
"""CLI entry point for QuantMind ingestion.

Usage:
    python -m quantmind.ingest --arxiv 2309.01234
    python -m quantmind.ingest --pdf /path/to/paper.pdf
    python -m quantmind.ingest --text "Some financial text..."
    python -m quantmind.ingest --arxiv-query "factor momentum risk" --max 10
"""

import argparse
import logging
import sys

from quantmind.config import LOG_LEVEL
from quantmind.ingest.pipeline import IngestionPipeline

logger = logging.getLogger("quantmind.ingest")


def main():
    parser = argparse.ArgumentParser(
        description="QuantMind Local — document ingestion CLI",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--arxiv", help="arXiv ID (e.g. 2309.01234)")
    source.add_argument("--pdf", help="Path to a PDF file")
    source.add_argument("--text", help="Plain text content to ingest")
    source.add_argument("--url", help="URL to a paper, PDF, or article")
    source.add_argument(
        "--arxiv-query",
        help="Keyword search on arXiv (e.g. 'factor momentum risk')",
    )
    parser.add_argument(
        "--max", type=int, default=5,
        help="Max results for --arxiv-query (default: 5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(levelname)s | %(name)s | %(message)s",
    )

    pipeline = IngestionPipeline()

    if args.arxiv:
        result = pipeline.ingest_arxiv(args.arxiv)
        _report(result)

    elif args.pdf:
        result = pipeline.ingest_pdf(args.pdf)
        _report(result)

    elif args.text:
        result = pipeline.ingest_text(args.text)
        _report(result)

    elif args.url:
        result = pipeline.ingest_url(args.url)
        _report(result)

    elif args.arxiv_query:
        from quantmind.ingest.arxiv import search_by_keyword

        papers = search_by_keyword(args.arxiv_query, max_results=args.max)
        print(f"Found {len(papers)} papers. Ingesting...")
        results = []
        for paper in papers:
            result = pipeline.ingest_arxiv(paper["source_id"])
            results.append(result)
            _report(result)

        ingested = sum(1 for r in results if r["status"] == "ingested")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        errors = sum(1 for r in results if r["status"] == "error")
        print(f"\nDone: {ingested} ingested, {skipped} skipped, {errors} errors")


def _report(result: dict):
    status = result["status"]
    sid = result.get("source_id", "?")
    if status == "ingested":
        print(f"  ✓ [{sid}] Ingested — {result.get('chunk_count', 0)} chunks, tags={result.get('tags', [])}")
    elif status == "skipped":
        print(f"  − [{sid}] Skipped — {result.get('reason', '')}")
    else:
        print(f"  ✗ [{sid}] Error — {result.get('reason', '')}")


if __name__ == "__main__":
    main()
