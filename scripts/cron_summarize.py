#!/usr/bin/env python3
"""Cron summarizer — auto-generate plain-language summaries for articles that
lack them.

Runs in the background, processes up to 5 unsummarised articles per tick,
and saves each summary to the SQLite `simple_summary` column.
"""

import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for cron runs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantmind.ingest.tagger import CONTROLLED_VOCABULARY
from quantmind.llm_client import LLMClient
from quantmind.store.sqlite import DocumentStore

LOG_DIR = Path.home() / ".quantmind"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "cron_summarize.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("cron_summarize")

MAX_PER_TICK = 5
SUMMARY_SYSTEM = (
    "You are a research communicator who distils complex papers into "
    "clear, accessible prose."
)


def _generate_simple_explanation(llm, title: str, abstract: str) -> str:
    """Ask the LLM to explain an article in plain language (one paragraph)."""
    prompt = (
        f"Explain the following research paper in one paragraph of plain, "
        f"accessible language. Assume the reader knows basic finance/economics "
        f"but is not an expert in this specific sub-field. Avoid jargon where "
        f"possible; define unavoidable terms briefly. Focus on what the paper "
        f"found and why it matters.\n\n"
        f"Title: {title or '(no title)'}\n"
        f"Abstract: {abstract or '(no abstract)'}\n\n"
        f"Plain-language summary:"
    )
    try:
        response = llm.complete(system=SUMMARY_SYSTEM, user=prompt)
        return response.strip()
    except Exception:
        logger.exception("Failed to generate summary for '%s'", title)
        return ""


def main():
    llm = LLMClient()
    store = DocumentStore()

    # Fetch documents without a summary
    all_docs = store.list_documents(offset=0, limit=5000)
    unsummarised = [d for d in all_docs if not d.get("simple_summary")]

    if not unsummarised:
        logger.info("All %d documents already have summaries — nothing to do.", len(all_docs))
        return

    batch = unsummarised[:MAX_PER_TICK]
    logger.info(
        "%d / %d documents lack summaries; processing %d this tick.",
        len(unsummarised), len(all_docs), len(batch),
    )

    for doc in batch:
        doc_id = doc.get("id")
        title = doc.get("title", "Untitled")
        abstract = doc.get("abstract", "")

        if not abstract:
            logger.info("Skipping doc %s — no abstract.", doc_id)
            continue

        logger.info("Summarising doc %s: %s", doc_id, title[:80])
        summary = _generate_simple_explanation(llm, title, abstract)
        if summary:
            store.update_summary(doc_id, summary)
            logger.info("Saved summary for doc %s (%d chars)", doc_id, len(summary))
        else:
            logger.warning("Empty summary for doc %s", doc_id)

        # Brief pause between LLM calls so we don't hammer Ollama
        time.sleep(1)

    logger.info("Summarisation tick complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Summarisation cron failed")
        sys.exit(1)
