#!/usr/bin/env python3
"""Cron runner — fetch Google Scholar Alerts and ingest article links.

Runs the pipeline for the latest unprocessed alerts. Logs to
~/.quantmind/cron_ingest.log. Exits 0 on success, 1 on error.
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for cron runs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantmind.ingest.pipeline import IngestionPipeline

LOG_DIR = Path.home() / ".quantmind"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "cron_ingest.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("cron_ingest")

try:
    pipeline = IngestionPipeline()
    result = pipeline.ingest_gmail_alerts(max_emails=5, max_links_per_email=8)
    logger.info("Ingestion complete: %s", result)
    print(f"Ingested {result.get('total_ingested', 0)} new articles from Gmail alerts.")
except Exception:
    logger.exception("Gmail alert ingestion failed")
    sys.exit(1)
