"""SQLite document metadata store — schema init, CRUD operations."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from quantmind.config import SQLITE_PATH


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type     TEXT    NOT NULL CHECK (source_type IN ('arxiv', 'pdf', 'text', 'web')),
    source_id       TEXT    NOT NULL UNIQUE,       -- arXiv ID, filename, or content hash
    title           TEXT,
    authors         TEXT,
    abstract        TEXT,
    ingestion_date  TEXT    NOT NULL,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    domain_tags     TEXT    DEFAULT '[]',          -- JSON array of tags
    status          TEXT    NOT NULL DEFAULT 'ingested'
                      CHECK (status IN ('ingested', 'parse_error', 'failed')),
    simple_summary  TEXT                        -- plain‑language summary (cached)
);

CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_documents_domain_tags ON documents(domain_tags);
"""


class DocumentStore:
    """Persistent SQLite-backed document metadata store."""

    def __init__(self, db_path: str = SQLITE_PATH):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist. Runs migrations for existing tables."""
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        self._migrate_v1_add_web_source_type()
        self._migrate_v2_add_simple_summary()

    def _migrate_v1_add_web_source_type(self) -> None:
        """Migrate existing tables whose CHECK constraint on source_type predates 'web'."""
        cur = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        row = cur.fetchone()
        if row is None:
            return
        sql = row[0]
        # If 'web' already in the constraint, nothing to do
        if "'web'" in sql:
            return

        logger = __import__('logging').getLogger("quantmind.store.sqlite")
        logger.info("Migrating documents table CHECK constraint to include 'web' source type")
        # Recreate the table with updated constraint
        self._conn.executescript("""
            PRAGMA foreign_keys=off;
            BEGIN TRANSACTION;
            ALTER TABLE documents RENAME TO documents_old;
            CREATE TABLE documents (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type     TEXT    NOT NULL CHECK (source_type IN ('arxiv', 'pdf', 'text', 'web')),
                source_id       TEXT    NOT NULL UNIQUE,
                title           TEXT,
                authors         TEXT,
                abstract        TEXT,
                ingestion_date  TEXT    NOT NULL,
                chunk_count     INTEGER NOT NULL DEFAULT 0,
                domain_tags     TEXT    DEFAULT '[]',
                status          TEXT    NOT NULL DEFAULT 'ingested'
                                  CHECK (status IN ('ingested', 'parse_error', 'failed'))
            );
            INSERT INTO documents SELECT * FROM documents_old;
            DROP TABLE documents_old;
            COMMIT;
            PRAGMA foreign_keys=on;
        """)
        self._conn.commit()

    def _migrate_v2_add_simple_summary(self) -> None:
        """Add simple_summary column to existing databases that lack it."""
        cur = self._conn.execute("PRAGMA table_info(documents)")
        cols = [r[1] for r in cur.fetchall()]
        if "simple_summary" not in cols:
            logger = __import__('logging').getLogger("quantmind.store.sqlite")
            logger.info("Migrating documents table — adding simple_summary column")
            self._conn.execute("ALTER TABLE documents ADD COLUMN simple_summary TEXT")
            self._conn.commit()

    # ── Insert ─────────────────────────────────────────────────────────────

    def add_document(
        self,
        source_type: str,
        source_id: str,
        title: Optional[str] = None,
        authors: Optional[str] = None,
        abstract: Optional[str] = None,
        chunk_count: int = 0,
        domain_tags: Optional[list[str]] = None,
    ) -> int:
        """Insert a new document record. Returns the new row id."""
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """INSERT INTO documents
               (source_type, source_id, title, authors, abstract,
                ingestion_date, chunk_count, domain_tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                source_type,
                source_id,
                title,
                authors,
                abstract,
                now,
                chunk_count,
                json.dumps(domain_tags or []),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    # ── Existence check (duplicate detection) ──────────────────────────────

    def exists(self, source_id: str) -> bool:
        """Return True if a document with the given source_id exists."""
        cur = self._conn.execute(
            "SELECT 1 FROM documents WHERE source_id = ? LIMIT 1",
            (source_id,),
        )
        return cur.fetchone() is not None

    # ── Queries ────────────────────────────────────────────────────────────

    def get_document(self, doc_id: int) -> Optional[dict]:
        """Fetch a single document by its id."""
        cur = self._conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_documents(
        self,
        offset: int = 0,
        limit: int = 50,
        source_type: Optional[str] = None,
        domain_tag: Optional[str] = None,
    ) -> list[dict]:
        """Paginated document listing with optional filters."""
        where_clauses = []
        params = []

        if source_type:
            where_clauses.append("source_type = ?")
            params.append(source_type)
        if domain_tag:
            where_clauses.append("domain_tags LIKE ?")
            params.append(f'%{domain_tag}%')

        where = ""
        if where_clauses:
            where = "WHERE " + " AND ".join(where_clauses)

        cur = self._conn.execute(
            f"SELECT * FROM documents {where} ORDER BY ingestion_date DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        )
        return [dict(r) for r in cur.fetchall()]

    def count_documents(
        self,
        source_type: Optional[str] = None,
        domain_tag: Optional[str] = None,
    ) -> int:
        """Count documents with optional filters."""
        where_clauses = []
        params = []

        if source_type:
            where_clauses.append("source_type = ?")
            params.append(source_type)
        if domain_tag:
            where_clauses.append("domain_tags LIKE ?")
            params.append(f'%{domain_tag}%')

        where = ""
        if where_clauses:
            where = "WHERE " + " AND ".join(where_clauses)

        cur = self._conn.execute(
            f"SELECT COUNT(*) as cnt FROM documents {where}", params
        )
        return cur.fetchone()["cnt"]

    def update_tags(self, doc_id: int, tags: list[str]) -> None:
        """Update domain_tags for a document."""
        self._conn.execute(
            "UPDATE documents SET domain_tags = ? WHERE id = ?",
            (json.dumps(tags), doc_id),
        )
        self._conn.commit()

    def update_chunk_count(self, doc_id: int, chunk_count: int) -> None:
        """Update chunk count for a document."""
        self._conn.execute(
            "UPDATE documents SET chunk_count = ? WHERE id = ?",
            (chunk_count, doc_id),
        )
        self._conn.commit()

    def update_summary(self, doc_id: int, summary: str) -> None:
        """Persist a plain‑language summary for a document."""
        self._conn.execute(
            "UPDATE documents SET simple_summary = ? WHERE id = ?",
            (summary, doc_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
