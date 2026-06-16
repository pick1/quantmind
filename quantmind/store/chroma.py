"""ChromaDB vector store wrapper — persistent, 768-dim, hard-fail on mismatch."""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings

from quantmind.config import CHROMA_PERSIST_DIR, EMBED_DIMENSIONS


COLLECTION_NAME = "document_chunks"
EXPECTED_DIMENSIONS = EMBED_DIMENSIONS  # 768 — nomic-embed-text


class VectorStore:
    """ChromaDB client wrapping a single persistent collection.

    Validates embedding dimensions at startup. If the existing collection
    has different dimensions, raises ValueError with clear reset instructions
    — never silently ignores a dimension mismatch.
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._init_collection()

    def _init_collection(self):
        """Get or create the collection, validating dimensions."""
        existing = self._client.list_collections()
        matching = [c for c in existing if c.name == COLLECTION_NAME]

        if matching:
            col = matching[0]
            # ChromaDB doesn't expose dimensions on the collection object
            # directly in all versions, so we peek one record to validate.
            count = col.count()
            if count > 0:
                sample = col.peek()  # returns first entry
                embeddings = sample.get("embeddings")
                if embeddings is not None and len(embeddings) > 0 and len(embeddings[0]) != EXPECTED_DIMENSIONS:
                    actual = len(sample["embeddings"][0])
                    raise ValueError(
                        f"ChromaDB collection '{COLLECTION_NAME}' has {actual}-dim "
                        f"embeddings but {EXPECTED_DIMENSIONS} was expected. "
                        f"This likely means a different embedding backend was used. "
                        f"To fix: delete the collection or reset the persist directory "
                        f"({CHROMA_PERSIST_DIR}) and re-ingest documents."
                    )
            return col
        else:
            # Create new collection — no dimension validation needed
            return self._client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

    @property
    def collection(self):
        return self._collection

    @property
    def count(self) -> int:
        return self._collection.count()

    # ── CRUD helpers ───────────────────────────────────────────────────────

    def add_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Insert chunk embeddings into the collection."""
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 20,
    ) -> dict:
        """Cosine similarity search. Returns ChromaDB QueryResult dict."""
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def delete_document_chunks(self, doc_id_prefix: str) -> None:
        """Delete all chunks belonging to a document (by id prefix)."""
        results = self._collection.get(
            where={"doc_id": doc_id_prefix},
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])

    def reset(self) -> None:
        """Delete and recreate the collection (for backend migration)."""
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except ValueError:
            pass  # collection didn't exist
        self._collection = self._client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
