"""Configuration — environment variables, constants, model names."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma")
)
SQLITE_PATH = os.getenv("SQLITE_PATH", str(DATA_DIR / "quantmind.db"))

# ── Ollama (primary inference backend) ─────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ── OpenRouter (fallback) ──────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-5"
)

# ── Embedding dimensions ───────────────────────────────────────────────────
# nomic-embed-text outputs 768-dim vectors. Do NOT change this without also
# resetting the ChromaDB collection — mismatched dimensions cause hard-fail.
EMBED_DIMENSIONS = 768

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 512       # tokens per chunk
CHUNK_OVERLAP = 64     # token overlap between chunks

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K_CANDIDATES = 20   # candidates retrieved from vector search
TOP_K_RERANK = 5        # final top-N after cross-encoder reranking

# ── arXiv ──────────────────────────────────────────────────────────────────
ARXIV_DEFAULT_CATEGORY = "q-fin.*"
ARXIV_MAX_RETRIES = 3
ARXIV_RETRY_DELAY_SECONDS = 5  # base delay for exponential backoff

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Auth ────────────────────────────────────────────────────────────────────
# Password for the Streamlit UI login page. Set in .env for production;
# defaults to a random string (lock it down if exposed to the internet).
QUANTMIND_PASSWORD = os.getenv("QUANTMIND_PASSWORD", "quantmind-dev")

# ── Gmail / Google Scholar Alerts ──────────────────────────────────────────
GMAIL_CREDENTIALS_FILE = os.getenv(
    "GMAIL_CREDENTIALS_FILE", str(PROJECT_ROOT / "data" / "gmail_credentials.json")
)
GMAIL_TOKEN_FILE = os.getenv(
    "GMAIL_TOKEN_FILE", str(PROJECT_ROOT / "data" / "gmail_token.json")
)
GMAIL_PROCESSED_FILE = os.getenv(
    "GMAIL_PROCESSED_FILE", str(PROJECT_ROOT / "data" / "gmail_processed.json")
)

# ── Provider enum values (strings for serialisability) ─────────────────────
PROVIDER_OLLAMA = "OLLAMA"
PROVIDER_OPENROUTER = "OPENROUTER"
PROVIDER_UNAVAILABLE = "UNAVAILABLE"
