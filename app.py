#!/usr/bin/env python3
"""QuantMind Local — Streamlit entrypoint.

Shared resources (LLMClient, DocumentStore, VectorStore, RetrievalPipeline)
are cached via @st.cache_resource so they're initialised once per session.
"""

import logging

import streamlit as st

from quantmind.auth import check_auth, logout
from quantmind.config import (
    CHROMA_PERSIST_DIR,
    LOG_LEVEL,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OPENROUTER_MODEL,
    SQLITE_PATH,
)
from quantmind.llm_client import LLMClient, Provider, ProviderUnavailableError

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("quantmind")

# ── Auth gate — must come before any other Streamlit command ────────────────
if not check_auth():
    st.stop()

# ── Cached resources ───────────────────────────────────────────────────────

@st.cache_resource
def get_llm_client() -> LLMClient:
    return LLMClient()


@st.cache_resource
def get_doc_store():
    from quantmind.store.sqlite import DocumentStore
    return DocumentStore()


@st.cache_resource
def get_vec_store():
    from quantmind.store.chroma import VectorStore
    return VectorStore()


@st.cache_resource
def get_retrieval_pipeline():
    from quantmind.retrieval.pipeline import RetrievalPipeline
    return RetrievalPipeline()


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantMind Local",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise session state ───────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 QuantMind")
    st.caption("Financial Knowledge Extraction & Retrieval")

    st.divider()

    # Logout button
    if st.button("🔒 Logout", use_container_width=True):
        logout()

    st.divider()

    # Provider status badge
    provider_badge = st.empty()
    try:
        client = get_llm_client()
        if client.provider == Provider.OLLAMA:
            provider_badge.success(f"Ollama @ `{client.active_host}` ✓")
        elif client.provider == Provider.OPENROUTER:
            provider_badge.warning(f"OpenRouter fallback ⚠")
        else:
            provider_badge.error("No provider ✗")
    except Exception:
        provider_badge.info("⏳ Initialising...")

    st.caption(f"Host: `{OLLAMA_HOST}`")
    st.caption(f"Model: `{OLLAMA_MODEL}`")

    st.divider()
    st.caption(f"Fallback: `{OPENROUTER_MODEL}`")

    # Storage stats in sidebar
    try:
        store = get_doc_store()
        doc_count = store.count_documents()
        st.caption(f"Documents: **{doc_count}**")
    except Exception:
        pass

    try:
        vs = get_vec_store()
        st.caption(f"Chunks: **{vs.count}**")
    except Exception:
        pass

# ── Welcome page ───────────────────────────────────────────────────────────
# This is the default view shown when no page is selected.
st.title("Welcome to QuantMind Local")
st.markdown(
    """
    A local financial knowledge extraction and retrieval system.
    Ingest academic papers, news, and financial documents — then query
    them using natural language.

    **Get started:**
    - Go to **Ingest** to add documents (arXiv, PDF, link, or text)
    - Visit **Library** to browse ingested documents
    - Use **Query** to ask questions against your knowledge base
    - Check **Settings** for provider status and system info
    """
)

with st.expander("Quick start"):
    st.markdown(
        """
        1. Ensure Ollama is running: `ollama pull nomic-embed-text && ollama pull qwen3:14b`
        2. Set `OLLAMA_HOST` in `.env` (default: `http://localhost:11434`)
        3. Ingest a paper: `python -m quantmind.ingest --arxiv 2309.01234`
        4. Open the UI: `streamlit run app.py`
        """
    )
