"""Settings page — provider status, active host, model info, ChromaDB stats."""

import logging

import streamlit as st

from quantmind.auth import check_auth

if not check_auth():
    st.stop()

from quantmind.config import (
    CHROMA_PERSIST_DIR,
    EMBED_DIMENSIONS,
    OLLAMA_EMBED_MODEL,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    SQLITE_PATH,
    TOP_K_CANDIDATES,
    TOP_K_RERANK,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from quantmind.llm_client import Provider

logger = logging.getLogger("quantmind.ui.settings")

st.set_page_config(page_title="Settings — QuantMind", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings")


def get_llm_client():
    from app import get_llm_client
    return get_llm_client()


def get_doc_store():
    from app import get_doc_store
    return get_doc_store()


def get_vec_store():
    from app import get_vec_store
    return get_vec_store()


# ── Provider info ──────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Provider Status")
    try:
        client = get_llm_client()
        provider_name = client.provider.value.upper() if client.provider else "UNKNOWN"
        host = client.active_host or "—"

        if client.provider == Provider.OLLAMA:
            st.success(f"**{provider_name}** @ `{host}`")
        elif client.provider == Provider.OPENROUTER:
            st.warning(f"**{provider_name}** @ `{host}`")
        else:
            st.error(f"**No provider** — check Ollama host or set OPENROUTER_API_KEY")
    except Exception as e:
        st.error(f"Failed to detect provider: {e}")

    st.markdown(f"- **Ollama host**: `{OLLAMA_HOST}`")
    st.markdown(f"- **Ollama model**: `{OLLAMA_MODEL}`")
    st.markdown(f"- **Embedding model**: `{OLLAMA_EMBED_MODEL}` ({EMBED_DIMENSIONS} dims)")
    st.markdown(f"- **Fallback**: `{OPENROUTER_MODEL}` @ `{OPENROUTER_BASE_URL}`")

# ── Storage info ───────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Storage")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**SQLite**")
        st.markdown(f"- Path: `{SQLITE_PATH}`")
        try:
            store = get_doc_store()
            st.markdown(f"- Documents: **{store.count_documents()}**")
        except Exception as e:
            st.caption(f"Unavailable: {e}")

    with col2:
        st.markdown("**ChromaDB**")
        st.markdown(f"- Persist dir: `{CHROMA_PERSIST_DIR}`")
        st.markdown(f"- Dimensions: **{EMBED_DIMENSIONS}** (nomic-embed-text)")
        try:
            vs = get_vec_store()
            st.markdown(f"- Chunks: **{vs.count}**")
        except Exception as e:
            st.caption(f"Unavailable: {e}")

# ── Configuration ──────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Configuration")
    st.markdown(f"- Chunk size: **{CHUNK_SIZE}** tokens")
    st.markdown(f"- Chunk overlap: **{CHUNK_OVERLAP}** tokens")
    st.markdown(f"- Top-k candidates (vector search): **{TOP_K_CANDIDATES}**")
    st.markdown(f"- Top-k after reranking: **{TOP_K_RERANK}**")

# ── Statistics ─────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Statistics")
    col1, col2 = st.columns(2)
    with col1:
        try:
            doc_count = get_doc_store().count_documents()
            st.metric("Documents ingested", doc_count)
        except Exception:
            st.metric("Documents ingested", "N/A")
    with col2:
        try:
            chunk_count = get_vec_store().count
            st.metric("Chunks stored", chunk_count)
        except Exception:
            st.metric("Chunks stored", "N/A")
