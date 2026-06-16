"""Query page — search bar, response panel, citation blocks, related docs."""

import logging

import streamlit as st

from quantmind.auth import check_auth

if not check_auth():
    st.stop()

from quantmind.retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger("quantmind.ui.query")

st.set_page_config(page_title="Query — QuantMind", page_icon="🔍", layout="wide")
st.title("🔍 Query")
st.markdown("Ask questions against your ingested knowledge base.")


def get_pipeline() -> RetrievalPipeline:
    """Retrieve the cached pipeline from app.py."""
    from app import get_retrieval_pipeline
    return get_retrieval_pipeline()


# ── Search input ───────────────────────────────────────────────────────────
with st.container():
    query = st.text_input(
        "Your question",
        placeholder="e.g. What are the main factor models used in equity portfolio construction?",
        key="query_input",
    )
    col1, col2 = st.columns([1, 8])
    with col1:
        submitted = st.button("Search", type="primary", key="search_btn")
    with col2:
        st.caption("Ask about ingested documents. Ingest documents first via the Library or Ingest pages.")


# ── Handle query ───────────────────────────────────────────────────────────
if submitted and query.strip():
    with st.spinner("Searching knowledge base..."):
        try:
            pipeline = get_pipeline()
            result = pipeline.query(query)
        except Exception as e:
            st.error(f"Query failed: {e}")
            logger.exception("Query error")
            result = None

    if result:
        # Store in session history
        st.session_state.query_history.append({
            "query": query,
            "result": result,
        })
        # Keep last 10
        st.session_state.query_history = st.session_state.query_history[-10:]

        # ── Response ───────────────────────────────────────────────────────
        st.divider()

        # Confidence indicator
        conf = result.get("confidence", "low")
        conf_icons = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        unverified = result.get("unverified", False)

        col1, col2, col3 = st.columns([6, 2, 2])
        with col1:
            st.markdown("### Answer")
        with col2:
            st.markdown(f"Confidence: {conf_icons.get(conf, '⚪')} **{conf}**")
        with col3:
            if unverified:
                st.warning("⚠️ Unverified")
            else:
                st.success("✓ Verified")

        # Answer text
        st.markdown(result.get("answer", ""))

        # ── Citations ──────────────────────────────────────────────────────
        citations = result.get("citations", [])
        if citations:
            st.markdown("### Citations")
            for i, cit in enumerate(citations, 1):
                with st.container(border=True):
                    st.markdown(f"**{cit.get('title', 'Source')}**")
                    st.markdown(f"> {cit.get('excerpt', '')}")

        # ── Related docs ───────────────────────────────────────────────────
        context = result.get("context", [])
        if context:
            with st.expander("Related documents"):
                seen_titles = set()
                for chunk in context:
                    title = chunk.get("title", "Untitled")
                    if title not in seen_titles:
                        seen_titles.add(title)
                        st.markdown(f"- **{title}** (score: {chunk.get('relevance_score', 'N/A'):.3f})")

        # ── Used chunks (debug) ─────────────────────────────────────────────
        with st.expander("Retrieval details"):
            used = result.get("used_chunks", [])
            st.markdown(f"Used **{len(used)}** chunks")
            for cid in used:
                st.code(cid)

elif submitted and not query.strip():
    st.warning("Please enter a question.")


# ── Query history ──────────────────────────────────────────────────────────
if st.session_state.query_history:
    with st.sidebar:
        st.divider()
        st.markdown("**Query History**")
        for i, entry in enumerate(reversed(st.session_state.query_history[-5:])):
            q = entry["query"]
            short_q = q[:60] + "..." if len(q) > 60 else q
            st.caption(f"{i+1}. {short_q}")
