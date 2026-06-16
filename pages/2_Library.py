"""Library page — paginated table of ingested documents with filter/search."""

import logging

import streamlit as st

from quantmind.auth import check_auth

if not check_auth():
    st.stop()

from quantmind.ingest.tagger import CONTROLLED_VOCABULARY

logger = logging.getLogger("quantmind.ui.library")

st.set_page_config(page_title="Library — QuantMind", page_icon="📚", layout="wide")
st.title("📚 Document Library")
st.markdown("Browse and manage ingested documents.")


def get_store():
    from app import get_doc_store
    return get_doc_store()


PAGE_SIZE = 20

# ── Filters ────────────────────────────────────────────────────────────────
store = get_store()

col1, col2 = st.columns([2, 3])
with col1:
    source_filter = st.selectbox(
        "Source type",
        ["All", "arxiv", "pdf", "text", "web"],
        index=0,
    )
with col2:
    tag_filter = st.selectbox(
        "Domain tag",
        ["All"] + sorted(CONTROLLED_VOCABULARY),
        index=0,
    )

# Pagination
total = store.count_documents(
    source_type=None if source_filter == "All" else source_filter,
    domain_tag=None if tag_filter == "All" else tag_filter,
)

if "library_page" not in st.session_state:
    st.session_state.library_page = 0

total_pages = max((total - 1) // PAGE_SIZE + 1, 1)

col_prev, col_page, col_next = st.columns([1, 3, 1])
with col_prev:
    if st.button("◀ Previous", disabled=st.session_state.library_page == 0):
        st.session_state.library_page -= 1
        st.rerun()
with col_page:
    st.markdown(
        f"<div style='text-align: center'>Page **{st.session_state.library_page + 1}** of **{total_pages}** ({total} documents)</div>",
        unsafe_allow_html=True,
    )
with col_next:
    if st.button("Next ▶", disabled=st.session_state.library_page >= total_pages - 1):
        st.session_state.library_page += 1
        st.rerun()

# ── Document table ─────────────────────────────────────────────────────────
offset = st.session_state.library_page * PAGE_SIZE
docs = store.list_documents(
    offset=offset,
    limit=PAGE_SIZE,
    source_type=None if source_filter == "All" else source_filter,
    domain_tag=None if tag_filter == "All" else tag_filter,
)

if not docs:
    st.info("No documents found. Go to the **Ingest** page to add documents.")
else:
    # Display as a simple table
    for doc in docs:
        tags = doc.get("domain_tags", "[]")
        if isinstance(tags, str):
            try:
                import json
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []
        tags_str = ", ".join(tags) if tags else "—"

        with st.container(border=True):
            col_t, col_s, col_c = st.columns([5, 1, 1])
            with col_t:
                st.markdown(f"**{doc.get('title', 'Untitled')}**")
                st.caption(f"{doc.get('source_type', '?')} · {doc.get('source_id', '')}")
            with col_s:
                st.caption(f"Chunks: {doc.get('chunk_count', 0)}")
            with col_c:
                status = doc.get("status", "ingested")
                if status == "ingested":
                    st.success("✓")
                elif status == "parse_error":
                    st.error("⚠")
                else:
                    st.caption(status)

            st.markdown(f"Tags: {tags_str}")
