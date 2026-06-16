"""Knowledge Base — card-based wiki organised by domain category.

Pulls ingested documents from the library, groups them by domain tag,
and displays each as a digestible card with optional AI-powered
plain-language explanations and category digests.

Summaries are persisted in SQLite so they survive page refreshes.
"""

import json
import logging

import streamlit as st

from quantmind.auth import check_auth

if not check_auth():
    st.stop()

from quantmind.ingest.tagger import CONTROLLED_VOCABULARY
from quantmind.llm_client import LLMClient, ProviderUnavailableError

logger = logging.getLogger("quantmind.ui.kb")

st.set_page_config(
    page_title="Knowledge Base — QuantMind",
    page_icon="🧠",
    layout="wide",
)
st.title("🧠 Knowledge Base")
st.markdown("Browse your ingested articles by category, with plain-language summaries.")


# ── Helpers ───────────────────────────────────────────────────────────────────

_SOURCE_ICONS = {
    "arxiv": "📄",
    "pdf": "📕",
    "web": "🌐",
    "text": "📝",
    "gmail": "📧",
}


def _source_icon(source_type: str) -> str:
    return _SOURCE_ICONS.get(source_type, "📄")


def get_store():
    from app import get_doc_store
    return get_doc_store()


def _parse_tags(tags_raw) -> list[str]:
    if isinstance(tags_raw, list):
        return tags_raw
    if isinstance(tags_raw, str):
        try:
            return json.loads(tags_raw)
        except (json.JSONDecodeError, TypeError):
            return []
    return []


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
        response = llm.complete(
            system="You are a research communicator who distils complex papers into clear, accessible prose.",
            user=prompt,
        )
        return response.strip()
    except ProviderUnavailableError:
        return "⚠️ LLM unavailable — cannot generate summary."
    except Exception:
        logger.exception("Failed to generate simple explanation")
        return "⚠️ Could not generate summary."


def _generate_category_digest(llm, articles: list[dict]) -> str:
    """Generate a 2-3 sentence digest of what a category covers."""
    titles = [a.get("title", "Untitled") for a in articles if a.get("title")]
    if not titles:
        return ""
    prompt = (
        f"Here are the titles of several research articles in a financial "
        f"domain. Write 2-3 sentences summarising the main themes and topics "
        f"this collection covers.\n\n"
        f"Titles:\n" + "\n".join(f"- {t}" for t in titles) + "\n\n"
        f"Thematic summary:"
    )
    try:
        response = llm.complete(
            system="You are a research librarian who synthesises collections into concise thematic overviews.",
            user=prompt,
        )
        return response.strip()
    except Exception:
        return ""


# ── Load data ────────────────────────────────────────────────────────────────

store = get_store()
all_docs = store.list_documents(offset=0, limit=5000, source_type=None)

if not all_docs:
    st.info("No documents found. Go to the **Ingest** page to add documents.")
    st.stop()

# Categorise documents by domain tag
categorized: dict[str, list[dict]] = {tag: [] for tag in CONTROLLED_VOCABULARY}
uncategorized: list[dict] = []

for doc in all_docs:
    tags = _parse_tags(doc.get("domain_tags", "[]"))
    if tags:
        for tag in tags:
            if tag in categorized:
                categorized[tag].append(doc)
    else:
        uncategorized.append(doc)

# Remove empty categories
categorized = {k: v for k, v in categorized.items() if v}

# Sort categories by article count (most articles first)
category_order = sorted(categorized.keys(), key=lambda t: len(categorized[t]), reverse=True)


# ── Sidebar — category count summary ─────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 Summary")
    st.markdown(f"**Total documents:** {len(all_docs)}")
    st.markdown(f"**Categories:** {len(categorized)}")
    for tag in category_order:
        st.markdown(f"- **{tag}:** {len(categorized[tag])}")
    if uncategorized:
        st.markdown(f"- *Uncategorized:* {len(uncategorized)}")

    count = len(all_docs)
    if count > 20:
        st.caption(f"Showing all {count} documents.")


# ── Category tabs ────────────────────────────────────────────────────────────

tab_labels = ["All"] + category_order
if uncategorized:
    tab_labels.append("Uncategorized")

tabs = st.tabs(tab_labels)

llm = None
try:
    llm = LLMClient()
except Exception:
    pass  # will show fallback messages


# ── Persistent summary cache ─────────────────────────────────────────────────
# Load any existing summaries from the DB into session state on first load
if "kb_simple_explanations" not in st.session_state:
    st.session_state.kb_simple_explanations = {}
    for doc in all_docs:
        doc_id = doc.get("id")
        summary = doc.get("simple_summary") or ""
        if doc_id and summary:
            st.session_state.kb_simple_explanations[f"persist_{doc_id}"] = summary
if "kb_digests" not in st.session_state:
    st.session_state.kb_digests = {}


# ═══════════════════════════════════════════════════════════════════════════
# Card renderer (defined before use below)
# ═══════════════════════════════════════════════════════════════════════════

def _render_card(doc: dict, llm, _key_suffix: str = "") -> None:
    """Render a single article as a card inside a container."""
    title = doc.get("title", "Untitled")
    source_type = doc.get("source_type", "?")
    authors = doc.get("authors", "") or ""
    abstract = doc.get("abstract", "") or ""
    chunk_count = doc.get("chunk_count", 0)
    ingestion_date = doc.get("ingestion_date", "")[:10] if doc.get("ingestion_date") else ""
    source_id = doc.get("source_id", "")
    doc_id = doc.get("id", "")
    tags = _parse_tags(doc.get("domain_tags", "[]"))
    uid = f"{doc_id}_{source_id[:12]}" if doc_id else f"doc_{hash(source_id) % 10**6}"
    unique_key = f"explain_{uid}{_key_suffix}"

    # Check for an existing persisted summary
    persist_key = f"persist_{doc_id}" if doc_id else None
    existing_summary = (
        st.session_state.kb_simple_explanations.get(persist_key, "")
        if persist_key else ""
    )
    should_show = existing_summary or unique_key in st.session_state.kb_simple_explanations

    with st.container(border=True):
        # ── Header row ──
        col_icon, col_title, col_meta = st.columns([0.5, 5, 2])
        with col_icon:
            st.markdown(f"<div style='font-size:1.8rem; text-align:center'>{_source_icon(source_type)}</div>",
                        unsafe_allow_html=True)
        with col_title:
            st.markdown(f"**{title}**")
            if authors:
                st.caption(authors)
        with col_meta:
            st.caption(f"`{source_type}` · {chunk_count} chunks" + (f" · {ingestion_date}" if ingestion_date else ""))

        # ── Body row ──
        col_abstract, col_actions = st.columns([4, 1.5])
        with col_abstract:
            if abstract:
                st.markdown(abstract[:300] + ("…" if len(abstract) > 300 else ""))
            else:
                st.caption("No abstract available.")
            if tags:
                st.caption("🏷️ " + ", ".join(f"`{t}`" for t in tags))

        with col_actions:
            btn_label = "🤖 Explain"
            if existing_summary:
                btn_label = "🤖 Regenerate"

            if st.button(btn_label, key=unique_key, use_container_width=True):
                if llm:
                    with st.spinner("Generating plain-language summary..."):
                        explanation = _generate_simple_explanation(llm, title, abstract)
                    # Save to session state
                    st.session_state.kb_simple_explanations[unique_key] = explanation
                    # Persist to SQLite
                    if doc_id:
                        try:
                            store.update_summary(doc_id, explanation)
                            st.session_state.kb_simple_explanations[persist_key] = explanation
                        except Exception:
                            logger.exception("Failed to persist summary for doc %s", doc_id)
                else:
                    st.session_state.kb_simple_explanations[unique_key] = "⚠️ LLM unavailable."

            # Show summary: prefer newly generated, fall back to persisted
            if unique_key in st.session_state.kb_simple_explanations:
                st.info(st.session_state.kb_simple_explanations[unique_key])
            elif existing_summary:
                st.info(existing_summary)


# ── Render tabs ─────────────────────────────────────────────────────────────

# ── All tab ──
with tabs[0]:
    st.markdown(
        f"### All Documents ({len(all_docs)})"
    )

    # Show items grouped by category
    for tag in category_order:
        articles = categorized[tag]
        with st.expander(f"**{tag}** ({len(articles)} articles)", expanded=True):
            for doc in articles:
                _render_card(doc, llm, _key_suffix=f"_{tag}")

    if uncategorized:
        with st.expander(f"**Uncategorized** ({len(uncategorized)} items)", expanded=False):
            for doc in uncategorized:
                _render_card(doc, llm)


# ── Per-category tabs ──
for idx, tag in enumerate(category_order):
    with tabs[idx + 1]:
        articles = categorized[tag]
        st.markdown(f"### {tag} ({len(articles)} articles)")

        # Category digest — generate once per session
        digest_key = f"digest_{tag}"
        if digest_key not in st.session_state.kb_digests and llm:
            with st.spinner("Generating category overview..."):
                digest = _generate_category_digest(llm, articles)
                if digest:
                    st.session_state.kb_digests[digest_key] = digest

        digest = st.session_state.kb_digests.get(digest_key, "")
        if digest:
            with st.container(border=True):
                st.markdown(f"**🧠 Category overview**")
                st.markdown(digest)

        for doc in articles:
            _render_card(doc, llm)


# ── Uncategorized tab ──
if uncategorized:
    with tabs[-1]:
        st.markdown(f"### Uncategorized ({len(uncategorized)} items)")
        for doc in uncategorized:
            _render_card(doc, llm)
