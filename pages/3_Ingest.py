"""Ingest page — arXiv ID input, PDF upload, text paste, URL link, progress feedback."""

import logging
import tempfile
from pathlib import Path

import streamlit as st

from quantmind.auth import check_auth

if not check_auth():
    st.stop()

logger = logging.getLogger("quantmind.ui.ingest")

st.set_page_config(page_title="Ingest — QuantMind", page_icon="📥", layout="wide")
st.title("📥 Ingest Documents")
st.markdown("Add documents to your knowledge base — drag & drop files, paste a link, or type text.")


def get_pipeline():
    from app import get_retrieval_pipeline
    return None


def show_status_card(result):
    """Display a detailed status card after ingestion."""
    status = result["status"]

    if status == "ingested":
        source_type = result.get("source_type", "?")
        title = result.get("title", "Untitled")
        chunk_count = result.get("chunk_count", 0)
        tags = result.get("tags", [])

        # Source-type icon
        icons = {"arxiv": "📄", "pdf": "📕", "text": "📝", "web": "🌐"}

        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"{icons.get(source_type, '📄')} **{title}**  "
                    f"<span style='color:gray;font-size:0.85em'>— {source_type}</span>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"<span style='font-size:1.8em;color:green'>✅</span>",
                    unsafe_allow_html=True,
                )

            meta_cols = st.columns(4)
            meta_cols[0].metric("Chunks", chunk_count)
            meta_cols[1].metric("Source ID", result.get("source_id", "")[:12] + "…")
            meta_cols[2].metric("Doc ID", result.get("doc_id", ""))
            meta_cols[3].metric("Tags", ", ".join(tags) if tags else "—")

            # Content preview
            preview = result.get("content_preview", "")
            if preview:
                with st.expander("Content preview", expanded=False):
                    st.text(preview[:1000] + ("…" if len(preview) > 1000 else ""))

            # arXiv extras
            if source_type == "arxiv":
                authors = result.get("authors", "")
                abstract = result.get("abstract", "")
                if authors:
                    st.caption(f"👤 {authors[:200]}")
                if abstract:
                    with st.expander("Abstract", expanded=False):
                        st.text(abstract[:1500])

            st.caption(
                f"Ingested at {__import__('datetime').datetime.now().strftime('%H:%M:%S')}"
            )

    elif status == "skipped":
        with st.container(border=True):
            st.markdown(f"⏭️ **Skipped** — {result.get('reason', 'already exists')}")
            st.caption(f"Source: `{result.get('source_id', '?')}`")

    else:
        st.error(f"Error — {result.get('reason', 'unknown')}")


def run_ingest(method, **kwargs):
    """Run ingestion in a pipeline with detailed status reporting."""
    from quantmind.ingest.pipeline import IngestionPipeline
    from app import get_llm_client, get_doc_store, get_vec_store

    pipeline = IngestionPipeline(
        doc_store=get_doc_store(),
        vec_store=get_vec_store(),
        llm=get_llm_client(),
    )

    with st.spinner("Processing..."):
        if method == "arxiv":
            result = pipeline.ingest_arxiv(kwargs["arxiv_id"])
        elif method == "pdf":
            result = pipeline.ingest_pdf(kwargs["pdf_path"])
        elif method == "text":
            result = pipeline.ingest_text(
                kwargs["text"],
                title=kwargs.get("title"),
                source_id=kwargs.get("source_id"),
            )
        elif method == "url":
            result = pipeline.ingest_url(kwargs["url"])
        else:
            st.error(f"Unknown method: {method}")
            return

    show_status_card(result)

    # Store in session state for sidebar reference
    if result["status"] in ("ingested", "skipped"):
        st.session_state["last_ingest"] = result


def show_recent_ingestions():
    """Show the last 8 ingested documents."""
    try:
        from app import get_doc_store
        docs = get_doc_store().list_documents(limit=8)
    except Exception:
        return

    if not docs:
        return

    st.divider()
    st.subheader("📋 Recent ingestions")

    icons = {"arxiv": "📄", "pdf": "📕", "text": "📝", "web": "🌐"}

    for doc in docs:
        dt = doc.get("ingestion_date", "")
        try:
            import datetime
            parsed = datetime.datetime.fromisoformat(dt)
            date_str = parsed.strftime("%b %d, %H:%M")
        except Exception:
            date_str = dt[-16:-7] if len(dt) > 16 else dt

        tags = doc.get("domain_tags", "[]")
        try:
            import json
            tag_str = ", ".join(json.loads(tags)) if tags and json.loads(tags) else ""
        except Exception:
            tag_str = ""

        icon = icons.get(doc.get("source_type", ""), "📄")
        chunk_str = f" — {doc.get('chunk_count', 0)} chunks"
        if tag_str:
            chunk_str += f" | 🏷️ {tag_str}"

        st.markdown(
            f"{icon} **{doc.get('title', '?')[:80]}** "
            f"<span style='color:gray;font-size:0.85em'>{date_str}{chunk_str}</span>",
            unsafe_allow_html=True,
        )


# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "arXiv", "Drag & Drop / Upload", "Plain Text", "Link / URL", "📧 Gmail Alerts"
])


# ── Tab 1: arXiv ───────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        arxiv_id = st.text_input(
            "arXiv ID",
            placeholder="e.g. 2309.01234",
            key="arxiv_id_input",
        )
    with col2:
        arxiv_submit = st.button("Fetch & Ingest", type="primary", key="arxiv_btn")

    if arxiv_submit:
        if arxiv_id.strip():
            run_ingest("arxiv", arxiv_id=arxiv_id.strip())
        else:
            st.warning("Please enter an arXiv ID.")

    st.divider()
    st.markdown("**Bulk keyword search**")
    col_q, col_m, col_b = st.columns([4, 1, 1])
    with col_q:
        arxiv_query = st.text_input(
            "Keyword search",
            placeholder="e.g. factor momentum risk",
            key="arxiv_query_input",
        )
    with col_m:
        max_results = st.number_input("Max", min_value=1, max_value=50, value=5, key="arxiv_max")
    with col_b:
        query_submit = st.button("Search & Ingest", key="arxiv_query_btn")

    if query_submit and arxiv_query.strip():
        from quantmind.ingest.arxiv import search_by_keyword
        from quantmind.ingest.pipeline import IngestionPipeline
        from app import get_llm_client, get_doc_store, get_vec_store

        with st.spinner(f"Searching arXiv for \"{arxiv_query}\"..."):
            try:
                papers = search_by_keyword(arxiv_query, max_results=int(max_results))
            except Exception as e:
                st.error(f"arXiv search failed: {e}")
                papers = []

        if not papers:
            st.warning("No papers found on arXiv for that query.")
        else:
            st.success(f"Found {len(papers)} papers. Ingesting...")
            progress = st.progress(0)
            results = []
            pipeline = IngestionPipeline(
                doc_store=get_doc_store(),
                vec_store=get_vec_store(),
                llm=get_llm_client(),
            )
            for i, paper in enumerate(papers):
                result = pipeline.ingest_arxiv(paper["source_id"])
                results.append(result)
                progress.progress((i + 1) / len(papers))

            ingested = sum(1 for r in results if r["status"] == "ingested")
            skipped = sum(1 for r in results if r["status"] == "skipped")
            errors = sum(1 for r in results if r["status"] == "error")

            if ingested:
                st.success(f"{ingested} ingested, {skipped} skipped, {errors} errors")
            elif errors:
                st.error(f"{errors} errors, {skipped} skipped")
            else:
                st.info(f"All {skipped} already in library")

            # Show summary cards for each ingested paper
            for result in results:
                if result["status"] == "ingested":
                    show_status_card(result)

    elif query_submit:
        st.warning("Please enter a search query.")


# ── Tab 2: Drag & Drop / Upload ────────────────────────────────────────────
with tab2:
    st.markdown(
        "Drop a **PDF** or **text file** below, or click to browse."
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md"],
        key="file_uploader",
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_ext = Path(uploaded_file.name).suffix.lower()

        if file_ext == ".pdf":
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            st.success(f"📄 **{uploaded_file.name}** ({len(uploaded_file.getbuffer())} bytes)")

            if st.button("Ingest PDF", type="primary", key="pdf_btn"):
                run_ingest("pdf", pdf_path=tmp_path)

        elif file_ext in (".txt", ".md"):
            content = uploaded_file.read().decode("utf-8", errors="replace")
            title = Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ").title()

            st.success(f"📝 **{uploaded_file.name}** ({len(content)} characters)")

            with st.expander("Preview", expanded=False):
                st.text(content[:2000] + ("..." if len(content) > 2000 else ""))

            if st.button("Ingest text file", type="primary", key="txt_btn"):
                run_ingest("text", text=content, title=title)

    else:
        st.caption("Supported formats: PDF, TXT, Markdown (.md)")


# ── Tab 3: Plain Text ──────────────────────────────────────────────────────
with tab3:
    text_title = st.text_input("Document title (optional)", key="text_title")
    text_content = st.text_area("Paste text content", height=300, key="text_content")

    col1, col2 = st.columns([1, 4])
    with col1:
        text_submit = st.button("Ingest text", type="primary", key="text_btn")
    with col2:
        if text_content.strip():
            word_count = len(text_content.split())
            st.caption(f"{len(text_content)} chars / ~{word_count} words → ~{max(1, word_count // 512)} chunks")

    if text_submit:
        if text_content.strip():
            run_ingest(
                "text",
                text=text_content.strip(),
                title=text_title.strip() or None,
            )
        else:
            st.warning("Please paste some text content.")


# ── Tab 4: Link / URL ──────────────────────────────────────────────────────
with tab4:
    st.markdown(
        "Paste a link to a paper or article. QuantMind will auto-detect the source type."
    )

    url_input = st.text_input(
        "Article URL",
        placeholder="e.g. https://arxiv.org/abs/2309.01234  or  https://example.com/paper.pdf",
        key="url_input",
    )

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.caption(
            "**Auto-detects:** arXiv links, PDF URLs, and general article pages. "
            "For PDFs and web pages, the content is fetched and ingested automatically."
        )
    with col_btn:
        url_submit = st.button("Ingest from URL", type="primary", key="url_btn")

    if url_submit:
        if url_input.strip():
            run_ingest("url", url=url_input.strip())
        else:
            st.warning("Please paste a URL.")

    st.divider()
    st.markdown("**Supported link types**")

    link_types = {
        "arXiv": "`arxiv.org/abs/…` — extracted via arXiv API with title, authors, abstract",
        "PDF": "`…paper.pdf` or any URL serving PDF content — downloaded and extracted",
        "Web article": "Any other URL — HTML is fetched and cleaned to readable text",
    }
    for label, desc in link_types.items():
        st.markdown(f"- **{label}:** {desc}")


# ── Tab 5: Gmail Alerts ──────────────────────────────────────────────────
with tab5:
    st.markdown(
        "**Google Scholar Alert emails** — fetch new article links from your "
        "inbox and ingest them automatically."
    )

    # Check if Gmail is configured
    from quantmind.config import GMAIL_CREDENTIALS_FILE, GMAIL_TOKEN_FILE
    from pathlib import Path

    creds_exist = Path(GMAIL_CREDENTIALS_FILE).exists()
    token_exists = Path(GMAIL_TOKEN_FILE).exists()

    if not creds_exist:
        st.warning(
            "⚠️  Gmail not configured yet.\n\n"
            "**One-time setup:**\n"
            "1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)\n"
            "2. Create a project → Enable Gmail API\n"
            "3. Create OAuth 2.0 Client ID → **Desktop application**\n"
            "4. Download the JSON and save it to:\n"
            f"   `{GMAIL_CREDENTIALS_FILE}`\n"
        )

        with st.expander("Detailed setup guide", expanded=False):
            st.markdown(
                """
                1. **Create a Google Cloud Project**
                   - Go to https://console.cloud.google.com/projectcreate
                   - Name it (e.g. "QuantMind Gmail Connector")
                2. **Enable the Gmail API**
                   - Go to APIs & Services → Library
                   - Search for "Gmail API" → Enable
                3. **Create OAuth credentials**
                   - Go to APIs & Services → Credentials
                   - Click "Create Credentials" → OAuth client ID
                   - Application type: **Desktop application**
                   - Name: "QuantMind CLI"
                   - Download the JSON file
                4. **Save the credentials file**
                   - Place it at `data/gmail_credentials.json` in the QuantMind project
                5. **Run OAuth setup** (one-time):
                   ```
                   cd /home/dp/projects/quantmind
                   python3 -m quantmind.ingest.gmail_alerts setup
                   ```
                   This will give you a URL to visit, then paste the code back.
                """
            )
    elif not token_exists:
        st.info(
            "📋 Credentials file found. **Next step:** run the OAuth setup:\n\n"
            "```\n"
            "cd /home/dp/projects/quantmind && "
            "python3 -m quantmind.ingest.gmail_alerts setup\n"
            "```\n\n"
            "This will open a browser for Google authorization."
        )
    else:
        st.success(f"✅ Gmail authenticated — token: `{Path(GMAIL_TOKEN_FILE).name}`")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            max_emails = st.number_input(
                "Max alert emails", min_value=1, max_value=50, value=10,
                key="gmail_max_emails",
            )
        with col2:
            max_links = st.number_input(
                "Links per email", min_value=1, max_value=20, value=5,
                key="gmail_max_links",
            )
        with col3:
            re_process = st.checkbox(
                "Re-process seen", value=False, key="gmail_reprocess",
                help="Re-process alert emails that were already processed before",
            )

        check_btn = st.button(
            "📧 Fetch & Ingest Google Scholar Alerts", type="primary",
            key="gmail_btn", use_container_width=True,
        )

        if check_btn:
            from quantmind.ingest.pipeline import IngestionPipeline
            from app import get_llm_client, get_doc_store, get_vec_store

            pipeline = IngestionPipeline(
                doc_store=get_doc_store(),
                vec_store=get_vec_store(),
                llm=get_llm_client(),
            )

            with st.spinner("Fetching Gmail alerts and ingesting articles..."):
                result = pipeline.ingest_gmail_alerts(
                    max_emails=int(max_emails),
                    max_links_per_email=int(max_links),
                    re_process=re_process,
                )

            if result["status"] == "error":
                st.error(result.get("reason", "Unknown error"))
            else:
                # Summary card
                ingested = result.get("urls_ingested", 0)
                skipped = result.get("urls_skipped", 0)
                errors = result.get("urls_error", 0)
                found = result.get("urls_found", 0)
                checked = result.get("emails_checked", 0)

                st.success(
                    f"✅ **{checked}** emails checked, "
                    f"**{found}** article links found — "
                    f"{ingested} ingested, {skipped} already in library, {errors} errors"
                )

                # Per-email breakdown
                for email_res in result.get("results", []):
                    with st.container(border=True):
                        st.markdown(f"**{email_res['subject'][:80]}**")
                        st.caption(email_res.get("date", ""))
                        for url_res in email_res["urls"]:
                            status_icon = {
                                "ingested": "✅",
                                "skipped": "⏭️",
                                "error": "❌",
                            }.get(url_res["status"], "❓")
                            title = url_res.get("title", "")[:60]
                            display = f"{status_icon} {url_res['url'][:70]}…"
                            if title:
                                display += f" — *{title}*"
                            st.markdown(display)

        # Section: Preview recent alert emails
        st.divider()
        with st.expander("Preview recent alert emails", expanded=False):
            try:
                from quantmind.ingest.gmail_alerts import list_alert_emails
                gmail_preview = st.button(
                    "📋 Show recent alerts", key="gmail_preview_btn"
                )
                if gmail_preview:
                    with st.spinner("Fetching recent alerts..."):
                        emails = list_alert_emails(max_results=5)
                    if not emails:
                        st.info("No alerts found.")
                    else:
                        for e in emails:
                            st.markdown(
                                f"- **{e.get('subject', '?')[:80]}** "
                                f"`{e.get('date', '')[:20]}`"
                            )
            except Exception:
                st.caption("Not authenticated yet — run OAuth setup first.")

# ── Recent ingestions (shown at bottom of page) ──────────────────────────
show_recent_ingestions()
