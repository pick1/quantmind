"""Gmail connector — fetch Google Scholar alert emails and extract article links.

Two modes:
  1. CLI setup:  python3 -m quantmind.ingest.gmail_alerts setup
  2. Pipeline:   pipeline.ingest_gmail_alerts()

Requires a one-time Google Cloud OAuth setup (see HANDOFF.md or README).
"""

import json
import logging
import os
import re
import webbrowser
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from quantmind.config import (
    GMAIL_CREDENTIALS_FILE,
    GMAIL_TOKEN_FILE,
    GMAIL_PROCESSED_FILE,
)

logger = logging.getLogger("quantmind.ingest.gmail_alerts")

# Google Scholar Alert email search query
SCHOLAR_QUERY = "from:scholaralerts-noreply@google.com"

# OAuth scopes — read-only is sufficient (we never send or delete)
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# ── Helpers ────────────────────────────────────────────────────────────────────

_SCHOLAR_URL_PATTERN = re.compile(
    r"scholar\.google\.com/scholar_url\?.*?url=(https?://[^\s&\"]+)",
    re.IGNORECASE,
)

# Direct http/https URLs (non-Google, non-scholar)
_DIRECT_URL_PATTERN = re.compile(
    r'(?:href=["\'])?(https?://(?:www\.)?(?!scholar\.google|google\.com|accounts\.google)[^\s<"\']+(?:\.html?|/)?[^\s<"\',;]*)',
    re.IGNORECASE,
)


def _get_credentials() -> "google.oauth2.credentials.Credentials":
    """Load existing token or return None."""
    from google.oauth2.credentials import Credentials

    token_path = Path(GMAIL_TOKEN_FILE)
    if token_path.exists():
        return Credentials.from_authorized_user_file(str(token_path), GMAIL_SCOPES)
    return None


def _save_token(credentials) -> None:
    """Persist OAuth token to disk."""
    token_path = Path(GMAIL_TOKEN_FILE)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_data = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
        "expiry": credentials.expiry.isoformat() if credentials.expiry else None,
    }
    with open(token_path, "w") as f:
        json.dump(token_data, f, indent=2)
    logger.info("OAuth token saved to %s", token_path)


def _get_service():
    """Get an authenticated Gmail API service, refreshing token if needed."""
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = _get_credentials()

    # Refresh if expired
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_token(creds)

    if not creds or not creds.valid:
        raise ConnectionError(
            "Gmail not authenticated. Run this once from a machine with a browser:\n"
            "  python3 -m quantmind.ingest.gmail_alerts setup\n"
            "Then copy the resulting token file to the server."
        )

    return build("gmail", "v1", credentials=creds)


# ── OAuth setup (one-time) ───────────────────────────────────────────────────


def do_oauth_setup() -> None:
    """Run the interactive OAuth flow — prints a URL, user authorizes, pastes code.

    Must be run on a machine with a browser at least once. Saves token to
    GMAIL_TOKEN_FILE (default: data/gmail_token.json).
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds_path = Path(GMAIL_CREDENTIALS_FILE)
    if not creds_path.exists():
        print(
            f"❌ Credentials file not found at {creds_path}.\n\n"
            "To create one:\n"
            "  1. Go to https://console.cloud.google.com/apis/credentials\n"
            "  2. Create OAuth 2.0 Client ID → 'Desktop application'\n"
            "  3. Download the JSON and save it as the path above\n"
        )
        return

    flow = InstalledAppFlow.from_client_secrets_file(
        str(creds_path), GMAIL_SCOPES
    )

    print("Opening browser for Google OAuth...")
    creds = flow.run_local_server(port=0, open_browser=True)
    _save_token(creds)
    print(f"✅ OAuth complete. Token saved to {GMAIL_TOKEN_FILE}")


def do_oauth_setup_headless() -> None:
    """Headless OAuth — prints URL, user visits in their browser, pastes the
    authorization code or the full redirect URL.

    Google deprecated the old 'urn:ietf:wg:oauth:2.0:oob' flow.
    Instead we use 'http://localhost' as redirect_uri — when the user
    authorizes in their browser, Google redirects to a URL like:

        http://localhost/?code=...&scope=...

    The redirect will fail (no server running here), but the code is in
    the URL. The user copies either the code or the full URL and pastes
    it back.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds_path = Path(GMAIL_CREDENTIALS_FILE)
    if not creds_path.exists():
        print(
            f"❌ Credentials file not found at {creds_path}.\n\n"
            "To create one:\n"
            "  1. Go to https://console.cloud.google.com/apis/credentials\n"
            "  2. Create OAuth 2.0 Client ID → 'Desktop application'\n"
            "  3. Download the JSON and save it as the path above\n"
        )
        return

    flow = InstalledAppFlow.from_client_secrets_file(
        str(creds_path), GMAIL_SCOPES
    )

    # Use localhost redirect — Google redirects the browser to
    # http://localhost/?code=... after authorization.
    # The user copies the code (or full URL) from the address bar.
    flow.redirect_uri = "http://localhost"

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    print("\n" + "=" * 72)
    print("  GMAIL OAUTH — HEADLESS SETUP")
    print("=" * 72)
    print(f"\n  1. Open this URL in your browser:\n")
    print(f"     {auth_url}")
    print(f"\n  2. Authorize the app (use your Gmail account).")
    print(f"  3. After authorizing, Google will redirect to")
    print(f"     http://localhost/?code=... — that page will fail to load")
    print(f"     (no server running), but the CODE is in the URL.")
    print(f"  4. Copy the 'code=...' part from your browser's address bar")
    print(f"     (or paste the full redirect URL).\n")

    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    raw = input("  Authorization code (or full redirect URL): ").strip()

    # If user pasted the full redirect URL, extract the code
    if raw.startswith("http://") and "code=" in raw:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(raw)
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
    else:
        code = raw

    if not code:
        print("\n  ❌ No code provided. Run 'setup' again to retry.")
        return

    flow.fetch_token(code=code)
    creds = flow.credentials
    _save_token(creds)
    print(f"\n  ✅ OAuth complete. Token saved to {GMAIL_TOKEN_FILE}")


# ── Email fetching and parsing ──────────────────────────────────────────────


def list_alert_emails(
    service=None, max_results: int = 10
) -> list[dict]:
    """List Google Scholar Alert emails from the inbox.

    Returns list of dicts with keys: id, threadId, subject, date, snippet.
    """
    if service is None:
        service = _get_service()

    # Search for Google Scholar Alert emails
    results = (
        service.users()
        .messages()
        .list(userId="me", q=SCHOLAR_QUERY, maxResults=max_results)
        .execute()
    )

    messages = results.get("messages", [])
    if not messages:
        logger.info("No Google Scholar Alert emails found")
        return []

    # Fetch metadata for each message
    emails = []
    for msg in messages:
        full = (
            service.users()
            .messages()
            .get(userId="me", id=msg["id"], format="metadata")
            .execute()
        )

        headers = {h["name"]: h["value"] for h in full.get("payload", {}).get("headers", [])}
        emails.append(
            {
                "id": full["id"],
                "threadId": full.get("threadId", ""),
                "subject": headers.get("Subject", "(no subject)"),
                "date": headers.get("Date", ""),
                "from": headers.get("From", ""),
                "snippet": full.get("snippet", ""),
            }
        )

    return emails


def get_email_body(service, msg_id: str) -> str:
    """Get the full HTML body of an email message."""
    msg = service.users().messages().get(
        userId="me", id=msg_id, format="full"
    ).execute()

    def _extract_text(payload) -> str:
        """Recursively extract text from MIME parts."""
        mime_type = payload.get("mimeType", "")
        if mime_type == "text/html" and payload.get("body", {}).get("data"):
            import base64
            data = payload["body"]["data"]
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        if mime_type == "multipart/alternative" or mime_type == "multipart/mixed":
            parts = payload.get("parts", [])
            # Prefer HTML over plain text — iterate backwards (last = best)
            for part in reversed(parts):
                result = _extract_text(part)
                if result:
                    return result
            # Fallback: try all parts
            for part in parts:
                result = _extract_text(part)
                if result:
                    return result
        return ""

    return _extract_text(msg.get("payload", {}))


def extract_article_urls(html_body: str, subject: str = "") -> list[str]:
    """Extract article URLs from a Google Scholar Alert email body.

    Handles two formats:
      1. Google Scholar redirect URLs (scholar.google.com/scholar_url?url=...)
      2. Direct article URLs in the email body

    Returns a list of unique, clean article URLs.
    """
    urls = set()

    # Pattern 1: Google Scholar redirect URLs (most common in alerts)
    for match in _SCHOLAR_URL_PATTERN.finditer(html_body):
        raw = match.group(1)
        # URL-decode
        from urllib.parse import unquote
        url = unquote(raw)
        # Strip tracking parameters
        url = url.split("&")[0].split("?")[0] if "?" in url else url.split("&")[0]
        if url.startswith("http"):
            urls.add(url)

    # Pattern 2: Direct URLs in <a href> — skip if we already got scholar URLs
    if not urls:
        for match in _DIRECT_URL_PATTERN.finditer(html_body):
            url = match.group(1).rstrip("/.,;:!?)")
            # Filter out common non-article URLs
            if any(
                domain in url
                for domain in [
                    "google.com",
                    "scholar.google",
                    "accounts.google",
                    "support.google",
                    "youtube.com",
                ]
            ):
                continue
            if url.startswith("http"):
                urls.add(url)

    # Deduplicate and sort
    unique = sorted(urls)

    # Filter out known non-article URLs
    _SKIP_DOMAINS = {
        "scholar.google.com",
        "books.google.com",
        "accounts.google.com",
        "support.google.com",
        "youtube.com",
    }
    _SKIP_EXTENSIONS = {".gif", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".css", ".js"}
    filtered = []
    for url in unique:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        # Skip known non-article domains
        if any(d in domain for d in _SKIP_DOMAINS):
            continue
        # Skip image/media files
        if any(path.endswith(ext) for ext in _SKIP_EXTENSIONS):
            continue
        filtered.append(url)

    logger.info("Extracted %d unique article URLs from email", len(filtered))
    return filtered


# ── Email processed-tracking ────────────────────────────────────────────────


def _load_processed_ids() -> set[str]:
    """Load set of already-processed email IDs from JSON file."""
    path = Path(GMAIL_PROCESSED_FILE)
    if path.exists():
        with open(path) as f:
            return set(json.load(f))
    return set()


def _save_processed_ids(ids: set[str]) -> None:
    """Persist processed email IDs to JSON file."""
    path = Path(GMAIL_PROCESSED_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sorted(ids), f, indent=2)


# ── Main entry point for pipeline ───────────────────────────────────────────


def fetch_and_ingest(
    pipeline: "IngestionPipeline",
    max_emails: int = 10,
    max_links_per_email: int = 5,
    re_process: bool = False,
) -> dict:
    """Fetch Google Scholar Alert emails and ingest new article links.

    Args:
        pipeline: An IngestionPipeline instance.
        max_emails: Max alert emails to check.
        max_links_per_email: Max article links to ingest per email.
        re_process: If True, re-process already-seen emails.

    Returns:
        Summary dict with counts of ingested/skipped/errored URLs.
    """
    try:
        service = _get_service()
    except ConnectionError as e:
        return {"status": "error", "reason": str(e)}

    emails = list_alert_emails(service, max_results=max_emails)
    if not emails:
        return {"status": "ok", "emails_checked": 0, "urls_ingested": 0, "urls_skipped": 0, "urls_error": 0}

    processed = _load_processed_ids() if not re_process else set()
    summary = {
        "status": "ok",
        "emails_checked": len(emails),
        "urls_found": 0,
        "urls_ingested": 0,
        "urls_skipped": 0,
        "urls_error": 0,
        "results": [],
    }

    newly_processed = set(processed)

    for email in emails:
        email_id = email["id"]
        subject = email.get("subject", "")
        date = email.get("date", "")[:25] if email.get("date") else ""

        if email_id in processed:
            logger.info("Skipping already-processed email: %s", subject)
            continue

        logger.info("Processing alert: %s (%s)", subject, date)

        try:
            body = get_email_body(service, email_id)
        except Exception as e:
            logger.exception("Failed to fetch email body for %s", subject)
            summary["urls_error"] += 1
            continue

        if not body:
            logger.warning("No HTML body found in email: %s", subject)
            newly_processed.add(email_id)
            continue

        urls = extract_article_urls(body, subject)
        summary["urls_found"] += len(urls)

        email_results = {"subject": subject, "date": date, "urls": []}

        for url in urls[:max_links_per_email]:
            try:
                result = pipeline.ingest_url(url)
                if result["status"] == "ingested":
                    summary["urls_ingested"] += 1
                elif result["status"] == "skipped":
                    summary["urls_skipped"] += 1
                else:
                    summary["urls_error"] += 1
                email_results["urls"].append(
                    {"url": url, "status": result["status"], "title": result.get("title", "")}
                )
            except Exception as e:
                logger.exception("Failed to ingest URL %s", url)
                summary["urls_error"] += 1
                email_results["urls"].append(
                    {"url": url, "status": "error", "title": str(e)[:80]}
                )

        newly_processed.add(email_id)
        summary["results"].append(email_results)

    # Save processed email IDs
    _save_processed_ids(newly_processed)

    logger.info(
        "Gmail alerts digest: %d emails, %d URLs found, "
        "%d ingested, %d skipped, %d errors",
        summary["emails_checked"],
        summary["urls_found"],
        summary["urls_ingested"],
        summary["urls_skipped"],
        summary["urls_error"],
    )

    return summary


# ── CLI entry point ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        do_oauth_setup_headless()
    elif len(sys.argv) > 1 and sys.argv[1] == "setup-gui":
        do_oauth_setup()
    elif len(sys.argv) > 1 and sys.argv[1] == "list":
        emails = list_alert_emails()
        if not emails:
            print("No Google Scholar Alert emails found.")
        else:
            print(f"\n{'ID':<20} {'Date':<25} {'Subject'}")
            print("-" * 80)
            for e in emails:
                eid = e["id"][:18] + "…" if len(e["id"]) > 20 else e["id"]
                print(f"{eid:<20} {e['date'][:25]:<25} {e['subject'][:60]}")
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick test: fetch and print URLs from latest alert
        emails = list_alert_emails(max_results=1)
        if not emails:
            print("No emails found.")
        else:
            service = _get_service()
            body = get_email_body(service, emails[0]["id"])
            urls = extract_article_urls(body, emails[0].get("subject", ""))
            print(f"\nEmail: {emails[0]['subject']}")
            print(f"URLs found: {len(urls)}")
            for u in urls[:10]:
                print(f"  • {u}")
            if len(urls) > 10:
                print(f"  … and {len(urls) - 10} more")
    else:
        print("Usage:")
        print("  python3 -m quantmind.ingest.gmail_alerts setup      # Headless OAuth (server)")
        print("  python3 -m quantmind.ingest.gmail_alerts setup-gui  # GUI OAuth (desktop)")
        print("  python3 -m quantmind.ingest.gmail_alerts list       # List recent alerts")
        print("  python3 -m quantmind.ingest.gmail_alerts test       # Test fetch latest alert")
