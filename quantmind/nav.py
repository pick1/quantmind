"""Shared sidebar nav for sequoiaanalytics.com tools.

Renders a single sidebar with links to all sister apps. Each tool
calls ``render_sidebar_nav()`` near the top of its entry script so
users see the same navigation no matter which app they're in.

Apps that fail a quick health probe are marked as offline (greyed
out) rather than removed — that way users see the full topology
even when a sister app is restarting.
"""

import streamlit as st

# All sister apps under the sequoiaanalytics.com tunnel
# (current_app is the label of THIS app, used for highlighting)
APPS = [
    ("Quanto", "https://quanto.sequoiaanalytics.com"),
    ("Distillate", "https://distillate.sequoiaanalytics.com"),
    ("LocalLens", "https://locallens.sequoiaanalytics.com"),
    ("Signal", "https://signal.sequoiaanalytics.com"),
    ("Signal-Strategies", "https://signal-strat.sequoiaanalytics.com"),
    ("Aqualytica", "https://aqualytica.sequoiaanalytics.com"),
    ("Watermark", "https://watermark.sequoiaanalytics.com"),
    ("Librarian", "https://librarian.sequoiaanalytics.com"),
]


@st.cache_data(ttl=60, show_spinner=False)
def _probe_apps() -> dict[str, bool]:
    """Quick HTTP probe of each sister app. Cached for 60s."""
    import urllib.request
    import urllib.error
    import socket

    socket.setdefaulttimeout(3)
    status = {}
    for label, url in APPS:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                status[label] = resp.status < 500
        except Exception:
            status[label] = False
    return status


def render_sidebar_nav(current_app: str = "") -> None:
    """Show a uniform sidebar nav linking to all sister apps.

    Args:
        current_app: Display label of the calling app (e.g., "Quanto").
                     It is rendered bold but not as a link.
    """
    with st.sidebar:
        st.markdown("### 🌐 Sequoia Apps")
        try:
            status = _probe_apps()
        except Exception:
            status = {}

        for label, url in APPS:
            if label == current_app:
                st.markdown(f"**• {label}** _(this app)_")
                continue
            online = status.get(label, True)
            if online:
                st.markdown(f"[{label}]({url})")
            else:
                st.markdown(
                    f"<span style='color:#666'>{label} (offline)</span>",
                    unsafe_allow_html=True,
                )
        st.markdown("---")
