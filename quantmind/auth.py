"""Authentication module for QuantMind Streamlit UI.

Simple password-based login with session state.
Call `check_auth()` at the top of every page to enforce login.
"""

import streamlit as st

from quantmind.config import QUANTMIND_PASSWORD


def check_auth() -> bool:
    """Enforce authentication — shows a login wall if not authenticated.

    Call at the very top of every page (before any other Streamlit commands).
    If not authenticated, renders a login form and calls ``st.stop()``.
    Returns True if the user is authenticated and page rendering should continue.
    """
    if st.session_state.get("authenticated"):
        return True

    # ── Login wall ────────────────────────────────────────────────────────
    st.set_page_config(page_title="Login — QuantMind", page_icon="🔐")
    st.markdown(
        """
        <style>
        .login-container { max-width: 380px; margin: 10vh auto 0; text-align: center; }
        </style>
        <div class="login-container">
            <h1>🔐 QuantMind</h1>
            <p style="color: #666; margin-bottom: 2rem;">
                Enter the password to access the application.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 2, 1])
    with col:
        with st.form("login_form"):
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password…",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        if submitted:
            if password == QUANTMIND_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")

    st.stop()
    return False  # never reached


def logout():
    """Clear authentication and return to login."""
    for key in ("authenticated",):
        st.session_state.pop(key, None)
    st.rerun()
