# OpenCortex
# ui/auth.py — Login and registration UI.

import streamlit as st


def render_auth(auth):
    """Display the login / sign-up page for unauthenticated users."""
    st.title("Welcome to OpenCortex")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                success, msg = auth.verify_user(u, p)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.rerun()
                else:
                    st.error(msg)

    with tab2:
        with st.form("signup"):
            new_u = st.text_input("New Username")
            new_p = st.text_input("New Password", type="password")
            if st.form_submit_button("Register"):
                success, msg = auth.create_user(new_u, new_p)
                st.success(msg) if success else st.error(msg)
