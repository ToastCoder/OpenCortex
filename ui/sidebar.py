# OpenCortex
# ui/sidebar.py — Sidebar controls: document management and model configuration.

import streamlit as st

import src.embeddings as embeddings
import src.ingestion.dispatcher as dispatcher
import src.rag.vectors as vectors
from src.config import get_params, set_vision_model
from src.ingestion.audio import check_audio_available


def render_sidebar():
    """Render the sidebar for an authenticated user."""
    # Session management
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.messages = []
        st.rerun()

    # File upload and sync
    uploaded_files = st.file_uploader(
        "Upload Documents, Images & Audio",
        type=["pdf", "txt", "png", "jpg", "jpeg", "mp3", "wav", "m4a", "ogg", "flac", "webm"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Sync"):
        any_ok = False
        for file in uploaded_files:
            ok, msg = dispatcher.process_uploaded_files([file], st.session_state.username)
            if ok:
                st.success(f"{file.name}: {msg}")
                any_ok = True
            else:
                st.error(f"{file.name}: {msg}")
        if any_ok:
            pass

    # Indexed document count
    doc_count = vectors.indexed_doc_count(st.session_state.username)
    st.caption(f"Indexed chunks: {doc_count}")

    if st.button("Clear Synced Documents"):
        vectors.clear_user_documents(st.session_state.username)
        st.success("Cleared all synced documents!")

    st.divider()
    st.subheader("Model Configuration")

    # Chat model selector
    params = get_params()
    chat_model_options = ["llama3.2:1b", "llama3.2", "llama3.2-vision:latest"]
    st.session_state.model_standard = st.selectbox(
        "Chat Model",
        options=chat_model_options,
        index=(
            chat_model_options.index(st.session_state.model_standard)
            if st.session_state.model_standard in chat_model_options
            else 0
        ),
    )

    # Vision model selector
    vision_model_options = ["moondream", "llama3.2-vision:latest"]
    current_vision = params["llm"]["vision_model"]
    selected_vision = st.selectbox(
        "Vision Model",
        options=vision_model_options,
        index=vision_model_options.index(current_vision) if current_vision in vision_model_options else 0,
    )
    set_vision_model(selected_vision)

    # Service health indicators
    audio_ok, audio_msg = check_audio_available()
    st.caption(f"Audio: {'[available]' if audio_ok else '[unavailable]'} {audio_msg}")

    embed_ok, embed_msg = embeddings.check_embeddings()
    if not embed_ok:
        st.caption(f"Embeddings: [unavailable] {embed_msg}")
