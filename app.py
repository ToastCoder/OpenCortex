# OpenCortex
# app.py (image scaling and context constraint)

# Importing Libraries 
import os
import warnings

import streamlit as st

import src.core as core
from src.database import MongoManager
from utils.logger import setup_logger

# Ignore Transformer Warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Catch any leftover Python-level warnings
warnings.filterwarnings("ignore")

logger = setup_logger("app_ui")
st.set_page_config(page_title="OpenCortex", layout="wide")


@st.cache_resource
def init_db():
    return MongoManager()


db = init_db()

# Initialize session state for user
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if "model_standard" not in st.session_state:
    st.session_state.model_standard = core.PARAMS["llm"]["model_standard"]

if not st.session_state.logged_in:
    # Login / Signup UI
    st.title("Welcome to OpenCortex")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # Login Tab
    with tab1:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")

            # Login button
            if st.form_submit_button("Login"):
                success, msg = db.verify_user(u, p)

                # Login success
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.rerun()

                # Login failed
                else:
                    st.error(msg)

    # Signup Tab
    with tab2:
        with st.form("signup"):
            new_u = st.text_input("New Username")
            new_p = st.text_input("New Password", type="password")

            # Signup button
            if st.form_submit_button("Register"):
                success, msg = db.create_user(new_u, new_p)
                st.success(msg) if success else st.error(msg)

# User is logged in
else:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = db.get_history(st.session_state.username)

    # Main UI
    st.title("OpenCortex")
    st.caption(f"Authenticated as: {st.session_state.username}")

    # Sidebar
    with st.sidebar:
        # Logout button
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.messages = []
            st.rerun()

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents, Images & Audio",
            type=["pdf", "txt", "png", "jpg", "jpeg", "mp3", "wav", "m4a", "ogg", "flac", "webm"],
            accept_multiple_files=True,
        )

        # Sync button
        if uploaded_files and st.button("Sync"):
            any_ok = False
            for file in uploaded_files:
                ok, msg = core.process_uploaded_files([file], st.session_state.username)
                if ok:
                    st.success(f"{file.name}: {msg}")
                    any_ok = True
                else:
                    st.error(f"{file.name}: {msg}")
            if any_ok:
                st.balloons()

        doc_count = core.indexed_doc_count(st.session_state.username)
        st.caption(f":open_file_folder: Indexed chunks: {doc_count}")

        # Clear button
        if st.button("Clear Synced Documents"):
            core.clear_user_documents(st.session_state.username)
            st.success("Cleared all synced documents!")

        st.divider()
        st.subheader("Model Configuration")
        
        chat_model_options = ["llama3.2:1b", "llama3.2", "llama3.2-vision:latest"]
        st.session_state.model_standard = st.selectbox(
            "Chat Model",
            options=chat_model_options,
            index=chat_model_options.index(st.session_state.model_standard) if st.session_state.model_standard in chat_model_options else 0
        )
        
        vision_model_options = ["moondream", "llama3.2-vision:latest"]
        current_vision = core.PARAMS["llm"]["vision_model"]
        selected_vision = st.selectbox(
            "Vision Model",
            options=vision_model_options,
            index=vision_model_options.index(current_vision) if current_vision in vision_model_options else 0
        )
        core.PARAMS["llm"]["vision_model"] = selected_vision

        audio_ok, audio_msg = core.check_audio_available()
        st.caption(f":microphone: Audio: {'✅' if audio_ok else '❌'} {audio_msg}")

        embed_ok, embed_msg = core.check_embeddings()
        if not embed_ok:
            st.caption(f":warning: Embeddings: ❌ {embed_msg}")

    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask OpenCortex..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Save user message
        db.save_message(st.session_state.username, "user", prompt)

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant message
        with st.chat_message("assistant"):
            # Retrieve context
            context = core.retrieve_context(prompt, st.session_state.username)

            logger.info(f"RETRIEVED CONTEXT: {context}")

            # Stream response
            full_response = st.write_stream(
                core.opencortex_response_stream(
                    st.session_state.model_standard, prompt, context
                )
            )

            # Save assistant message
            db.save_message(st.session_state.username, "assistant", full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
