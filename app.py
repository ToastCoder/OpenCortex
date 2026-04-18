# app.py
import streamlit as st
import src.core as core
from src.database import MongoManager
from utils.logger import setup_logger
import warnings
import os

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
        st.divider()
        uploaded_files = st.file_uploader(
                                            "Upload Documents & Images",
                                            type=["pdf", "txt", "png", "jpg", "jpeg"], 
                                            accept_multiple_files=True
                                            )

        # Sync button
        if uploaded_files and st.button("Sync"):
            core.process_uploaded_files(uploaded_files, st.session_state.username)
            st.success("Synced!")

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

            # Stream response
            full_response = st.write_stream(core.opencortex_response_stream("llama3.2", prompt, context))
            
            # Save assistant message
            db.save_message(st.session_state.username, "assistant", full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
