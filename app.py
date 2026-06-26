# OpenCortex
# app.py — Application entry point. Wires up services and delegates to UI components.

import os
import warnings

import streamlit as st

from src.auth import AuthManager
from src.chat_history import ChatHistory
from src.config import get_params
from src.database import MongoManager
from utils.logger import setup_logger

# Suppress noisy transformer and tokenizer logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

logger = setup_logger("app_ui")
st.set_page_config(page_title="OpenCortex", layout="wide")


@st.cache_resource
def init_db():
    """Initialise and cache the MongoDB connection across re-runs."""
    return MongoManager()


# Wire up service layer (inject the same db instance into all consumers)
db = init_db()
auth = AuthManager(db)
chat_history = ChatHistory(db)

# Initialise session defaults on first visit
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if "model_standard" not in st.session_state:
    st.session_state.model_standard = get_params()["llm"]["model_standard"]

# Render the correct page based on auth state
if not st.session_state.logged_in:
    from ui.auth import render_auth

    render_auth(auth)
else:
    from ui.chat import render_chat
    from ui.sidebar import render_sidebar

    st.title("OpenCortex")
    st.caption(f"Authenticated as: {st.session_state.username}")

    with st.sidebar:
        render_sidebar()

    render_chat(chat_history)
