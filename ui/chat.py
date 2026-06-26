# OpenCortex
# ui/chat.py — Chat message display, input, and streaming response.

import streamlit as st

import src.llm as llm
import src.rag.retriever as retriever
from utils.logger import setup_logger

logger = setup_logger("chat_ui")


def render_chat(chat_history):
    """Render the conversation interface for an authenticated user."""
    # Load history on first render
    if "messages" not in st.session_state:
        st.session_state.messages = chat_history.get_history(st.session_state.username)

    # Display all previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask OpenCortex..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        chat_history.save_message(st.session_state.username, "user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            context = retriever.retrieve_context(prompt, st.session_state.username)
            logger.info(f"RETRIEVED CONTEXT: {context}")

            full_response = st.write_stream(
                llm.opencortex_response_stream(
                    st.session_state.model_standard, prompt, context
                )
            )

            chat_history.save_message(st.session_state.username, "assistant", full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
