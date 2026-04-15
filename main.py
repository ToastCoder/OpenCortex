# OpenCortex - main.py

# Required Libraries
import streamlit as st
import ollama

# Check for Ollama
def check_ollama():
    try:
        ollama.list()
        return True
    except:
        return False

if not check_ollama():
    st.error("OpenCortex cannot find Ollama. Please check for proper installation.")
    st.stop()

# System Prompt for AI to strictly follow
system_prompt = """
You are OpenCortex. Use ONLY the provided context to answer questions. 
If the answer is not in the context, strictly state that you do not know. 
Do not use outside knowledge.
"""

# OpenCortext Response Stream
def opencortex_response_stream(model_name, user_prompt, context):
    full_prompt = f"Context from Sources:\n{context}\n\nUser Question: {user_prompt}"

    full_message = [
        {"role":"system", "context": system_prompt},
        {"role":"user", "context": full_prompt}
    ]

    # Output stream the response
    for chunk in ollama.chat(model = model_name, message = full_message, stream = True):
        yield chunk['message']['context']


# Page Configuration
st.set_page_config(page_title="OpenCortex", layout="wide")

# Sidebar for Data Sources
with st.sidebar:
    st.header("Data Sources")
    st.markdown("Add different data sources to the local brain")

    # Upload files with different modalities
    uploaded_files = st.file_uploader("Upload Documents or Images",
                                      accept_multiple_files = True,
                                      type=['txt', 'pdf', 'png', 'jpg', 'jpeg', 'docx', 'doc', 'odt'])
    
    # Check files and output the number of uploaded files
    if uploaded_files:
        st.success(f"{len(uploaded_files)} files ready for indexing.")

        # Button for triggering the RAG pipeline
        if st.button("Sync to OpenCortex"):
            st.info("Preparing to index data sources...")


# Main Interface
st.title ("OpenCortex")
st.caption("Intelligence, but Open, Private and Local")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Input and Logic Loop
if prompt := st.chat_input("Ask OpenCortex..."):

    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder for AI Response
    with st.chat_message("assistant"):
        # We need to gather our 'context' string from our database/files first
        # For now, we'll use a placeholder variable
        context_placeholder = "The retrieved information will go here."
        
        # Call the function and write the stream to the UI
        full_response = st.write_stream(opencortex_response_stream("gemma4", prompt, context_placeholder))
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})

