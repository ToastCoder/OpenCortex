# OpenCortex
# src/core.py

# Import Libraries
import io
import json
import os

import fitz  # PyMuPDF
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client
from PIL import Image, ImageEnhance, ImageOps

from utils.logger import setup_logger

# Setup Logger
logger = setup_logger("core_engine")

# Load Configurations
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
with open(os.path.join(CONFIG_DIR, "parameters.json")) as f:
    PARAMS = json.load(f)
with open(os.path.join(CONFIG_DIR, "prompts.json")) as f:
    PROMPTS = json.load(f)
    # Ensure that key templates expected to be strings are treated as such.
    for key, value in PROMPTS.items():
        if isinstance(value, list):
            PROMPTS[key] = "\n".join(value)

# Define the Ollama URL
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
ollama_client = Client(host=OLLAMA_URL)

# Define our database folder and embedding model
CHROMA_PATH = "./opencortex_db"
EMBEDDING_MODEL = OllamaEmbeddings(
    model=PARAMS["rag"]["embedding_model"], base_url=OLLAMA_URL
)


# Check if Ollama is reachable
def check_ollama():
    """Verify if the Ollama service is reachable."""

    # Try to connect to Ollama
    try:
        ollama_client.list()
        logger.info("Successfully connected to Ollama service.")
        return True

    # Handle connection errors
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        return False


# Process uploaded files
def process_uploaded_files(files, username):
    """Extract text, chunk it, and save to ChromaDB using Nomic embeddings."""

    # Initialize combined text
    combined_text = ""

    # Process each file
    for file in files:
        if file.name.endswith(".pdf"):
            combined_text += extract_pdf_text_and_images(file.getvalue(), username)
        elif file.name.endswith(".txt"):
            combined_text += process_text_file(file.getvalue().decode("utf-8"))
        elif file.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            combined_text += process_image_vision(
                file.getvalue(), f"{username}_P{0}_Img{0}"
            )

    # Split the text into manageable chunks
    if not combined_text.strip():
        return False

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARAMS["rag"]["chunk_size"],
        chunk_overlap=PARAMS["rag"]["chunk_overlap"],
    )

    chunks = text_splitter.split_text(combined_text)

    # Create metadata for each chunk
    metadatas = [{"user_id": username} for _ in chunks]

    # Embed the chunks with Nomic and store in ChromaDB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_MODEL)
    for chunk, meta in zip(chunks, metadatas):
        try:
            db.add_texts(texts=[chunk], metadatas=[meta])
        except Exception as e:
            logger.error(f"Failed to add text to ChromaDB: {e}")

    return True


# Extract text from PDFs and process images
def extract_pdf_text_and_images(file_bytes, username):
    combined_text = ""

    doc = fitz.open(stream=file_bytes, filetype="pdf")

    for page_index, page in enumerate(doc):
        combined_text += f"\n--- Page {page_index + 1} ---\n"

        # Extract text
        combined_text += extract_page_text(page)

        # Extract images and process them
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            combined_text += process_image_vision(
                image_bytes, f"{username}_P{page_index + 1}_Img{img_index}"
            )

    return combined_text


# Extract text from a single page
def extract_page_text(page):
    blocks = page.get_text("dict", sort=True)["blocks"]
    text_blocks = []

    for block in blocks:
        if block["type"] == 0:  # Text Block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_blocks.append(span.get("text", "") + " ")
                text_blocks.append("\n")

    return "".join(text_blocks)


# Process text files
def process_text_file(file_content):
    return file_content


# Process image using Ollama vision model
def process_image_vision(image_bytes, source_name):
    """
    Handles the Vision pass for an image using only the LLM (Gemma4/Vision Model).
    Removes manual Tesseract OCR to save compute and simplify the pipeline.
    """
    try:
        # Resize image for faster processing on consumer hardware
        img = Image.open(io.BytesIO(image_bytes))
        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            out_bytes = io.BytesIO()
            img.save(out_bytes, format="PNG")
            image_bytes = out_bytes.getvalue()

        # Pass 1: Vision (The single source of truth for visual content)
        vision_response = ollama_client.chat(
            model=PARAMS["llm"]["vision_model"],
            messages=[
                {
                    "role": "user",
                    "content": PROMPTS["vision_prompt"],
                    "images": [image_bytes],
                }
            ],
            options={
                "num_ctx": 4096,
                "temperature": 0.0,
            },
            keep_alive=0,
        )
        image_semantics = vision_response["message"]["content"]

        # We now rely completely on the LLM's JSON output for the description/text.
        context_block = f"\n[Visual Element: {source_name}]\n"
        context_block += f"Description: {image_semantics}\n"

        # If the model's output guarantees structure, we can add a general fallback:
        context_block += f"End of visual analysis for {source_name}.\n"

        return context_block

    except Exception as e:
        logger.error(f"Vision Pipeline Error for {source_name}: {e}")
        return f"\n[Error processing visual element: {source_name}]\n"


# Retrieve context from ChromaDB
def retrieve_context(user_prompt, username):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_MODEL)
    results = db.similarity_search(
        query=user_prompt, k=PARAMS["rag"]["k_neighbors"], filter={"user_id": username}
    )

    context = "\n\n".join([doc.page_content for doc in results])
    return context


# Generate response using the pre-retrieved context
def opencortex_response_stream(model_name, user_prompt, context):
    full_prompt = PROMPTS["rag_template"].format(
        context=context, user_query=user_prompt
    )

    messages = [
        {"role": "system", "content": PROMPTS["system_message"]},
        {"role": "user", "content": full_prompt},
    ]

    try:
        for chunk in ollama_client.chat(
            model=model_name,
            messages=messages,
            stream=True,
            options={
                "temperature": PARAMS["llm"]["temperature"],
                "num_predict": PARAMS["llm"]["max_tokens"],
            },
            keep_alive=0,
        ):
            yield chunk["message"]["content"]

    except Exception as e:
        logger.error(f"Error during AI streaming: {e}")
        yield "I encountered an error while trying to process that."


# Clear user documents from ChromaDB
def clear_user_documents(username):
    """Delete all documents for a given user from the Chroma vector store."""
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_MODEL)
        db._collection.delete(where={"user_id": username})
        logger.info(f"Successfully cleared documents for user: {username}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear documents for user {username}: {e}")
        return False
