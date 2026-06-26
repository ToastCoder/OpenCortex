# OpenCortex
# src/ingestion/dispatcher.py — File ingestion orchestrator.
# Routes uploaded files to the correct extractor by extension, then chunks
# and indexes the extracted text into the vector store.

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_params
from src.ingestion.audio import process_audio
from src.ingestion.image import process_image_vision
from src.ingestion.pdf import extract_pdf_text_and_images
from src.ingestion.text import process_text_file
from src.rag.vectors import add_texts
from utils.logger import setup_logger

logger = setup_logger("dispatcher")

# Supported file types and their corresponding extraction paths
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}


def _file_ext(name):
    """Return the lowercase extension of a filename."""
    _, ext = os.path.splitext(name)
    return ext.lower()


def _extract_text(file, username):
    """
    Route a single uploaded file to the correct extractor based on extension.
    Returns the extracted text content, or '' if the file type is unsupported.
    """
    ext = _file_ext(file.name)
    if ext == ".pdf":
        return extract_pdf_text_and_images(file.getvalue(), username)
    if ext == ".txt":
        return process_text_file(file.getvalue().decode("utf-8"))
    if ext in IMAGE_EXTS:
        return process_image_vision(file.getvalue(), f"{username}_{file.name}")
    if ext in AUDIO_EXTS:
        return process_audio(file.getvalue(), f"{username}_{file.name}")
    return ""


def process_uploaded_files(files, username):
    """
    Extract, chunk, and index all uploaded files for a user.

    Each file is extracted individually; chunks are batched and written to
    ChromaDB in a single transaction. Returns (success: bool, message: str).
    """
    params = get_params()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=params["rag"]["chunk_size"],
        chunk_overlap=params["rag"]["chunk_overlap"],
    )

    all_chunks = []
    all_metadatas = []

    for file in files:
        content = _extract_text(file, username)
        if not content.strip():
            logger.warning(f"Skipping {file.name} — no extractable content")
            continue

        chunks = text_splitter.split_text(content)
        all_chunks.extend(chunks)
        all_metadatas.extend(
            {"user_id": username, "source": file.name} for _ in chunks
        )
        logger.info(f"{file.name}: {len(chunks)} chunks")

    if not all_chunks:
        return False, "No extractable content found"

    try:
        add_texts(texts=all_chunks, metadatas=all_metadatas)
        n = len(all_chunks)
        logger.info(f"Indexed {n} chunks for user {username}.")
        return True, f"Indexed {n} chunks"
    except Exception as e:
        logger.error(f"Failed to add texts to ChromaDB: {e}")
        return False, str(e)
