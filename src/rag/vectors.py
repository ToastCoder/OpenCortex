# OpenCortex
# src/rag/vectors.py — ChromaDB vector-store operations.
# Exposes a minimal CRUD surface: add chunks, count, and delete by user.
# The underlying store is a persisted Chroma (LangChain wrapper) database.

from langchain_community.vectorstores import Chroma

from src.embeddings import EMBEDDING_MODEL
from utils.logger import setup_logger

logger = setup_logger("vectors")

CHROMA_PATH = "./opencortex_db"


def _get_db():
    """Return a Chroma instance connected to the local persistence directory."""
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_MODEL)


def add_texts(texts, metadatas):
    """Index a batch of text chunks with per-chunk metadata."""
    db = _get_db()
    db.add_texts(texts=texts, metadatas=metadatas)


def indexed_doc_count(username):
    """Return the number of chunks stored for a given user."""
    try:
        db = _get_db()
        return db._collection.count(where={"user_id": username})
    except Exception:
        return 0


def clear_user_documents(username):
    """Delete every chunk belonging to the user from the vector store."""
    try:
        db = _get_db()
        db._collection.delete(where={"user_id": username})
        logger.info(f"Successfully cleared documents for user: {username}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear documents for user {username}: {e}")
        return False
