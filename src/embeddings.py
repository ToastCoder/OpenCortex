# OpenCortex
# src/embeddings.py — Embedding model singleton.
# Initialises the LangChain OllamaEmbeddings instance used by both the
# vector-store (for indexing) and the retriever (for query encoding).

from langchain_ollama import OllamaEmbeddings

from src.config import get_params
from src.llm import OLLAMA_URL
from utils.logger import setup_logger

logger = setup_logger("embeddings")

params = get_params()
EMBEDDING_MODEL = OllamaEmbeddings(
    model=params["rag"]["embedding_model"], base_url=OLLAMA_URL
)


def check_embeddings():
    """Validate the embedding model responds to a trivial query."""
    try:
        EMBEDDING_MODEL.embed_query("test")
        return True, "embedding model ready"
    except Exception as e:
        return False, str(e)
