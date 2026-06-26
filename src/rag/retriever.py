# OpenCortex
# src/rag/retriever.py — Context retrieval for RAG.
# Given a user's question, performs a similarity search over their indexed
# document chunks and returns the top-k results concatenated as a single string.

from langchain_community.vectorstores import Chroma

from src.config import get_params
from src.embeddings import EMBEDDING_MODEL
from src.rag.vectors import CHROMA_PATH
from utils.logger import setup_logger

logger = setup_logger("retriever")


def retrieve_context(user_prompt, username):
    """
    Query ChromaDB for the user's most relevant document chunks.
    Returns an empty string if the store is unreachable or the query fails.
    """
    try:
        params = get_params()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_MODEL)
        results = db.similarity_search(
            query=user_prompt,
            k=params["rag"]["k_neighbors"],
            filter={"user_id": username},
        )
        context = "\n\n".join([doc.page_content for doc in results])
        logger.info(
            f"Retrieved {len(results)} chunks ({len(context)} chars) for user {username}"
        )
        return context
    except Exception as e:
        logger.error(f"Retrieval failed for user {username}: {e}")
        return ""
