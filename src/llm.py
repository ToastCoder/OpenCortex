# OpenCortex
# src/llm.py — Ollama LLM client and response generator.
# Wraps the Ollama HTTP API with health-check and a streaming generator
# that injects RAG context into prompt templates loaded from config.

import os

from ollama import Client

from src.config import get_params, get_prompts
from utils.logger import setup_logger

logger = setup_logger("llm")

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
ollama_client = Client(host=OLLAMA_URL)


def check_ollama():
    """Verify the Ollama service is reachable by listing installed models."""
    try:
        ollama_client.list()
        logger.info("Successfully connected to Ollama service.")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        return False


def opencortex_response_stream(model_name, user_prompt, context):
    """
    Yield tokens from the LLM as they arrive (streaming).

    Injects the retrieved document context into the RAG prompt template,
    then sends the full message array to Ollama's chat endpoint.
    """
    params = get_params()
    prompts = get_prompts()

    # Fill the RAG template with context and the user's question
    full_prompt = prompts["rag_template"].replace(
        "{context}", context
    ).replace("{user_query}", user_prompt)

    messages = [
        {"role": "system", "content": prompts["system_message"]},
        {"role": "user", "content": full_prompt},
    ]

    try:
        for chunk in ollama_client.chat(
            model=model_name,
            messages=messages,
            stream=True,
            options={
                "temperature": params["llm"]["temperature"],
                "num_predict": params["llm"]["max_tokens"],
                "num_ctx": 32768,
            },
            keep_alive=0,
        ):
            yield chunk["message"]["content"]
    except Exception as e:
        logger.error(f"Error during AI streaming: {e}")
        yield "I encountered an error while trying to process that."
