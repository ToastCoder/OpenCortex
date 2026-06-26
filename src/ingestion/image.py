# OpenCortex
# src/ingestion/image.py — Vision-model-based image extraction.
# Sends image bytes to the configured Ollama vision model with a strict
# verbatim-transcription prompt, and wraps the result in a structured block.

import io

from ollama import Client
from PIL import Image

from src.config import get_params, get_prompts
from src.llm import OLLAMA_URL
from utils.logger import setup_logger

logger = setup_logger("image_extractor")
ollama_client = Client(host=OLLAMA_URL)


def process_image_vision(image_bytes, source_name, position=None):
    """
    Analyse an image with the vision LLM and return a structured description.

    Steps:
      1. Downscale images larger than max_dim (default 1024 px) to reduce VRAM.
      2. Send the (possibly resized) image to the vision model with zero temperature.
      3. Wrap the response in a [Visual Element] block for RAG indexing.
    """
    try:
        params = get_params()
        prompts = get_prompts()

        # Resize large images for faster inference on consumer hardware
        img = Image.open(io.BytesIO(image_bytes))
        max_dim = params.get("image", {}).get("max_dim", 1024)
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            out_bytes = io.BytesIO()
            img.save(out_bytes, format="PNG")
            image_bytes = out_bytes.getvalue()

        vision_response = ollama_client.chat(
            model=params["llm"]["vision_model"],
            messages=[
                {
                    "role": "user",
                    "content": prompts["vision_prompt"],
                    "images": [image_bytes],
                }
            ],
            options={
                "num_ctx": 4096,
                "temperature": 0.0,  # Deterministic output
            },
            keep_alive=0,
        )
        image_semantics = vision_response["message"]["content"]

        # Format as a delimited block so the RAG model can distinguish sources
        context_block = f"\n[Visual Element: {source_name}]\n"
        if position:
            context_block += f"Position: {position}\n"
        context_block += f"Description: {image_semantics}\n"
        context_block += f"End of visual analysis for {source_name}.\n"

        return context_block

    except Exception as e:
        logger.error(f"Vision Pipeline Error for {source_name}: {e}")
        return f"\n[Error processing visual element: {source_name}]\n"
