# OpenCortex
# src/core.py

# Import Libraries
import io
import json
import os
import shutil
import subprocess
import tempfile

import fitz  # PyMuPDF
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client
from PIL import Image

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


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}


def _file_ext(name):
    _, ext = os.path.splitext(name)
    return ext.lower()


# Process uploaded files
def _extract_text(file, username):
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
    """Extract text, chunk, and save to ChromaDB — each file processed independently."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARAMS["rag"]["chunk_size"],
        chunk_overlap=PARAMS["rag"]["chunk_overlap"],
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
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_MODEL)
        db.add_texts(texts=all_chunks, metadatas=all_metadatas)
        n = len(all_chunks)
        logger.info(f"Indexed {n} chunks for user {username}.")
        return True, f"Indexed {n} chunks"
    except Exception as e:
        logger.error(f"Failed to add texts to ChromaDB: {e}")
        return False, str(e)


def indexed_doc_count(username):
    """Return number of chunks indexed for a user."""
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_MODEL
        )
        return db._collection.count(where={"user_id": username})
    except Exception:
        return 0


# Extract text from PDFs and process images
def extract_pdf_text_and_images(file_bytes, username):
    combined_text = ""
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    for page_index, page in enumerate(doc):
        combined_text += f"\n--- Page {page_index + 1} ---\n"

        blocks = page.get_text("dict", sort=True)["blocks"]
        text_blocks = [b for b in blocks if b["type"] == 0]

        # Scanned page detection
        if not text_blocks:
            pix = page.get_pixmap(dpi=200)
            combined_text += process_image_vision(
                pix.tobytes("png"),
                f"{username}_P{page_index + 1}_fullpage",
                position="full-page",
            )
            continue

        # Collect text items with position
        items = []
        for block in text_blocks:
            bbox = block.get("bbox")
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "") + " "
                text += "\n"
            items.append((bbox[1], bbox[0], "text", text))

        # Collect visible images with position
        processed_xrefs = set()
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in processed_xrefs:
                continue
            processed_xrefs.add(xref)

            rects = page.get_image_rects(xref)
            if not rects:
                continue

            rect = rects[0]
            base_image = doc.extract_image(xref)
            items.append((rect.y0, rect.x0, "image", base_image["image"]))

        # Sort all items top-to-bottom, left-to-right (reading order)
        items.sort(key=lambda x: (x[0], x[1]))

        # Process items in reading order
        img_counter = 0
        page_rect = page.rect
        for y, x, item_type, data in items:
            if item_type == "text":
                combined_text += data
            else:
                source = f"{username}_P{page_index + 1}_img{img_counter}"
                img_counter += 1

                y_pos = (
                    "top"
                    if y < page_rect.height / 3
                    else "bottom"
                    if y >= page_rect.height * 2 / 3
                    else "middle"
                )
                x_pos = (
                    "left"
                    if x < page_rect.width / 3
                    else "right"
                    if x >= page_rect.width * 2 / 3
                    else "center"
                )
                combined_text += process_image_vision(
                    data, source, position=f"{y_pos}-{x_pos}"
                )

    logger.info(f"PDF processed: {len(doc)} pages for user {username}")
    return combined_text


# Process text files
def process_text_file(file_content):
    return file_content


# Process image using Ollama vision model
def process_image_vision(image_bytes, source_name, position=None):
    """
    Handles the Vision pass for an image using only the LLM (Vision Model).
    Removes manual Tesseract OCR to save compute and simplify the pipeline.
    """
    try:
        # Resize image for faster processing on consumer hardware
        img = Image.open(io.BytesIO(image_bytes))
        max_dim = PARAMS.get("image", {}).get("max_dim", 1024)
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

        context_block = f"\n[Visual Element: {source_name}]\n"
        if position:
            context_block += f"Position: {position}\n"
        context_block += f"Description: {image_semantics}\n"
        context_block += f"End of visual analysis for {source_name}.\n"

        return context_block

    except Exception as e:
        logger.error(f"Vision Pipeline Error for {source_name}: {e}")
        return f"\n[Error processing visual element: {source_name}]\n"


# Check if whisper-cpp binary and model are available
def check_audio_available():
    if not shutil.which("whisper-cpp"):
        return False, "whisper-cpp binary not in container"
    model_path = "/models/ggml-tiny.bin"
    if not os.path.exists(model_path):
        return False, f"model not found at {model_path}"
    return True, "whisper-cpp ready"


# Process audio using whisper-cpp (subprocess to compiled binary)
def process_audio(audio_bytes, source_name):
    _, ext = os.path.splitext(source_name)
    raw_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(audio_bytes)
            raw_path = f.name

        wav_path = raw_path + ".wav"

        ffmpeg_proc = subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-ar", "16000", "-ac", "1",
             "-sample_fmt", "s16", wav_path],
            capture_output=True, text=True
        )
        if ffmpeg_proc.returncode != 0:
            logger.error(f"ffmpeg failed for {source_name}: {ffmpeg_proc.stderr}")
            return f"\n[Audio Element: {source_name} — ffmpeg conversion failed]\n"

        model_path = "/models/ggml-tiny.bin"
        whisper_proc = subprocess.run(
            ["whisper-cpp", "-f", wav_path, "-m", model_path,
             "-nt", "-ng"],
            capture_output=True, text=True
        )
        if whisper_proc.returncode != 0:
            logger.error(f"whisper-cpp failed for {source_name}: {whisper_proc.stderr}")
            return f"\n[Audio Element: {source_name} — transcription failed]\n"

        transcription = whisper_proc.stdout.strip()
        if not transcription:
            transcription = "[no speech detected]"

        logger.info(f"Transcribed {source_name} ({len(transcription)} chars)")

        return (
            f"\n[Audio Element: {source_name}]\n"
            f"Transcription: {transcription}\n"
            f"End of audio transcription for {source_name}.\n"
        )

    except Exception as e:
        logger.error(f"Audio Pipeline Error for {source_name}: {e}")
        return f"\n[Audio Element: {source_name} — error: {e}]\n"
    finally:
        for p in (raw_path, wav_path):
            if p is not None:
                try:
                    os.unlink(p)
                except Exception:
                    pass


# Retrieve context from ChromaDB
def retrieve_context(user_prompt, username):
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_MODEL)
        results = db.similarity_search(
            query=user_prompt, k=PARAMS["rag"]["k_neighbors"],
            filter={"user_id": username}
        )
        context = "\n\n".join([doc.page_content for doc in results])
        logger.info(f"Retrieved {len(results)} chunks ({len(context)} chars) for user {username}")
        return context
    except Exception as e:
        logger.error(f"Retrieval failed for user {username}: {e}")
        return ""


def check_embeddings():
    """Verify the embedding model is reachable."""
    try:
        EMBEDDING_MODEL.embed_query("test")
        return True, "embedding model ready"
    except Exception as e:
        return False, str(e)


# Generate response using the pre-retrieved context
def opencortex_response_stream(model_name, user_prompt, context):
    full_prompt = PROMPTS["rag_template"].replace(
        "{context}", context
    ).replace("{user_query}", user_prompt)

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
                "num_ctx": 32768,
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
