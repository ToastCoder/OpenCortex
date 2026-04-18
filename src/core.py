# OpenCortex
# src/core.py

# Import Libraries
from ollama import Client
import os
import fitz # PyMuPDF
from utils.logger import setup_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import json
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
import io

# Setup Logger
logger = setup_logger("core_engine")

# Load Configurations
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
with open(os.path.join(CONFIG_DIR, 'parameters.json')) as f:
    PARAMS = json.load(f)
with open(os.path.join(CONFIG_DIR, 'prompts.json')) as f:
    PROMPTS = json.load(f)
    # Convert lists to multi-line strings for decluttered JSON
    for key in PROMPTS:
        if isinstance(PROMPTS[key], list):
            PROMPTS[key] = "\n".join(PROMPTS[key])

logger.info("Configuration files loaded successfully.")

# Define the Ollama URL
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
ollama_client = Client(host=OLLAMA_URL)

# Define our database folder and embedding model
CHROMA_PATH = "./opencortex_db"
EMBEDDING_MODEL = OllamaEmbeddings(
    model=PARAMS['rag']['embedding_model'],
    base_url=OLLAMA_URL
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
        file_bytes = file.getvalue() 

        # Extract text from PDF files
        if file.name.endswith(".pdf"):

            # Open the PDF file
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            # Extract text from each page
            for page in doc:
                combined_text += page.get_text() + "\n"
            logger.info(f"Extracted {len(doc)} pages from {file.name}")

        # Extract text from TXT files
        elif file.name.endswith(".txt"):

            # Decode the text file and add it to the combined text
            combined_text += file_bytes.decode("utf-8") + "\n"
            logger.info(f"Extracted text from {file.name}")
        
        # Process images using Moondream (Vision) + Tesseract (OCR)
        elif file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            logger.info(f"Running Dual-Pass extraction on {file.name}...")
            
            try:
                
                # Pass 1: Semantic Understanding (Moondream)
                vision_response = ollama_client.chat(
                    model=PARAMS['llm']['vision_model'],
                    messages=[{
                        'role': 'user',
                        'content': 'Analyze this image in high detail. If it is a flowchart or diagram, trace the logical flow and describe every step, decision (Yes/No branches), and outcome in order. Read all text within shapes and explain the relationships between them. Be as specific and accurate as possible in describing the process depicted.',
                        'images': [file_bytes]
                    }]
                )
                image_semantics = vision_response['message']['content']
                logger.info(f"Vision Semantics for {file.name}: {image_semantics}")

                # Pass 2: Exact Text Extraction (Tesseract)
                img = Image.open(io.BytesIO(file_bytes))
                
                # PRE-PROCESS: Upscale for better OCR accuracy
                w, h = img.size
                img = img.resize((w*2, h*2), Image.Resampling.LANCZOS)
                
                # Convert to Grayscale and boost contrast
                img = img.convert('L') 
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2.0) # Double the contrast
                
                # If you use Dark Mode, invert the colors so the background is white!
                # Uncomment the next line if your screenshots are dark mode:
                # img = ImageOps.invert(img)
                
                exact_text = pytesseract.image_to_string(img)
                logger.info(f"OCR Text for {file.name} (first 100 chars): {exact_text[:100]}...")
                

                combined_chunk = f"\n--- [Document Name: {file.name}] ---\n"
                combined_chunk += f"**Visual Description:** {image_semantics}\n\n"
                if exact_text.strip():
                    combined_chunk += f"**Exact Text/Code in Image:**\n{exact_text}\n"
                
                combined_text += combined_chunk + "\n"
                logger.info(f"Successfully merged Vision & OCR for {file.name}")
                
            except Exception as e:
                logger.error(f"Dual-Pass processing failed for {file.name}: {e}")

    # Check if combined text is empty
    if not combined_text.strip():
        return False

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARAMS['rag']['chunk_size'], 
        chunk_overlap=PARAMS['rag']['chunk_overlap']
    )

    chunks = text_splitter.split_text(combined_text)

    # Create metadata for each chunk
    metadatas = [{"user_id": username} for _ in chunks]

    # Embed the chunks with Nomic and store in ChromaDB
    logger.info(f"Embedding {len(chunks)} chunks into ChromaDB...")

    # Initialize ChromaDB and append new chunks
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=EMBEDDING_MODEL
    )
    db.add_texts(texts=chunks, metadatas=metadatas)

    return True

# Retrieve context from ChromaDB
def retrieve_context(user_prompt, username):
    """Search ChromaDB and return the raw text chunks."""

    # Retrieve the context from ChromaDB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_MODEL)
    results = db.similarity_search(
        query=user_prompt, 
        k=PARAMS['rag']['k_neighbors'],
        filter={"user_id": username}
    )

    # Combine the results into a single string
    context = "\n\n".join([doc.page_content for doc in results])
    return context

# Generate response using the pre-retrieved context
def opencortex_response_stream(model_name, user_prompt, context):
    """Stream the response using the pre-retrieved context with strict isolation."""
    
    # Apply Quantization Toggle Logic
    if PARAMS['llm']['use_quantization']:
        model_name = PARAMS['llm']['model_quantized']
    else:
        model_name = PARAMS['llm']['model_standard']

    full_prompt = PROMPTS['rag_template'].format(context=context, user_query=user_prompt)

    # Define the system prompt
    messages = [
        {"role": "system", "content": PROMPTS['system_message']},
        {"role": "user", "content": full_prompt}
    ]

    # Stream the response
    try:
        for chunk in ollama_client.chat(model=model_name,
                                        messages=messages, 
                                        stream=True,
                                        options={
                                            "temperature": PARAMS['llm']['temperature'],
                                            "num_predict": PARAMS['llm']['max_tokens']
                                        }
                                    ):
            yield chunk['message']['content']

    # Handle errors
    except Exception as e:
        logger.error(f"Error during AI streaming: {e}")
        yield "I encountered an error while trying to process that."