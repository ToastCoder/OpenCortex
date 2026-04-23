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

# Dual Pass Vision Function
def dual_pass_vision(image_bytes, source_name):
    """
    Handles both Semantic (Moondream) and Syntax (Tesseract) passes.
    Returns a formatted string ready for the RAG database.
    """
    logger.info(f"Running Dual-Pass Vision for: {source_name}")

    try:
        # Pass 1: Vision
        vision_response = ollama_client.chat(
                                            model=PARAMS['llm']['vision_model'],
                                            messages=[{
                                                'role': 'user',
                                                'content': PROMPTS['vision_prompt'],
                                                'images': [image_bytes]
                                            }],
                                            format='json'
        )
        image_semantics = vision_response['message']['content']
    
        img = Image.open(io.BytesIO(image_bytes))
        # Enhance image for better OCR
        gray_img = ImageOps.grayscale(img)
        enhancer = ImageEnhance.Contrast(gray_img)
        enhanced_img = enhancer.enhance(2.0)
        
        exact_text = pytesseract.image_to_string(enhanced_img)
        
        # Merge the findings
        context_block = f"\n[Visual Element: {source_name}]\n"
        context_block += f"Description: {image_semantics}\n"
        if exact_text.strip():
            context_block += f"Extracted Text/Code:\n{exact_text}\n"
        
        return context_block

    # Handle errors
    except Exception as e:
        logger.error(f"Vision Pipeline Error for {source_name}: {e}")
        return f"\n[Error processing visual element: {source_name}]\n"


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
            
            # Process each page
            for page_index, page in enumerate(doc):
                combined_text += f"\n--- Page {page_index + 1} ---\n"
                
                # Extract text and images sequentially using blocks, sorted for natural reading order
                blocks = page.get_text("dict", sort=True)["blocks"]
                img_index = 0
                
                for block in blocks:
                    # Text Block
                    if block["type"] == 0:
                        block_text = ""
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                block_text += span.get("text", "") + " "
                            block_text += "\n"
                        combined_text += block_text + "\n"
                        
                    # Image Block
                    elif block["type"] == 1:
                        # Filter out tiny images (like icons or separator lines)
                        if block.get("width", 0) < 50 or block.get("height", 0) < 50:
                            continue
                            
                        img_index += 1
                        image_bytes = block.get("image")
                        if image_bytes:
                            # Call the dual-pass vision pipeline
                            combined_text += dual_pass_vision(
                                image_bytes, 
                                f"{file.name}_P{page_index+1}_Img{img_index}"
                            )

        # Extract text from TXT files
        elif file.name.endswith(".txt"):

            # Decode the text file and add it to the combined text
            combined_text += file_bytes.decode("utf-8") + "\n"
            logger.info(f"Extracted text from {file.name}")
        
        # Process images using Moondream (Vision) + Tesseract (OCR)
        elif file.name.lower().endswith(('.png', '.jpg', '.jpeg','.webp')):
            combined_text += dual_pass_vision(file_bytes, file.name)

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