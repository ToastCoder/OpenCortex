# OpenCortex

OpenCortex is a private, **multimodal**, local-first document intelligence platform designed to function as a self-hosted alternative to NotebookLM. It enables users to interact with their own data—including complex text documents and visual media—using Large Language Models (LLMs) without sending sensitive information to the cloud, ensuring absolute data privacy by keeping all processing on-device.

## Key Features

* **Multimodal Intelligence**: Unlike standard RAG systems, OpenCortex treats images and text as equal citizens. It can "see" your screenshots, diagrams, and handwritten notes, converting visual spatial relationships into searchable semantic data.
* **Hardware-Optimized Multimodal Vision**: OpenCortex utilizes a high-fidelity vision engine (`llama3.2-vision:latest` or `moondream`) to transcribe screenshots, code, layouts, and diagrams verbatim.
* **Memory Lifecycle Management**: Enforces strict VRAM/RAM optimization using immediate model unloading (`keep_alive=0`) and automatic image downscaling (restricting max dimensions to 1024px). This minimizes memory footprint and prevents Out-of-Memory (OOM) crashes on consumer hardware.
* **Dynamic Model Configuration**: Select your preferred Chat Model (e.g. `llama3.2:1b`, `llama3.2:latest`, `llama3.2-vision:latest`) and Vision Model directly from the Streamlit sidebar.
* **Privacy-First Local RAG**: All data is indexed into a local ChromaDB vector store. No API keys are required, and no telemetry is sent to third parties.
* **User & Database Control**: Cleanly reset session history and wipe synced document embeddings from the vector database using sidebar controls.
* **Prompt Injection Defense**: System prompts and RAG templates are hardened with explicit guardrails against prompt injection attacks, including instruction isolation in `<context>` delimiters and directives to ignore embedded instructions.

OpenCortex is entirely modular. By editing the JSON files in the `/config` folder, you can change the platform's behavior.

### `parameters.json` (LLM Parameters)
* **Model Selection**: Switch default chat and vision models. Supports standard, quantized, and vision-capable LLMs.
* **Inference Settings**: Fine-tune `temperature` for creativity vs. accuracy and `max_tokens` for response length.
* **RAG Tuning**: Adjust `k_neighbors` (how many document chunks are retrieved, optimized to 15) and `chunk_size` to balance context depth against memory constraints.

### `prompts.json` (LLM System Prompts and Template)
* **System Persona**: Define strict instructions for how the AI should respond. Personas are hardened to prevent hallucinations, strictly report missing context facts, and resist prompt injection by treating `<context>` content as untrusted data rather than executable instructions.
* **RAG Template**: Change how the AI formats the retrieved context, prioritizing strict citations and context boundaries. Includes explicit warnings that document content is untrusted, preventing embedded instructions from overriding the system persona.
* **Vision Prompt**: Controls how images are transcribed. Hardened to treat any embedded text instructions as data to transcribe, not as commands to follow.

## Code structure

```
OpenCortex/
├── 📂 config/               JSON configuration (parameters, prompts)
├── 📂 opencortex_db/        Persistent ChromaDB vector storage
├── 📂 src/                  Application logic
│   ├── 📄 auth.py           User authentication (login, signup)
│   ├── 📄 chat_history.py   Message persistence & retrieval
│   ├── 📄 config.py         Configuration loader for JSON files
│   ├── 📄 database.py       MongoDB connection manager
│   ├── 📄 embeddings.py     Embedding model initialisation
│   ├── 📄 llm.py            Ollama LLM client & streaming response generator
│   ├── 📂 ingestion/        File processing pipeline
│   │   ├── 📄 audio.py      Whisper-based audio transcription
│   │   ├── 📄 dispatcher.py Routes files to correct processor by extension
│   │   ├── 📄 image.py      Ollama Vision-based image description
│   │   ├── 📄 pdf.py        PDF text extraction with reading-order layout
│   │   └── 📄 text.py       Plain text file handler
│   └── 📂 rag/              Retrieval-Augmented Generation
│       ├── 📄 retriever.py  ChromaDB similarity search
│       └── 📄 vectors.py    Vector store path & initialisation
├── 📂 ui/                   Streamlit UI pages
│   ├── 📄 auth.py           Login / signup form
│   ├── 📄 chat.py           Chat interface with message history
│   └── 📄 sidebar.py        Model config, file upload, session controls
├── 📂 utils/                Helper functions
│   └── 📄 logger.py         Standardized system logging
├── 📄 app.py                Application entry point, wires services & routing
├── 📄 Dockerfile            Container build instructions (Python slim environment)
├── 📄 docker-compose.yml    Service orchestration (Web/MongoDB)
├── 📄 requirements.txt      Python dependencies
└── 📄 run.sh                Diagnostic & Deployment script
```

## Requirements
- **Docker & Docker Compose**
- **Ollama** (Running natively on the host machine)
- **Minimum Hardware**: 4GB VRAM (GPU) / 8GB System RAM.

---

## Quick Start (The "One-Click" Setup)

`run.sh` script handles everything from installing necessary software to managing the environment. Enter the following in terminal:

```bash
chmod +x run.sh
./run.sh
```
