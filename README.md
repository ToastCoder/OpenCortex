# OpenCortex

OpenCortex is a private, **multimodal**, local-first document intelligence platform designed to function as a self-hosted alternative to NotebookLM. It enables users to interact with their own data—including complex text documents and visual media—using Large Language Models (LLMs) without sending sensitive information to the cloud, ensuring absolute data privacy by keeping all processing on-device.

## Key Features

* **Multimodal Intelligence**: Unlike standard RAG systems, OpenCortex treats images and text as equal citizens. It can "see" your screenshots, diagrams, and handwritten notes, converting visual spatial relationships into searchable semantic data.
* **Hardware-Optimized Multimodal Vision**: OpenCortex utilizes a high-fidelity vision engine (`llama3.2-vision:latest` or `moondream`) to transcribe screenshots, code, layouts, and diagrams verbatim.
* **Memory Lifecycle Management**: Enforces strict VRAM/RAM optimization using immediate model unloading (`keep_alive=0`) and automatic image downscaling (restricting max dimensions to 1024px). This minimizes memory footprint and prevents Out-of-Memory (OOM) crashes on consumer hardware.
* **Dynamic Model Configuration**: Select your preferred Chat Model (e.g. `llama3.2:1b`, `llama3.2:latest`, `llama3.2-vision:latest`) and Vision Model directly from the Streamlit sidebar.
* **Privacy-First Local RAG**: All data is indexed into a local ChromaDB vector store. No API keys are required, and no telemetry is sent to third parties.
* **User & Database Control**: Cleanly reset session history and wipe synced document embeddings from the vector database using sidebar controls.

OpenCortex is entirely modular. By editing the JSON files in the `/config` folder, you can change the platform's behavior.

### `parameters.json` (LLM Parameters)
* **Model Selection**: Switch default chat and vision models. Supports standard, quantized, and vision-capable LLMs.
* **Inference Settings**: Fine-tune `temperature` for creativity vs. accuracy and `max_tokens` for response length.
* **RAG Tuning**: Adjust `k_neighbors` (how many document chunks are retrieved, optimized to 15) and `chunk_size` to balance context depth against memory constraints.

### `prompts.json` (LLM System Prompts and Template)
* **System Persona**: Define strict instructions for how the AI should respond. Personas are hardened to prevent hallucinations and strictly report missing context facts.
* **RAG Template**: Change how the AI formats the retrieved context, prioritizing strict citations and context boundaries.

## Code structure

```
OpenCortex/
├── 📂 config/               Logic & Persona settings
├── 📂 opencortex_db/        Persistent ChromaDB vector storage
├── 📂 src/                  Core functions
│   ├── 📄 core.py           Vision Engine, image processing & RAG pipeline
│   └── 📄 database.py       MongoDB connection management
├── 📂 utils/                Helper functions
│   └── 📄 logger.py         Standardized system logging
├── 📄 app.py                Streamlit UI & frontend code
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
