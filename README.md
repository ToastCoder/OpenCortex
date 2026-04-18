# OpenCortex

OpenCortex is a private, **multimodal**, local-first document intelligence platform designed to function as a self-hosted alternative to NotebookLM. It enables users to interact with their own data—including complex text documents and visual media—using Large Language Models (LLMs) without sending sensitive information to the cloud, ensuring absolute data privacy by keeping all processing on-device.

## Key Features

* **Multimodal Intelligence**: Unlike standard RAG systems, OpenCortex treats images and text as equal citizens. It can "see" your screenshots, diagrams, and handwritten notes, converting visual spatial relationships into searchable semantic data.
* **Dual-Pass Vision**: Optimized for consumer-grade GPUs, OpenCortex splits visual processing into two high-fidelity streams:
    * **The Semantic Pass**: Uses the `moondream` (1.8B) model to describe the "big picture"—understanding layouts, UI elements, and diagram relationships.
    * **The Syntax Pass**: Utilizes `Tesseract OCR` on the CPU to extract pixel-perfect code blocks and text strings, ensuring the AI never "hallucinates" your variable names or technical data.
* **Privacy-First Local RAG**: All data is indexed into a local ChromaDB vector store. No API keys are required, and no telemetry is sent to third parties.
* **Hardware-Optimized Efficiency**: Designed to maximize performance on 4GB VRAM systems. It intelligently balances tasks between the GPU (for chat and vision) and the CPU (for OCR and embedding).
* **Cross-Platform Compatibility**: While OpenCortex is containerized for any OS, it is highly optimized for **Linux (Kubuntu/Ubuntu)** and **macOS** for the fastest local inference speeds.
  
OpenCortex is entirely modular. By editing the JSON files in the `/config` folder, you can change the platform's behavior.

### `parameters.json` (LLM Parameters)
* **Model Selection**: Switch between `llama3.2` (3B) for high intelligence or `llama3.2:1b` for lightning-fast responses on low-VRAM hardware.
* **Inference Settings**: Fine-tune `temperature` for creativity vs. accuracy and `max_tokens` for response length.
* **RAG Tuning**: Adjust `k_neighbors` (how many document chunks are retrieved) and `chunk_size` to optimize the context window for your specific GPU.

### `prompts.json` (LLM System Prompts and Template)
* **System Persona**: Define how the AI should act (e.g., "You are a professional SAP developer" or "You are a helpful study assistant").
* **RAG Template**: Change how the AI formats the retrieved context, allowing you to prioritize either strict citations or conversational flow.
* 
## Requirements
- **Docker & Docker Compose**
- **Ollama** (Running natively on the host machine)
- **Minimum Hardware**: 4GB VRAM (GPU) / 8GB System RAM.

---

## Quick Start (The "One-Click" Setup)

`setup.sh` script handles everything from installing necesary softwares to managing the environment. Enter the following in terminal.

```bash
chmod +x run.sh
./run.sh
```

