#!/bin/zsh

# OpenCortex
# run.sh

# Prevent conflicts with system MongoDB
sudo systemctl disable mongodb

echo -e "\nInitializing OpenCortex...\n"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker for Fedora/Linux first."
    exit 1
else
    echo "Docker is installed."
fi

# Check for Docker Compose
if ! docker compose version &> /dev/null; then
    echo "Docker Compose is not installed or not working."
    exit 1
else
    echo "Docker Compose is ready."
fi

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Installing now..."
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Start the service just in case the installer didn't
    sudo systemctl enable --now ollama
else
    echo "Ollama is installed."
fi

# Ensure Ollama is running before pulling models
if ! systemctl is-active --quiet ollama; then
    echo "Ollama service is stopped. Starting it now..."
    sudo systemctl start ollama
fi

# Pull required AI Models
echo "Checking AI Models (This may take a moment if downloading)..."
ollama pull llama3.2:1b
ollama pull llama3.2-vision
ollama pull llama3.2
ollama pull nomic-embed-text
echo "AI Models are ready."

# Create the persistent ChromaDB folder
echo "Checking vector database directories..."
mkdir -p ./opencortex_db
chmod 777 ./opencortex_db

# Start the Full-Stack Application
echo -e "\nStarting OpenCortex Stack...\n"
docker compose up --build