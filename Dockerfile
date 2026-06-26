# OpenCortex
# Dockerfile

# Start with a lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for some Python libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Build whisper.cpp (single C++ binary, no Python ML deps)
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp /tmp/whisper-cpp \
    && cmake -S /tmp/whisper-cpp -B /tmp/whisper-cpp/build \
    && cmake --build /tmp/whisper-cpp/build --config Release -j$(nproc) \
    && cp /tmp/whisper-cpp/build/bin/main /usr/local/bin/whisper-cpp \
    && rm -rf /tmp/whisper-cpp

# Download Whisper GGML model (~75 MB)
RUN mkdir -p /models && \
    curl -fSL -o /models/ggml-tiny.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

# Copy only the requirements first (to use Docker's cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py"]
