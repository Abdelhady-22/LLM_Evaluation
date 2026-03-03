# ==============================================================================
# LLM Evaluation Framework — Dockerfile
# Base: NVIDIA CUDA 12.2 runtime for GPU access on Salad Cloud
# ==============================================================================

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11, pip, curl, zstd, and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install Ollama (zstd is required for extraction, installed above)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY shared/ ./shared/
COPY tasks/ ./tasks/
COPY config/ ./config/
COPY eval_structured_output.py .
COPY eval_long_context.py .
COPY run_eval.py .
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create output directories
RUN mkdir -p /app/results /app/logs

# Environment variables (defaults, overridden at runtime)
ENV EVAL_SCRIPT=all
ENV EVAL_CONFIG=/app/config/example_config.yaml
# HF_TOKEN and HUGGING_FACE_HUB_TOKEN: pass at runtime via
#   docker run -e HF_TOKEN=hf_xxx  OR  Salad Cloud env vars

ENTRYPOINT ["./entrypoint.sh"]
