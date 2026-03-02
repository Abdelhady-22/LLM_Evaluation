#!/bin/bash
set -e

# ==============================================================================
# LLM Evaluation Framework — Container Entrypoint
# Features: Ollama health-check, auto-restart, model pull retry, resume support
# ==============================================================================

echo "============================================"
echo " LLM Evaluation Framework"
echo " Starting on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================"

# --- Ollama Startup with Health Check ---
start_ollama() {
    echo "[INIT] Starting Ollama server..."
    ollama serve &
    OLLAMA_PID=$!

    # Poll until ready (max 30s)
    for i in $(seq 1 60); do
        if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "[INIT] Ollama is ready (PID: $OLLAMA_PID)"
            return 0
        fi
        sleep 0.5
    done

    echo "[ERROR] Ollama failed to start within 30 seconds"
    exit 1
}

# --- Health Check Function (called before each task) ---
check_ollama() {
    if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[WARNING] Ollama not responding, restarting..."
        kill $OLLAMA_PID 2>/dev/null || true
        sleep 2
        start_ollama
    fi
}
export -f check_ollama

# --- Model Pull with Retry ---
pull_model() {
    local tag=$1
    local max_attempts=3

    for attempt in $(seq 1 $max_attempts); do
        echo "[MODEL PULL] Pulling $tag (attempt $attempt/$max_attempts)..."
        if ollama pull "$tag"; then
            echo "[MODEL PULL] Successfully pulled $tag"
            return 0
        fi
        echo "[MODEL PULL] Failed to pull $tag, retrying in 5s..."
        sleep 5
    done

    echo "[ERROR] Failed to pull $tag after $max_attempts attempts"
    return 1
}

# --- Main Flow ---
start_ollama

# Pull models from config
# TODO: Parse EVAL_CONFIG to extract model tags and pull each one
echo "[INIT] Model pulling will be implemented with config parser"
echo "[INIT] Config: $EVAL_CONFIG"
echo "[INIT] Script: $EVAL_SCRIPT"

# Run evaluation (resume-aware — skips completed tasks)
echo "[EVAL] Starting evaluation..."
python run_eval.py --config "$EVAL_CONFIG" --script "$EVAL_SCRIPT"

echo "============================================"
echo " Evaluation complete"
echo " Results in /app/results/"
echo " Logs in /app/logs/"
echo "============================================"

# Keep alive for result retrieval
tail -f /dev/null
