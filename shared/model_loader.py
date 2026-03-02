"""
model_loader.py — Unified Model Loader

Supports two backends:
    - OllamaBackend: Calls Ollama REST API (/api/chat with stream: true)
    - HuggingFaceBackend: Loads via transformers + quant libraries (GPTQ, AWQ, bitsandbytes)

Exposes a UnifiedModel class with:
    - generate(messages, params) -> dict
    - Returns: response_text, input_tokens, output_tokens, ttft_ms, total_latency_ms
"""


class OllamaBackend:
    """Ollama REST API backend for GGUF models."""

    def __init__(self, config):
        # TODO: Implement
        pass

    def generate(self, messages, params):
        # TODO: Implement
        raise NotImplementedError


class HuggingFaceBackend:
    """HuggingFace Transformers backend for GPTQ, AWQ, bitsandbytes models."""

    def __init__(self, config):
        # TODO: Implement
        pass

    def generate(self, messages, params):
        # TODO: Implement
        raise NotImplementedError


class UnifiedModel:
    """Unified interface wrapping either Ollama or HuggingFace backend."""

    def __init__(self, config):
        # TODO: Implement — select backend based on config
        pass

    def generate(self, messages, params=None):
        # TODO: Implement
        raise NotImplementedError

    def get_model_info(self):
        # TODO: Implement — return model metadata dict
        raise NotImplementedError
