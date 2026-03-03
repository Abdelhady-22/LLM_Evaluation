"""
model_loader.py — Unified Model Loader

Two backends, one interface:
    - OllamaBackend: REST API with streaming for TTFT measurement
    - HuggingFaceBackend: transformers + quant libraries (GPTQ, AWQ, bitsandbytes)

UnifiedModel wraps either backend and exposes:
    generate(messages, params) -> dict with:
        response_text, input_tokens, output_tokens, ttft_ms, total_latency_ms
"""

import os
import time
import json
import requests


class OllamaBackend:
    """Ollama REST API backend. Uses streaming to measure TTFT."""

    def __init__(self, config):
        self.model_tag = config["tag"]
        self.base_url = config.get("ollama_url", "http://localhost:11434")

    def generate(self, messages, params=None):
        """Generate response using Ollama /api/chat with streaming."""
        params = params or {}

        payload = {
            "model": self.model_tag,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": params.get("temperature", 0.0),
                "num_predict": params.get("max_tokens", 2048),
                "top_p": params.get("top_p", 1.0),
            },
        }

        start_time = time.perf_counter()
        ttft_ms = None
        response_text = ""
        input_tokens = 0
        output_tokens = 0

        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines():
                    if not line:
                        continue

                    chunk = json.loads(line)

                    # Capture TTFT on first content chunk
                    if ttft_ms is None and chunk.get("message", {}).get("content"):
                        ttft_ms = (time.perf_counter() - start_time) * 1000

                    # Accumulate response text
                    content = chunk.get("message", {}).get("content", "")
                    response_text += content

                    # Final chunk contains token counts
                    if chunk.get("done", False):
                        input_tokens = chunk.get("prompt_eval_count", 0)
                        output_tokens = chunk.get("eval_count", 0)

        except requests.exceptions.Timeout:
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            return {
                "response_text": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "ttft_ms": ttft_ms,
                "total_latency_ms": total_latency_ms,
                "error": "timeout",
            }
        except Exception as e:
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            return {
                "response_text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "ttft_ms": None,
                "total_latency_ms": total_latency_ms,
                "error": str(e),
            }

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": round(ttft_ms, 2) if ttft_ms else None,
            "total_latency_ms": round(total_latency_ms, 2),
            "error": None,
        }

    def is_available(self):
        """Check if the model is available in Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return any(m.get("name", "").startswith(self.model_tag) for m in models)
        except Exception:
            pass
        return False


class HuggingFaceBackend:
    """HuggingFace Transformers backend for GPTQ, AWQ, bitsandbytes models."""

    def __init__(self, config):
        self.model_id = config.get("model_id", config.get("tag"))
        self.quant = config.get("quant_type", "FP32")
        self.bit_depth = config.get("bit_depth")
        self.model = None
        self.tokenizer = None
        self._load_model(config)

    def _load_model(self, config):
        """Load model with appropriate quantization."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # HF_TOKEN: required for gated models (e.g. Llama-3)
        # Reads from: env var HF_TOKEN > config field hf_token > None
        hf_token = os.environ.get("HF_TOKEN") or config.get("hf_token")
        if hf_token:
            print(f"[HF] Using HuggingFace token (len={len(hf_token)})")
        else:
            print("[HF] No HF_TOKEN set — only public models will work")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=hf_token
        )

        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "token": hf_token,
        }

        if self.quant == "GPTQ":
            from auto_gptq import AutoGPTQForCausalLM
            self.model = AutoGPTQForCausalLM.from_quantized(
                self.model_id, **load_kwargs
            )
        elif self.quant == "AWQ":
            from awq import AutoAWQForCausalLM
            self.model = AutoAWQForCausalLM.from_quantized(
                self.model_id, **load_kwargs
            )
        elif self.bit_depth == "INT4":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=bnb_config, **load_kwargs
            )
        elif self.bit_depth == "INT8":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=bnb_config, **load_kwargs
            )
        else:
            # FP32 / BF16 / FP16 — explicit dtype mapping
            dtype_map = {
                "FP32": torch.float32,
                "BF16": torch.bfloat16,
                "FP16": torch.float16,
            }
            dtype = dtype_map.get(self.quant, torch.float32)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=dtype, **load_kwargs
            )

    def generate(self, messages, params=None):
        """Generate response using HuggingFace transformers."""
        import torch
        params = params or {}

        # Format messages into prompt using chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=params.get("max_tokens", 2048),
                    temperature=params.get("temperature", 0.0) or 1e-7,
                    top_p=params.get("top_p", 1.0),
                    do_sample=params.get("temperature", 0.0) > 0,
                )

            total_latency_ms = (time.perf_counter() - start_time) * 1000

            # Decode only new tokens
            new_tokens = outputs[0][input_tokens:]
            response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            output_tokens = len(new_tokens)

            return {
                "response_text": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "ttft_ms": None,  # Not measurable in batch mode
                "total_latency_ms": round(total_latency_ms, 2),
                "error": None,
            }

        except Exception as e:
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            return {
                "response_text": "",
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "ttft_ms": None,
                "total_latency_ms": round(total_latency_ms, 2),
                "error": str(e),
            }


class UnifiedModel:
    """Unified interface wrapping either Ollama or HuggingFace backend."""

    def __init__(self, config):
        self.config = config
        self.backend_name = config.get("backend", "ollama")
        self.model_name = config.get("tag") or config.get("model_id", "unknown")
        self.quant_type = config.get("quant_type", "unknown")
        self.bit_depth = config.get("bit_depth")
        self.is_reference = config.get("is_reference", False)
        self._load_time_s = None

        # Initialize backend
        start = time.perf_counter()
        if self.backend_name == "ollama":
            self.backend = OllamaBackend(config)
        elif self.backend_name == "huggingface":
            self.backend = HuggingFaceBackend(config)
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")
        self._load_time_s = round(time.perf_counter() - start, 2)

    def generate(self, messages, params=None):
        """Generate response via the active backend."""
        result = self.backend.generate(messages, params)
        result["model_name"] = self.model_name
        result["backend"] = self.backend_name
        return result

    def get_model_info(self):
        """Return model metadata dict for JSON result."""
        return {
            "model_name": self.model_name,
            "quant_type": self.quant_type,
            "bit_depth": self.bit_depth,
            "backend": self.backend_name,
            "is_reference": self.is_reference,
            "load_time_s": self._load_time_s,
        }

    @property
    def load_time_s(self):
        return self._load_time_s
