# LLM Evaluation Framework — Shared Module
"""
Shared utilities for the LLM Evaluation Framework.

Modules:
    - model_loader: Unified model loading (Ollama + HuggingFace)
    - hardware_monitor: GPU/VRAM/RAM sampling and Salad Cloud detection
    - live_logger: Structured JSON logging (terminal + file)
    - json_builder: Result JSON assembly and stdout output
    - metrics_aggregator: Latency percentiles, token stats, error rates
    - run_tracker: Resume support for interrupted runs
"""
