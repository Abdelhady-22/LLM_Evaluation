"""
live_logger.py — Structured JSON Logger

Dual-output logger (terminal + .jsonl file), flush-on-every-write for Salad Cloud resilience.

JSON format per log line:
    {"ts": "...", "event": "MODEL_READY", "model": "...", "quant": "...", ...}

Event tags:
    INIT, MODEL_LOAD, MODEL_READY, TASK_START, GENERATING, FIRST_TOKEN,
    COMPLETE, SCORING, SAVED, ERROR, TIMEOUT

Tracks error_count and timeout_count as running totals.
"""


class LiveLogger:
    """Structured JSON logger with terminal + file output."""

    def __init__(self, run_name, logs_dir):
        # TODO: Implement — create log file, init counters
        pass

    def log(self, event, detail=None, **kwargs):
        # TODO: Implement — write JSON line to file + human-readable to terminal
        raise NotImplementedError

    def get_key_events(self):
        # TODO: Implement — return list of key events for JSON result
        raise NotImplementedError

    def get_error_counts(self):
        # TODO: Implement — return error_count, timeout_count
        raise NotImplementedError

    def close(self):
        # TODO: Implement — flush and close log file
        pass
