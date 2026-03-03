"""
live_logger.py — Structured JSON Logger

Dual-output logger: structured JSON to .jsonl file + human-readable to terminal.
Flush-on-every-write for Salad Cloud resilience (node can die anytime).
Tracks error counts and timeout counts as running totals.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone


class LiveLogger:
    """Structured JSON logger with terminal + file output."""

    def __init__(self, run_name, logs_dir):
        self.run_name = run_name
        self.logs_dir = logs_dir
        self._key_events = []
        self._error_count = 0
        self._timeout_count = 0

        # Create logs directory
        os.makedirs(logs_dir, exist_ok=True)

        # Create log file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = run_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        self.log_file_path = os.path.join(logs_dir, f"{safe_name}_{timestamp}.jsonl")
        self._file = open(self.log_file_path, "a", encoding="utf-8")

    def log(self, event, detail=None, **kwargs):
        """Write a structured JSON log entry to file + human-readable to terminal."""
        now = datetime.now(timezone.utc)
        entry = {
            "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event": event,
        }

        if detail:
            entry["detail"] = detail

        entry.update(kwargs)

        # Track special events
        if event == "ERROR":
            self._error_count += 1
        elif event == "TIMEOUT":
            self._timeout_count += 1

        # Store key events for JSON result inclusion
        self._key_events.append({
            "time": now.strftime("%H:%M:%S"),
            "event": event,
            "detail": detail,
        })

        # Write JSON line to file (flush immediately)
        json_line = json.dumps(entry, ensure_ascii=False)
        self._file.write(json_line + "\n")
        self._file.flush()

        # Write human-readable to terminal
        terminal_line = self._format_terminal(now, event, detail, kwargs)
        print(terminal_line, flush=True)

    def _format_terminal(self, timestamp, event, detail, kwargs):
        """Format a log entry for human-readable terminal output."""
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        parts = [f"{time_str} | [{event}]"]

        if detail:
            parts.append(detail)

        # Add key-value pairs
        for k, v in kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.1f}")
            else:
                parts.append(f"{k}: {v}")

        return " | ".join(parts)

    def get_key_events(self):
        """Return list of key events for inclusion in JSON result."""
        return list(self._key_events)

    def get_error_counts(self):
        """Return error and timeout counts."""
        return {
            "error_count": self._error_count,
            "timeout_count": self._timeout_count,
        }

    def close(self):
        """Flush and close log file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
