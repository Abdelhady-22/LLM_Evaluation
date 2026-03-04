"""
status_server.py — Lightweight HTTP status server for Salad Cloud health checks.

Runs on port 8000 and serves a simple JSON status page showing evaluation progress.
This prevents Salad Cloud from showing 503 errors on the domain URL.
"""

import http.server
import json
import os
import threading
import glob


class StatusHandler(http.server.BaseHTTPRequestHandler):
    """Simple HTTP handler that returns evaluation status as JSON."""

    def do_GET(self):
        status = self._get_status()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(status, indent=2).encode())

    def do_HEAD(self):
        """Health check endpoint."""
        self.send_response(200)
        self.end_headers()

    def _get_status(self):
        results_dir = os.environ.get("RESULTS_DIR", "/app/results")
        results = glob.glob(os.path.join(results_dir, "*.json"))

        completed = []
        for r in sorted(results):
            try:
                with open(r) as f:
                    data = json.load(f)
                completed.append({
                    "task": data.get("test_meta", {}).get("task_name", "unknown"),
                    "model": data.get("model_config", {}).get("model_name", "unknown"),
                    "quant": data.get("model_config", {}).get("quant_type", ""),
                    "score": data.get("quality_scores", {}).get("overall_score"),
                    "passed": data.get("verdict", {}).get("passed"),
                })
            except Exception:
                pass

        return {
            "service": "LLM Evaluation Framework",
            "status": "running",
            "completed_tests": len(completed),
            "results": completed,
        }

    def log_message(self, format, *args):
        """Suppress HTTP request logs to keep container logs clean."""
        pass


def start_status_server(port=8000):
    """Start the status server in a background thread."""
    server = http.server.HTTPServer(("0.0.0.0", port), StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[STATUS] HTTP status server started on port {port}")
    return server
