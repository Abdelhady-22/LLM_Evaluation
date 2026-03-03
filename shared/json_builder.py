"""
json_builder.py — Result JSON Builder

Assembles the final result JSON from all collected components.
Saves to file and prints to stdout with delimiter tags for Salad Cloud capture.
"""

import json
import os
import uuid
from datetime import datetime, timezone


# Delimiter tags for stdout result capture
RESULT_START_TAG = "===RESULT_JSON_START==="
RESULT_END_TAG = "===RESULT_JSON_END==="


class JsonBuilder:
    """Assembles and saves evaluation result JSON files."""

    def __init__(self, results_dir, cost_per_hour_usd=None):
        self.results_dir = results_dir
        self.cost_per_hour_usd = cost_per_hour_usd
        os.makedirs(results_dir, exist_ok=True)

    def build_result(self, test_meta, model_config, hardware, input_data,
                     output_data, quality_scores, performance, logs,
                     task_specific, verdict):
        """Assemble full result dict from all components."""
        result = {
            "test_meta": {
                "test_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                **test_meta,
            },
            "model_config": model_config,
            "hardware": hardware,
            "input": input_data,
            "output": output_data,
            "quality_scores": quality_scores,
            "performance": {
                **performance,
                "estimated_cost_usd": self._calculate_cost(
                    performance.get("total_latency_ms", 0)
                ),
            },
            "logs": logs,
            "task_specific": task_specific,
            "verdict": verdict,
        }
        return result

    def save_and_output(self, result):
        """Save result to JSON file and print to stdout with delimiter tags."""
        # Build filename
        meta = result.get("test_meta", {})
        model = result.get("model_config", {})
        script_type = meta.get("script_type", "eval")
        model_name = model.get("model_name", "unknown").replace("/", "_").replace(":", "_")
        quant = model.get("quant_type", "unknown")
        timestamp = meta.get("timestamp", "").replace(":", "").replace("-", "")

        filename = f"{script_type}_{model_name}_{quant}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Print to stdout with delimiters (for Salad Cloud capture)
        print(f"\n{RESULT_START_TAG}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"{RESULT_END_TAG}\n", flush=True)

        return filepath

    def _calculate_cost(self, total_latency_ms):
        """Calculate estimated cost based on latency and hourly rate."""
        if self.cost_per_hour_usd is None or total_latency_ms is None:
            return None
        total_latency_s = total_latency_ms / 1000.0
        return round((total_latency_s / 3600.0) * self.cost_per_hour_usd, 6)
