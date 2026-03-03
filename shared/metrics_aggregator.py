"""
metrics_aggregator.py — Metrics Aggregator

Computes aggregate statistics across a batch of evaluation runs:
    - Latency percentiles: p50, p95, p99
    - Token stats: avg_input_tokens, avg_output_tokens, total_tokens
    - GPU stats: avg_gpu_utilization, peak_vram_mb
    - Reliability: error_count, timeout_count, error_rate_percent
"""

import json
import os
import statistics
from datetime import datetime, timezone

BATCH_SUMMARY_START_TAG = "===BATCH_SUMMARY_START==="
BATCH_SUMMARY_END_TAG = "===BATCH_SUMMARY_END==="


def percentile(data, pct):
    """Calculate percentile value from a sorted list."""
    if not data:
        return None
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class MetricsAggregator:
    """Aggregates metrics across multiple evaluation runs."""

    def __init__(self):
        pass

    def aggregate(self, run_results):
        """Compute percentiles, averages, and error rates from run results."""
        if not run_results:
            return {"total_runs": 0}

        # Extract latency values
        latencies = []
        input_tokens_list = []
        output_tokens_list = []
        gpu_util_list = []
        vram_peak_list = []
        error_count = 0
        timeout_count = 0
        success_count = 0

        for result in run_results:
            perf = result.get("performance", {})
            hardware = result.get("hardware", {})
            verdict = result.get("verdict", {})
            logs = result.get("logs", {})

            # Latency
            latency = perf.get("total_latency_ms")
            if latency is not None:
                latencies.append(latency)

            # Tokens
            inp = perf.get("input_tokens") or result.get("input", {}).get("input_tokens")
            out = perf.get("output_tokens") or result.get("output", {}).get("output_tokens")
            if inp is not None:
                input_tokens_list.append(inp)
            if out is not None:
                output_tokens_list.append(out)

            # GPU
            gpu_util = hardware.get("gpu_utilization_avg_percent")
            if gpu_util is not None:
                gpu_util_list.append(gpu_util)
            vram_peak = hardware.get("vram_peak_mb")
            if vram_peak is not None:
                vram_peak_list.append(vram_peak)

            # Errors
            error_counts = logs.get("error_counts", {})
            error_count += error_counts.get("error_count", 0)
            timeout_count += error_counts.get("timeout_count", 0)

            if verdict.get("passed", False):
                success_count += 1

        total_runs = len(run_results)

        summary = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_runs": total_runs,
            "successful_runs": success_count,
            "failed_runs": total_runs - success_count,

            # Latency percentiles
            "p50_latency_ms": round(percentile(latencies, 50), 2) if latencies else None,
            "p95_latency_ms": round(percentile(latencies, 95), 2) if latencies else None,
            "p99_latency_ms": round(percentile(latencies, 99), 2) if latencies else None,
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
            "min_latency_ms": round(min(latencies), 2) if latencies else None,
            "max_latency_ms": round(max(latencies), 2) if latencies else None,

            # Token stats
            "avg_input_tokens": round(statistics.mean(input_tokens_list), 1) if input_tokens_list else None,
            "avg_output_tokens": round(statistics.mean(output_tokens_list), 1) if output_tokens_list else None,
            "total_tokens": sum(input_tokens_list) + sum(output_tokens_list),

            # GPU stats
            "avg_gpu_utilization_percent": round(statistics.mean(gpu_util_list), 1) if gpu_util_list else None,
            "peak_vram_mb": max(vram_peak_list) if vram_peak_list else None,

            # Reliability
            "error_count": error_count,
            "timeout_count": timeout_count,
            "error_rate_percent": round((error_count / total_runs) * 100, 2) if total_runs > 0 else 0,
            "success_rate_percent": round((success_count / total_runs) * 100, 2) if total_runs > 0 else 0,
        }

        return summary

    def save_and_output(self, summary, results_dir):
        """Save batch_summary.json and print to stdout with delimiters."""
        filepath = os.path.join(results_dir, "batch_summary.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Print to stdout for Salad Cloud capture
        print(f"\n{BATCH_SUMMARY_START_TAG}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"{BATCH_SUMMARY_END_TAG}\n", flush=True)

        return filepath
