"""
metrics_aggregator.py — Metrics Aggregator

Computes aggregate statistics across a batch of evaluation runs:
    - Latency percentiles: p50, p95, p99
    - Token stats: avg_input_tokens, avg_output_tokens, total_tokens
    - GPU stats: avg_gpu_utilization, peak_vram_mb
    - Reliability: error_count, timeout_count, error_rate_percent, success_rate_percent

Outputs batch_summary.json and prints to stdout with delimiter tags.
"""


class MetricsAggregator:
    """Aggregates metrics across multiple evaluation runs."""

    def __init__(self):
        # TODO: Implement
        pass

    def aggregate(self, run_results):
        # TODO: Implement — compute percentiles, averages, error rates
        raise NotImplementedError

    def save_and_output(self, summary, results_dir):
        # TODO: Implement — save batch_summary.json + print to stdout
        raise NotImplementedError
