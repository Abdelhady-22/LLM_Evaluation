"""
json_builder.py — Result JSON Builder

Assembles the final result JSON from all collected components and saves to file.
Also prints the full result to stdout with delimiter tags for Salad Cloud capture.

Delimiter tags:
    ===RESULT_JSON_START===
    { ... full result JSON ... }
    ===RESULT_JSON_END===

Sections:
    test_meta, model_config, hardware, input, output,
    quality_scores, performance, logs, task_specific, verdict
"""


class JsonBuilder:
    """Assembles and saves evaluation result JSON files."""

    def __init__(self, results_dir, cost_per_hour_usd=None):
        # TODO: Implement
        pass

    def build_result(self, test_meta, model_config, hardware, input_data,
                     output_data, quality_scores, performance, logs,
                     task_specific, verdict):
        # TODO: Implement — assemble full result dict
        raise NotImplementedError

    def save_and_output(self, result):
        # TODO: Implement — save to file + print to stdout with delimiters
        raise NotImplementedError

    def _calculate_cost(self, total_latency_s):
        # TODO: Implement — (latency / 3600) * cost_per_hour_usd
        raise NotImplementedError
