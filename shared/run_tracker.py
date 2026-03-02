"""
run_tracker.py — Run Tracker

Tracks completed evaluation runs for resume support on Salad Cloud.

Maintains run_manifest.json in the results directory:
    - Each entry: {model_tag, quant_type, script, task_name, status, result_file}
    - Before starting a run, checks if already completed -> skips if done
    - On startup, scans existing result files to rebuild manifest if needed
"""


class RunTracker:
    """Tracks completed runs for resume support."""

    def __init__(self, results_dir):
        # TODO: Implement — load or create run_manifest.json
        pass

    def is_completed(self, run_key):
        # TODO: Implement — check if run_key exists and completed
        raise NotImplementedError

    def mark_completed(self, run_key, result_path):
        # TODO: Implement — add to manifest, save immediately
        raise NotImplementedError

    def rebuild_from_results(self):
        # TODO: Implement — scan result files to rebuild manifest
        raise NotImplementedError

    def get_pending_runs(self, all_runs):
        # TODO: Implement — filter out completed runs from full list
        raise NotImplementedError
