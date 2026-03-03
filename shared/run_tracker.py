"""
run_tracker.py — Run Tracker

Tracks completed evaluation runs for resume support on Salad Cloud.
Maintains run_manifest.json in the results directory.
Before each run, checks if already completed and skips if done.
"""

import json
import os
import glob


class RunTracker:
    """Tracks completed runs for resume support."""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.manifest_path = os.path.join(results_dir, "run_manifest.json")
        self._manifest = {}

        os.makedirs(results_dir, exist_ok=True)
        self._load_manifest()

    def _load_manifest(self):
        """Load existing manifest or create empty one."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self._manifest = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._manifest = {}
        else:
            self._manifest = {}

    def _save_manifest(self):
        """Save manifest to disk immediately."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, ensure_ascii=False)

    @staticmethod
    def make_run_key(model_tag, script_type, task_name, **kwargs):
        """Create a unique key for a specific run configuration."""
        parts = [model_tag, script_type, task_name]
        for k, v in sorted(kwargs.items()):
            parts.append(f"{k}={v}")
        return "::".join(str(p) for p in parts)

    def is_completed(self, run_key):
        """Check if run_key exists and is marked as completed."""
        entry = self._manifest.get(run_key)
        if entry and entry.get("status") == "completed":
            # Also verify result file still exists
            result_path = entry.get("result_file")
            if result_path and os.path.exists(result_path):
                return True
        return False

    def mark_completed(self, run_key, result_path):
        """Mark a run as completed and save manifest immediately."""
        self._manifest[run_key] = {
            "status": "completed",
            "result_file": result_path,
        }
        self._save_manifest()

    def mark_failed(self, run_key, error):
        """Mark a run as failed."""
        self._manifest[run_key] = {
            "status": "failed",
            "error": str(error),
        }
        self._save_manifest()

    def rebuild_from_results(self):
        """Scan existing result files to rebuild manifest."""
        pattern = os.path.join(self.results_dir, "*.json")
        for filepath in glob.glob(pattern):
            if os.path.basename(filepath) in ("run_manifest.json", "batch_summary.json"):
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    result = json.load(f)

                meta = result.get("test_meta", {})
                model = result.get("model_config", {})
                task_specific = result.get("task_specific", {})

                run_key = self.make_run_key(
                    model.get("model_name", "unknown"),
                    meta.get("script_type", "unknown"),
                    meta.get("task_name", "unknown"),
                    level=task_specific.get("schema_complexity_level", ""),
                    doc_size=task_specific.get("doc_size", ""),
                    needle_depth=task_specific.get("needle_depth_percent", ""),
                )

                self._manifest[run_key] = {
                    "status": "completed",
                    "result_file": filepath,
                }
            except (json.JSONDecodeError, IOError, KeyError):
                continue

        self._save_manifest()

    def get_pending_runs(self, all_runs):
        """Filter out completed runs from full list. Returns list of pending run dicts."""
        pending = []
        for run in all_runs:
            run_key = run.get("run_key", "")
            if not self.is_completed(run_key):
                pending.append(run)
        return pending

    def get_completed_count(self):
        """Return count of completed runs."""
        return sum(1 for v in self._manifest.values() if v.get("status") == "completed")

    def get_total_count(self):
        """Return total tracked runs."""
        return len(self._manifest)
