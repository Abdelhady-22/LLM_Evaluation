"""
hardware_monitor.py — Hardware Monitor

Background thread that samples GPU/CPU/RAM every 500ms.

Auto-detects:
    - GPU model, VRAM total, CUDA version
    - CPU model, core count, system RAM
    - SALAD_NODE_ID environment variable

Tracks per task:
    - vram_before_load, vram_after_load, vram_peak_during_task, vram_task_delta
    - gpu_utilization (min/avg/max), gpu_temp_max
    - ram_used_mb
"""


class HardwareMonitor:
    """Monitors GPU, CPU, and RAM usage with background sampling."""

    def __init__(self):
        # TODO: Implement — init pynvml, detect hardware
        pass

    def get_system_info(self):
        # TODO: Implement — return static hardware info dict
        raise NotImplementedError

    def snapshot(self):
        # TODO: Implement — take a single VRAM/RAM reading
        raise NotImplementedError

    def start_monitoring(self):
        # TODO: Implement — start background sampling thread
        raise NotImplementedError

    def stop_monitoring(self):
        # TODO: Implement — stop background thread, return stats
        raise NotImplementedError

    def get_report(self):
        # TODO: Implement — return full hardware report dict
        raise NotImplementedError
