"""
hardware_monitor.py — Hardware Monitor

Background thread that samples GPU/CPU/RAM every 500ms.
Auto-detects GPU, CPU, RAM, CUDA version, and Salad Cloud node ID.
Tracks VRAM/RAM per task with before/after/peak measurements.
"""

import os
import time
import threading
import platform

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class HardwareMonitor:
    """Monitors GPU, CPU, and RAM usage with background sampling."""

    def __init__(self):
        self._gpu_available = False
        self._gpu_handle = None
        self._monitoring = False
        self._monitor_thread = None
        self._samples = []
        self._lock = threading.Lock()

        # Initialize NVML for GPU monitoring
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._gpu_available = True
            except Exception:
                self._gpu_available = False

    def get_system_info(self):
        """Return static hardware info dict (called once at startup)."""
        info = {
            "salad_node_id": os.environ.get("SALAD_NODE_ID", None),
            "cpu_model": platform.processor() or "Unknown",
            "cpu_cores": os.cpu_count() or 0,
            "ram_total_mb": round(psutil.virtual_memory().total / (1024 ** 2)) if PSUTIL_AVAILABLE else None,
            "os": f"{platform.system()} {platform.release()}",
            "gpu_model": None,
            "gpu_vram_total_mb": None,
            "cuda_version": None,
        }

        if self._gpu_available:
            try:
                info["gpu_model"] = pynvml.nvmlDeviceGetName(self._gpu_handle)
                if isinstance(info["gpu_model"], bytes):
                    info["gpu_model"] = info["gpu_model"].decode("utf-8")
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                info["gpu_vram_total_mb"] = round(mem_info.total / (1024 ** 2))
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                info["cuda_version"] = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
            except Exception:
                pass

        return info

    def snapshot(self):
        """Take a single VRAM/RAM/GPU utilization reading."""
        snap = {
            "timestamp": time.time(),
            "vram_used_mb": None,
            "vram_free_mb": None,
            "gpu_utilization_percent": None,
            "gpu_temp_c": None,
            "ram_used_mb": None,
        }

        if self._gpu_available:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                snap["vram_used_mb"] = round(mem_info.used / (1024 ** 2))
                snap["vram_free_mb"] = round(mem_info.free / (1024 ** 2))

                utilization = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                snap["gpu_utilization_percent"] = utilization.gpu

                temp = pynvml.nvmlDeviceGetTemperature(self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                snap["gpu_temp_c"] = temp
            except Exception:
                pass

        if PSUTIL_AVAILABLE:
            try:
                snap["ram_used_mb"] = round(psutil.virtual_memory().used / (1024 ** 2))
            except Exception:
                pass

        return snap

    def start_monitoring(self, interval_ms=500):
        """Start background sampling thread."""
        self._monitoring = True
        self._samples = []

        def _sample_loop():
            while self._monitoring:
                snap = self.snapshot()
                with self._lock:
                    self._samples.append(snap)
                time.sleep(interval_ms / 1000.0)

        self._monitor_thread = threading.Thread(target=_sample_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop background thread."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)

    def get_report(self):
        """Return full hardware report dict from collected samples."""
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return {
                "vram_peak_mb": None,
                "vram_min_mb": None,
                "gpu_utilization_avg_percent": None,
                "gpu_utilization_max_percent": None,
                "gpu_utilization_min_percent": None,
                "gpu_temp_max_c": None,
                "ram_peak_mb": None,
                "sample_count": 0,
            }

        vram_values = [s["vram_used_mb"] for s in samples if s["vram_used_mb"] is not None]
        gpu_util_values = [s["gpu_utilization_percent"] for s in samples if s["gpu_utilization_percent"] is not None]
        gpu_temp_values = [s["gpu_temp_c"] for s in samples if s["gpu_temp_c"] is not None]
        ram_values = [s["ram_used_mb"] for s in samples if s["ram_used_mb"] is not None]

        return {
            "vram_peak_mb": max(vram_values) if vram_values else None,
            "vram_min_mb": min(vram_values) if vram_values else None,
            "gpu_utilization_avg_percent": round(sum(gpu_util_values) / len(gpu_util_values), 1) if gpu_util_values else None,
            "gpu_utilization_max_percent": max(gpu_util_values) if gpu_util_values else None,
            "gpu_utilization_min_percent": min(gpu_util_values) if gpu_util_values else None,
            "gpu_temp_max_c": max(gpu_temp_values) if gpu_temp_values else None,
            "ram_peak_mb": max(ram_values) if ram_values else None,
            "sample_count": len(samples),
        }

    def get_vram_used_mb(self):
        """Quick single read of current VRAM used (for before/after comparisons)."""
        if self._gpu_available:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                return round(mem_info.used / (1024 ** 2))
            except Exception:
                return None
        return None

    def cleanup(self):
        """Shutdown NVML."""
        self.stop_monitoring()
        if PYNVML_AVAILABLE and self._gpu_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
